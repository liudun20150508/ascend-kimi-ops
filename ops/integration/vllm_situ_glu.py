"""把 SiTU-GLU 接进 **vLLM（昇腾）** —— 两处改动，**不碰 vLLM 主干**。

为什么是"模型层接线"而不是"注册成 vLLM 通用算子"
------------------------------------------------
* vLLM 的通用算子（`SiluAndMul`、`RMSNorm`、`RotaryEmbedding`…）都是 `CustomOp` 子类，按 platform
  dispatch：`forward_native` / `forward_cuda` / `forward_npu`。
* **SiTU-GLU 不是 vLLM 的算子**（vLLM 只有 SiLU 融合）⇒ 它**没有挂载点**可补 `forward_npu`，
  唯一正确的位置是 **KIMI3 的模型实现文件**（vllm-ascend 或你自己的 fork）。
  这就是 `docs/12` 里说的"③b：在模型文件里接"，而不是"注册成通用算子"。

两处改动
--------
① 注册算子（一行 import，`torch.ops.kimi3_ops.*` 由此可用）：

```python
from kimi3_ops.integration import torch_ops_kimi3   # noqa: F401
```

② 模型里换掉激活（KIMI3 的 MLP；其余一字不改）：

```python
# vLLM 侧 KIMI3 模型文件（如 vllm/model_executor/models/kimi_k3.py 或 vllm-ascend 的对应文件）
# 原本：
#     gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
#     out     = self.down_proj(self.act_fn(gate_up))
self.act_fn = SituAndMulModule(beta=4.0, linear_beta=25.0)   # ← 只换这一行
```

> 输入布局天然对齐：vLLM 的 `gate_up_proj` 输出就是 `[..., 2d]`（gate 在前、up 在后），
> 与我们 kernel 的输入约定完全一致 —— 不需要额外 cat/split。

自检（不需要 vLLM、不需要权重，随机数据即可）：

```bash
/usr/local/python3.11.15/bin/python3 kimi3_ops/integration/vllm_situ_glu.py
```

端到端验证清单（本机缺 vLLM 与权重，标注了可做/不可做）
------------------------------------------------------
| 项 | 本机可做？ | 做法 |
|---|---|---|
| 数值一致性（native vs npu 算子） | ✅ | 本文件 `_selfcheck()` |
| 单层 MLP latency 对比（替换前/后） | ✅ | 随机权重跑两层 Linear + 激活，比 t |
| 与 HF 参考输出端到端一致 | ⛔ | 需权重 |
| TTFT / TPOT / 吞吐 | ⛔ | 需权重 + vLLM |
| 长上下文 / EP64 均衡 | ⛔ | 需多卡 + 权重 |
"""

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_KIMI3 = os.path.dirname(_HERE)
if _KIMI3 not in sys.path:
    sys.path.insert(0, _KIMI3)

try:
    from torch_ops_kimi3 import LIB  # noqa: F401
except ImportError:                  # 作为脚本直接跑时的路径差异
    sys.path.insert(0, _HERE)
    from torch_ops_kimi3 import LIB  # noqa: F401


class SituAndMulModule(nn.Module):
    """KIMI3 的 SiTU-GLU 激活，可直接替换 MLP 里的 `self.act_fn`。

    与 `vllm.model_executor.layers.activation` 里的 C 类同名结构一致：
    输入 `[..., 2d]`（gate 在前、up 在后），输出 `[..., d]`。
    内部走 `torch.ops.kimi3_ops.situ_and_mul`（已注册 fake/meta ⇒ 图捕获不会断）。
    """

    def __init__(self, beta: float = 4.0, linear_beta: float = 25.0):
        super().__init__()
        self.beta = float(beta)
        self.linear_beta = float(linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.kimi3_ops.situ_and_mul(x, self.beta, self.linear_beta)


class MergedSituGluMlp(nn.Module):
    """（为**融合版**预留的接口）一颗 GEMM 出 `[.., 2d]` + SiTU epilogue，中间张量不落 HBM。

    现状（2026-09-19 更新）：融合算子**已真机跑通**（4 组 shape 数值 PASS，`docs/14` §16.4），
    并且已注册成 `torch.ops.kimi3_ops.situ_glu_fused`（含 fake/meta + 图捕获）。
    ⇒ 这里可以从"两段式"切到"融合版"：把 `forward` 里两行合成一次算子调用即可，**调用点不用改**。

    ⚠️ 接融合版前必须处理**权重重排**：算子要求 weight 是 `W'`（按 subN 交错重排），
       而 subN 由 host tiling 决定 ⇒ 见 `../aclnn_ops.py` 的 `situ_glu_fused_auto()`（两段式自动重排）。
       忘了重排**不会报错、只会算错**。
    """

    def __init__(self, hidden_size: int, intermediate_size: int, beta: float = 4.0,
                 linear_beta: float = 25.0, dtype=torch.bfloat16):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.act_fn = SituAndMulModule(beta, linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ① 两段式（当前）：gate_up → SiTU-GLU（我们的算子）→ down
        gate_up = self.gate_up_proj(x)
        return self.down_proj(self.act_fn(gate_up))
        # ② 融合版（待 SituGluFused 跑通）：让 gate_up_proj 的 epilogue 直接算 SiTU-GLU


def _selfcheck():
    """不需要 vLLM / 权重：随机数据下比"torch 组合（native 等价物）"与"我们的 NPU 算子"。"""
    import torch_npu  # noqa: F401
    from reference.situ import situ_and_mul as golden

    print("=" * 70)
    beta, lin = 4.0, 25.0
    x = (torch.randn(8, 512, 22016, dtype=torch.bfloat16) * 6.0).npu()

    # ① 等价性判据（公平）：native 用 **fp32 中间量**，与算子内部一致，最后统一 cast bf16
    xf = x.float()
    g32, u32 = xf.chunk(2, dim=-1)
    native_f32 = (beta * torch.tanh(g32 / beta) * torch.sigmoid(g32)
                  * (lin * torch.tanh(u32 / lin))).to(x.dtype)

    # ② 信息项：**真实 vLLM 的 native 路径是逐算子 bf16**（~6 次舍入），比我们的算子粗糙
    gb, ub = x.chunk(2, dim=-1)
    native_bf16 = beta * torch.tanh(gb / beta) * torch.sigmoid(gb) * (lin * torch.tanh(ub / lin))

    m = SituAndMulModule(beta, lin).to(x.device)      # 替换后的实现
    out = m(x)

    def nbad(a, b, atol=1e-2, rtol=1e-2):
        """标准判据 ±(atol + rtol·|ref|)。

        别用纯相对误差：|ref|≈0 处会把 bf16 的正常绝对误差放大成假失败
        （docs/14 §10 有实测记录；§14 又复现一次）。
        """
        over = (a.float() - b.float()).abs().gt(atol + rtol * b.float().abs())
        return int(over.sum().item()), a.numel()

    ref = golden(x.cpu(), beta, lin)
    bad_eq, n = nbad(out, native_f32)
    bad_golden, _ = nbad(out.cpu(), ref)
    bad_bf16, _ = nbad(out, native_bf16)

    print(f"  shape {tuple(x.shape)} @ {x.device}  ->  {tuple(out.shape)}")
    print(f"  vs native(fp32 中间量, 公平对照) : 超容差 {bad_eq} / {n} 元素")
    print(f"  vs golden(CPU 直白实现)         : 超容差 {bad_golden} / {n} 元素")
    print(f"  [信息] vs native(真实 vLLM 的逐算子 bf16 路径) : 超容差 {bad_bf16} / {n} 元素")
    ok = (bad_eq == 0) and (bad_golden == 0)
    print("  =>", "OK：替换后与 vLLM 原生实现（同精度）及 CPU golden 完全一致" if ok else "FAIL")
    print("  注：逐算子 bf16 的基线本身有 ~6 次舍入，比算子内的 fp32 内核粗糙 ——"
          "\n      也就是说换成我们的算子，精度反而更稳。")
    print("=" * 70)
    return ok


if __name__ == "__main__":
    sys.exit(0 if _selfcheck() else 1)
