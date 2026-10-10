"""把 SiTU-GLU 接进 **SGLang（昇腾）** —— 与 vLLM 一一对应，机制完全相同。

映射关系（`docs/12` §5 那张表的代码版）
--------------------------------------
| vLLM | SGLang | 本文件对应 |
|---|---|---|
| `CustomOp.forward_npu` | `sglang/srt/layers/activation.py` 里的层类 | `SituAndMulModule` |
| `vllm/platforms/` | `sglang/srt/hardware_backend/` | 不需要改（SGLang 已按设备分发） |
| Model Registry | `sglang/srt/models/*.py` + `model_config` | KIMI3 模型文件里的 `act_fn` |

> 结论同 vLLM：**SiTU-GLU 不是 SGLang 的既有算子**（它只有 `SiluAndMul`），
> 所以接入点必然在 **KIMI3 的模型实现**里，而不是"注册成通用算子"。

两处改动
--------
① 注册（一行）：

```python
from kimi3_ops.integration import torch_ops_kimi3   # noqa: F401
```

② `sglang/srt/models/kimi_k3.py`（或你的 fork）里的 MLP：

```python
from kimi3_ops.integration.sglang_situ_glu import SituAndMulModule

class KimiMLP(nn.Module):
    def __init__(self, ...):
        ...
        self.act_fn = SituAndMulModule(beta=4.0, linear_beta=25.0)   # ← 换这一行
    def forward(self, x):
        gate_up = torch.cat([self.gate_proj(x), self.up_proj(x)], dim=-1)
        return self.down_proj(self.act_fn(gate_up))
```

自检：

```bash
/usr/local/python3.11.15/bin/python3 kimi3_ops/integration/sglang_situ_glu.py
```
"""

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_KIMI3 = os.path.dirname(_HERE)
for p in (_KIMI3, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch_ops_kimi3  # noqa: E402,F401  ← 必须 import 才会注册 torch.ops.kimi3_ops.*


class SituAndMulModule(nn.Module):
    """SGLang 侧的 SiTU-GLU 层，可直接放进 KIMI3 模型实现的 MLP。

    对照 SGLang 内置的 `SiluAndMul`：同样是"输入 `[..., 2d]`（gate/up 拼接）→ 输出 `[..., d]`"，
    只是激活函数从 `silu(gate) * up` 换成 KIMI3 的 `situ(gate) * tanh_scale(up)`。
    """

    def __init__(self, beta: float = 4.0, linear_beta: float = 25.0):
        super().__init__()
        self.beta = float(beta)
        self.linear_beta = float(linear_beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.kimi3_ops.situ_and_mul(x, self.beta, self.linear_beta)

    # SGLang 里有些层会带 quant 方法（与算子无关，这里只做占位，避免假接口）
    def extra_repr(self) -> str:
        return f"beta={self.beta}, linear_beta={self.linear_beta}"


def _selfcheck():
    import torch_npu  # noqa: F401
    from reference.situ import situ_and_mul as golden

    print("=" * 70)
    beta, lin = 4.0, 25.0
    x = (torch.randn(4, 1024, 22016, dtype=torch.bfloat16) * 6.0).npu()
    m = SituAndMulModule(beta, lin).to(x.device)
    out = m(x)

    xf = x.float()
    g32, u32 = xf.chunk(2, dim=-1)
    native = (beta * torch.tanh(g32 / beta) * torch.sigmoid(g32)
              * (lin * torch.tanh(u32 / lin))).to(x.dtype)

    def nbad(a, b, atol=1e-2, rtol=1e-2):
        return int((a.float() - b.float()).abs().gt(atol + rtol * b.float().abs()).sum().item())

    ref = golden(x.cpu(), beta, lin)
    print(f"  {tuple(x.shape)} @ {x.device} -> {tuple(out.shape)}")
    print(f"  超容差 vs native(fp32 公平对照) : {nbad(out, native)} / {out.numel()}")
    print(f"  超容差 vs golden(CPU)          : {nbad(out.cpu(), ref)} / {out.numel()}")
    print("=" * 70)
    return nbad(out, native) == 0 and nbad(out.cpu(), ref) == 0


if __name__ == "__main__":
    sys.exit(0 if _selfcheck() else 1)
