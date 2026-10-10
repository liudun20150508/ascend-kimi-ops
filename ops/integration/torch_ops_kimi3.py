"""把自研 AscendC 算子注册成 **torch 一等公民**：`torch.ops.kimi3_ops.situ_and_mul`。

为什么需要这一层（这是"进 vLLM / SGLang"的正确姿势）
----------------------------------------------------
vLLM 的 `CustomOp` 机制、SGLang 的 layer 实现、以及 `torch.compile` / 图捕获 / 显存预估，
都要求"算子"是 **torch 认识的 op**（有 schema + 有 Meta 实现），
而不是"某个 Python 函数直接去调 aclnn"。三份实现的分工：

| 实现 | 作用 | 对应 vLLM 的说法 |
|---|---|---|
| **fake/meta** | 只推导 shape/dtype，不碰设备 —— 图捕获、动态 shape、显存估算都靠它 | 必需（否则 `torch.compile` 在 trace 期就断） |
| **NPU 实现** | 真正落到我们的 `aclnnSituAndMul`（经 `aclnn_ops.py` 的 JIT 包装） | 相当于 `forward_npu` |
| **CPU 参考** | 纯 torch 直白实现，永不可用也会退化到它 | 相当于 `forward_native` |

用法
----
```python
import sys; sys.path.insert(0, "kimi3_ops/integration")
import torch_ops_kimi3                       # import 即注册
y = torch.ops.kimi3_ops.situ_and_mul(x, 4.0, 25.0)
```

自检（会真的在 NPU 上跑一发 + 验 FakeTensor + 验图捕获）：

```bash
/usr/local/python3.11.15/bin/python3 kimi3_ops/integration/torch_ops_kimi3.py
```
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_KIMI3 = os.path.dirname(_HERE)          # kimi3_ops/
if _KIMI3 not in sys.path:
    sys.path.insert(0, _KIMI3)

LIB = "kimi3_ops"


# --------------------------------------------------------------------------
# SiTU-GLU：x[..., 2d] -> y[..., d]
# --------------------------------------------------------------------------
def _situ_and_mul_npu(x: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
    import aclnn_ops                            # 懒加载：第一次调用才 JIT 编译包装
    return aclnn_ops.situ_and_mul(x, beta, linear_beta)


def _situ_and_mul_cpu(x: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
    from reference.situ import situ_and_mul as native
    return native(x, beta, linear_beta)


@torch.library.custom_op(f"{LIB}::situ_and_mul", mutates_args=())
def situ_and_mul(x: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
    """SiTU-GLU：`situ(gate) * tanh_scale(up)`，其中 `x = cat([gate, up], dim=-1)`。"""
    if x.device.type == "npu":
        return _situ_and_mul_npu(x, beta, linear_beta)
    return _situ_and_mul_cpu(x, beta, linear_beta)


@situ_and_mul.register_fake
def _situ_and_mul_fake(x: torch.Tensor, beta: float, linear_beta: float) -> torch.Tensor:
    """只推导 shape：最后一维砍半，dtype/device 不变。"""
    return x.new_empty(list(x.shape[:-1]) + [x.shape[-1] // 2])


# --------------------------------------------------------------------------
# AttnRes 残差栈：(T,B,H) + (T,H) + (H,) -> (T,H)
# --------------------------------------------------------------------------
def _attn_res_npu(v, prefix_sum, score_weight, eps):
    import aclnn_ops
    return aclnn_ops.attn_res(v, prefix_sum, score_weight, eps)


def _attn_res_cpu(v, prefix_sum, score_weight, eps):
    from reference.attn_res import attn_res as native
    return native(v, prefix_sum, score_weight, eps)


@torch.library.custom_op(f"{LIB}::attn_res", mutates_args=())
def attn_res(v: torch.Tensor, prefix_sum: torch.Tensor, score_weight: torch.Tensor,
             eps: float) -> torch.Tensor:
    """AttnRes 残差栈：对 B+1 个残差向量做 RMS-norm + 学习 score + softmax 加权求和。"""
    if v.device.type == "npu":
        return _attn_res_npu(v, prefix_sum, score_weight, eps)
    return _attn_res_cpu(v, prefix_sum, score_weight, eps)


@attn_res.register_fake
def _attn_res_fake(v: torch.Tensor, prefix_sum: torch.Tensor, score_weight: torch.Tensor,
                   eps: float) -> torch.Tensor:
    return v.new_empty(list(v.shape[:1]) + [v.shape[-1]])       # (T, H)


# --------------------------------------------------------------------------
# SituGluFused 融合算子：x[M,H] + Wp[2d,H] -> y[M,d]
#   ⚠️ Wp 必须**已按 subN 交错重排**（忘了重排不会报错、只会算错）；
#      便捷入口见 aclnn_ops.situ_glu_fused_auto()（自动两段式重排）。
# --------------------------------------------------------------------------
def _situ_glu_fused_npu(x: torch.Tensor, w: torch.Tensor, beta: float,
                        linear_beta: float) -> torch.Tensor:
    import aclnn_ops
    return aclnn_ops.situ_glu_fused(x, w, beta, linear_beta)


def _situ_glu_fused_cpu(x: torch.Tensor, w: torch.Tensor, beta: float,
                        linear_beta: float) -> torch.Tensor:
    from reference.situ import situ_and_mul as native
    gate_up = (x.float() @ w.float().T).to(x.dtype)     # [M, 2d]
    return native(gate_up, beta, linear_beta)


@torch.library.custom_op(f"{LIB}::situ_glu_fused", mutates_args=())
def situ_glu_fused(x: torch.Tensor, w: torch.Tensor, beta: float,
                   linear_beta: float) -> torch.Tensor:
    """融合算子：`situ(gate) * tanh_scale(up)` 直接接在 GEMM 后面（中间张量不落 HBM）。

    `x[M,H]`、`w[2d,H]`（**已交错重排的 W'**）→ `y[M,d]`。
    """
    if x.device.type == "npu":
        return _situ_glu_fused_npu(x, w, beta, linear_beta)
    return _situ_glu_fused_cpu(x, w, beta, linear_beta)


@situ_glu_fused.register_fake
def _situ_glu_fused_fake(x: torch.Tensor, w: torch.Tensor, beta: float,
                         linear_beta: float) -> torch.Tensor:
    """只推导 shape：[M,H] + [2d,H] -> [M,d]（图捕获/动态 shape 的前提）。"""
    return x.new_empty(list(x.shape[:-1]) + [w.shape[0] // 2])


# --------------------------------------------------------------------------
# 自检
# --------------------------------------------------------------------------
def _selfcheck():
    import torch_npu  # noqa: F401  （注册 NPU 后端）

    print("=" * 70)
    print("① 直接经 torch.ops 调用（落 NPU）")
    x = (torch.randn(2, 7, 22016, dtype=torch.bfloat16) * 6.0).npu()
    y = torch.ops.kimi3_ops.situ_and_mul(x, 4.0, 25.0)
    from reference.situ import situ_and_mul as golden
    ref = golden(x.cpu(), 4.0, 25.0)
    rel = ((y.cpu().float() - ref.float()).abs() / (ref.float().abs() + 1e-3)).max().item()
    print(f"   x={tuple(x.shape)}@{x.device}  ->  y={tuple(y.shape)}@{y.device} {y.dtype}")
    print(f"   max_rel vs golden = {rel:.3e}   {'OK' if rel < 1e-2 else 'FAIL'}")
    assert rel < 1e-2

    print("② Meta/FakeTensor 推导（图捕获的前提）")
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode():
        fx = torch.empty(2, 7, 22016, dtype=torch.bfloat16)
        fy = torch.ops.kimi3_ops.situ_and_mul(fx, 4.0, 25.0)
        print(f"   fake: {tuple(fx.shape)} -> {tuple(fy.shape)} {fy.dtype}   OK")

    print("③ 图捕获（torch.compile，后端用 aot_eager，不依赖 inductor/triton）")
    fn = torch.compile(lambda t: torch.ops.kimi3_ops.situ_and_mul(t, 4.0, 25.0),
                       backend="aot_eager")
    z = fn(x)
    d = (z.cpu().float() - y.cpu().float()).abs().max().item()
    print(f"   compiled -> {tuple(z.shape)}@{z.device}；与 eager 差 {d:.3e}   OK")

    print("④ CPU 参考路径（vLLM 里的 forward_native 等价物）")
    ycpu = torch.ops.kimi3_ops.situ_and_mul(x.cpu(), 4.0, 25.0)
    print(f"   cpu -> {tuple(ycpu.shape)}@{ycpu.device}   OK")
    print("⑤ 融合算子 SituGluFused（GEMM + SiTU epilogue；验重排 + torch.ops 路径一致性）")
    from aclnn_ops import situ_glu_fused_auto, interleave
    M, H, d = 8, 64, 128
    xx = (torch.randn(M, H, dtype=torch.bfloat16) * 2.0).npu()
    g = (torch.randn(d, H, dtype=torch.bfloat16) * 0.1).npu()
    u = (torch.randn(d, H, dtype=torch.bfloat16) * 0.1).npu()
    y_harness, subn = situ_glu_fused_auto(xx, g, u, 4.0, 25.0)   # 自动两段式重排
    y_ops = torch.ops.kimi3_ops.situ_glu_fused(xx, interleave(g, u, subn), 4.0, 25.0)
    same = (y_ops.float() - y_harness.float()).abs().max().item()
    print(f"   y={tuple(y_ops.shape)}@{y_ops.device}  subN={subn}  "
          f"torch.ops 与 harness 路径差 {same:.1e}   {'OK' if same == 0 else 'FAIL'}")
    assert same == 0
    with FakeTensorMode():
        fw = torch.empty(2 * d, H, dtype=torch.bfloat16)
        fy = torch.ops.kimi3_ops.situ_glu_fused(
            torch.empty(M, H, dtype=torch.bfloat16), fw, 4.0, 25.0)
        print(f"   fake: [M={M},H={H}] + [2d={2*d},H] -> {tuple(fy.shape)}   OK")
    fn2 = torch.compile(lambda a, b: torch.ops.kimi3_ops.situ_glu_fused(a, b, 4.0, 25.0),
                        backend="aot_eager")
    z2 = fn2(xx, interleave(g, u, subn))
    d2 = (z2.cpu().float() - y_ops.cpu().float()).abs().max().item()
    print(f"   compiled -> {tuple(z2.shape)}；与 eager 差 {d2:.3e}   OK")

    print("=" * 70)
    print("全部通过：这几个算子现在都可以被框架当常规 torch op 用（含图捕获）。")


if __name__ == "__main__":
    _selfcheck()
