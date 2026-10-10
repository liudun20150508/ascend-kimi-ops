"""SiTU-GLU 激活函数的参考实现（正确性金标准）。

直接取自 MoonshotAI/Kimi-K3 的 modeling_kimi_linear.py 中 `SituAndMul`，
数学定义（KIMI3 配置 activation_situ_beta=4.0, activation_situ_linear_beta=25.0）：

    gate, up = x[..., :d], x[..., d:]      # d = intermediate_size
    situ(gate) = beta * tanh(gate / beta) * sigmoid(gate)
    up'        = linear_beta * tanh(up / linear_beta)
    out        = situ(gate) * up'

注意：SiTU 与 SiLU 不同 —— SiLU(x) = x * sigmoid(x)，而 SiTU 是
sigmoid-tanh 的乘积形式，且 up 分支也被 linear_beta 做 tanh 压缩。

参考实现内部在 float32 下计算，最后转回输入 dtype（bf16），这是算子
数值对齐时的关键容差依据：fp32 内部 → bf16 输出。
"""

import torch


def situ_and_mul(
    x: torch.Tensor,
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
) -> torch.Tensor:
    """SiTU-GLU 融合激活。输入 x 形状 [..., 2*d]，输出 [..., d]。

    与 modeling 源码 `SituAndMul.forward` 逐行等价，仅去掉了 nn.Module 包装。
    """
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)


def situ_glu(
    x: torch.Tensor,
    beta: float = 4.0,
    linear_beta: float | None = 25.0,
) -> torch.Tensor:
    """SiTU-GLU 别名（与 situ_and_mul 相同，语义更贴合 MLP 的 gate/up 用法）。"""
    return situ_and_mul(x, beta=beta, linear_beta=linear_beta)
