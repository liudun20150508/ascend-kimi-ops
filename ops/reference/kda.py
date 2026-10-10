"""KDA（Kimi Delta Attention）参考实现 —— 正确性金标准。

从 MoonshotAI/Kimi-K3 的 modeling 源码 + flash-linear-attention 的 chunk_kda
逆向出的完整数学。KDA 本质是 **Gated Delta Rule（门控增量规则）**：

    递推（逐 token，head 维度串行）:
        S_t = S_{t-1} * exp(g_t) + β_t · k_t · (v_t - k_tᵀ · S_{t-1})ᵀ
        o_t = q_tᵀ · S_t

    其中 S ∈ R^{K×V} 是定长递归状态（K=head_dim, V=head_dim）。

KIMI3 调用 chunk_kda 时启用了三个 in-kernel 预处理（modeling 源码参数），
它们是算子语义的一部分，参考实现必须包含：

1. q/k L2 norm（head_dim 维，eps=1e-6）
2. gate 构建（safe_gate=True, gate_lower_bound=-5.0）:
       g = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))
3. beta sigmoid:
       beta = sigmoid(beta_raw)

shape 约定（KIMI3: H=96, K=128, V=128, G=1 即 num_k_heads == num_heads）:
    q, k : [B, T, H, K]
    v    : [B, T, H, V]
    g_raw: [B, T, H, K]   （f_b_proj(f_a_proj(x)) 输出）
    beta_raw: [B, T, H]   （b_proj(x) 输出）
    A_log  : [H]          （log 空间，初始 log(uniform(1,16))）
    dt_bias: [H, K]       （reshape 自 [H*K]）

本文件只依赖 torch，不依赖 fla / einops，可在 CPU/NPU 上直接跑随机数据。
"""

import torch
import torch.nn.functional as F


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """在最后一维（head_dim）做 L2 归一化，与 fla.modules.l2norm 等价。"""
    return x * torch.rsqrt(torch.sum(x.float() * x.float(), dim=-1, keepdim=True) + eps)


def kda_gate(g_raw: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor,
             lower_bound: float = -5.0) -> torch.Tensor:
    """safe_gate=True 的 gate 构建。

    g_raw: [B, T, H, K], A_log: [H], dt_bias: [H, K]（或 [H*K] 会自动 view）
    返回 log 空间 per-token decay: [B, T, H, K]
    """
    g = g_raw.float()
    if dt_bias.numel() == g.shape[-2] * g.shape[-1]:
        dt_bias = dt_bias.view(g.shape[-2], g.shape[-1])
    g = g + dt_bias  # broadcast [H, K]
    if A_log is not None:
        g = A_log.view(-1, 1).float().exp() * g  # exp(A_log) ∈ [1,16]
    g = lower_bound * torch.sigmoid(g)
    return g.to(g_raw.dtype)


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_raw: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float = -5.0,
    use_qk_l2norm: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid: bool = True,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
):
    """KDA 逐 token 串行递推（正确性基准，慢但精确）。

    与 modeling 源码调用 chunk_kda 的参数语义一一对应。
    """
    dtype = v.dtype
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    q, k, v = [x.float() for x in (q, k, v)]
    g = g_raw.float()
    beta = beta_raw.float()

    # 1. q/k L2 norm
    if use_qk_l2norm:
        q = l2norm(q)
        k = l2norm(k)

    # 2. beta sigmoid
    if use_beta_sigmoid:
        beta = torch.sigmoid(beta)

    # 3. gate 构建（log-space per-token decay）
    if use_gate_in_kernel:
        g = kda_gate(g_raw, A_log, dt_bias, lower_bound=lower_bound)
    # else: g 已是构建好的 log-space decay

    # GVA 展开：q/k 从 [B,T,H,K] 复制到 value head 数
    q = q.repeat_interleave(G, dim=2) * scale   # [B,T,HV,K]
    k = k.repeat_interleave(G, dim=2)           # [B,T,HV,K]

    S = k.new_zeros(B, HV, K, V)
    if initial_state is not None:
        S = S + initial_state
    o = torch.zeros(B, T, HV, V, dtype=torch.float32, device=v.device)

    for i in range(T):
        q_i = q[:, i]                 # [B, HV, K]
        k_i = k[:, i]                 # [B, HV, K]
        v_i = v[:, i]                 # [B, HV, V]
        g_i = g[:, i]                 # [B, HV, K]
        b_i = beta[:, i]              # [B, HV]

        # S = S * exp(g) + beta * k (v - k^T S)^T
        S = S * g_i[..., None].exp()
        kS = torch.einsum('bhk,bhkv->bhv', k_i, S)          # [B,HV,V]
        delta = v_i - kS                                       # [B,HV,V]
        S = S + torch.einsum('bh,bhk,bhv->bhkv', b_i, k_i, delta)
        o[:, i] = torch.einsum('bhk,bhkv->bhv', q_i, S)

    o = o.to(dtype)
    if not output_final_state:
        S = None
    return o, S
