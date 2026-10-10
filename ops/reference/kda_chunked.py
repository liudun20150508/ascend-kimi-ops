"""KDA 分块（chunk）参考实现 —— 严格按华为 chunk_kda_fwd README 的数学公式。

与 `kda.naive_recurrent_kda`（逐 token 递推）是**同一数学**的两种组织方式：
分块算法把因果结构拆成「chunk 内下三角求逆 + chunk 间状态传播」，等价但更适合 Cube。

本文件只依赖 torch，用于交叉验证华为算子的分块公式是否 == KDA 递推。

公式（per chunk c，省略 batch/head；i,j 为 chunk 内 token，l_c 为 chunk 末 token）：
    gamma[i,d]     = lower_bound * sigmoid(exp(A_log) * (g[i,d] + dt_bias[d]))   # 已激活 log-space gate
    E[i,d]         = exp(cumsum(gamma)[i,d])                                      # = 2^{gk[i,d]}
    qg[i,d]        = q[i,d] * E[i,d]
    kg[i,d]        = k[i,d] * E[l_c,d] / E[i,d]
    k_hat[i,d]     = k[i,d] / E[i,d]
    k_tilde[i,d]   = k[i,d] * E[i,d]
    Aqk[i,j]       = 1[j<=i] * scale * sum_d qg[i,d] * k_hat[j,d]
    L[i,j]         = 1[j<i]  * beta[i] * sum_d k_tilde[i,d] * k_hat[j,d]
    Akk            = (I + L)^{-1}
    w_seed[i,d]    = beta[i] * k_tilde[i,d]
    u_seed[i,e]    = beta[i] * v[i,e]
    w = Akk @ w_seed ; u = Akk @ u_seed
    v_new          = u - w @ h_c
    h_{c+1}[k,e]   = E[l_c,k] * h_c[k,e] + (kg^T @ v_new)[k,e]
    attn_out_c[i,e]= scale * (qg @ h_c)[i,e] + (Aqk @ v_new)[i,e]

语义与华为算子一致：**gate 在 kernel 内做**（safe_gate）；**不做 q/k L2 norm、不做 beta sigmoid**
（这两项由调用方在外部完成，见 modeling_kimi_linear.py 的 use_qk_l2norm_in_kernel / use_beta_sigmoid_in_kernel）。

计算 dtype：bf16/fp16 输入提升到 fp32（对齐华为 kernel 的 FP32 累积）；fp64 输入保持 fp64
（用于精确性交叉验证）。
"""

import torch


def chunk_kda_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    chunk_size: int = 64,
    lower_bound: float = -5.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
):
    """分块 KDA，与华为 chunk_kda_fwd 数学一致（不内建 l2norm / beta sigmoid）。

    q,k: [B,T,H,K]；v: [B,T,HV,V]；g_raw: [B,T,HV,K]；beta: [B,T,HV]
    A_log: [HV]；dt_bias: [HV,K]（或 [HV*K] 自动 view）。
    """
    dtype = v.dtype
    cdtype = torch.promote_types(v.dtype, torch.float32)
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K ** -0.5

    q = q.to(cdtype).repeat_interleave(G, dim=2)   # [B,T,HV,K]
    k = k.to(cdtype).repeat_interleave(G, dim=2)   # [B,T,HV,K]
    v = v.to(cdtype)                                # [B,T,HV,V]
    g = g_raw.to(cdtype)                            # [B,T,HV,K]
    beta = beta.to(cdtype)                          # [B,T,HV]

    if dt_bias.numel() == HV * K:
        dt_bias = dt_bias.view(HV, K)
    dt_bias = dt_bias.to(cdtype)

    # gate 构建（safe_gate=true）：gamma = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))
    gamma = g + dt_bias.view(1, 1, HV, K)                        # [B,T,HV,K]
    gamma = A_log.view(1, 1, HV, 1).to(cdtype).exp() * gamma     # exp(A_log) 广播到 HV
    gamma = lower_bound * torch.sigmoid(gamma)                   # [B,T,HV,K]

    h = torch.zeros(B, HV, K, V, dtype=cdtype, device=v.device)
    if initial_state is not None:
        h = h + initial_state.to(cdtype)

    o = torch.zeros(B, T, HV, V, dtype=cdtype, device=v.device)
    num_chunks = (T + chunk_size - 1) // chunk_size

    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, T)

        # 统一到 [B,HV,C,...] 轴序做 chunk-local 计算
        qc = q[:, start:end].transpose(1, 2)      # [B,HV,C,K]
        kc = k[:, start:end].transpose(1, 2)      # [B,HV,C,K]
        vc = v[:, start:end].transpose(1, 2)      # [B,HV,C,V]
        gc = gamma[:, start:end].transpose(1, 2)  # [B,HV,C,K]
        bc = beta[:, start:end].transpose(1, 2)   # [B,HV,C]

        E = torch.exp(torch.cumsum(gc, dim=2))    # [B,HV,C,K]
        E_last = E[:, :, -1]                       # [B,HV,K]

        qg = qc * E                                # [B,HV,C,K]
        k_hat = kc / E                             # [B,HV,C,K]
        k_tilde = kc * E                           # [B,HV,C,K]
        kg = kc * (E_last.unsqueeze(2) / E)        # [B,HV,C,K]

        # Aqk[b,h,i,j] = scale * sum_d qg[i,d] * k_hat[j,d]
        Aqk = scale * torch.tril(qg @ k_hat.transpose(-1, -2), diagonal=0)

        # L[b,h,i,j] = beta[i] * sum_d k_tilde[i,d] * k_hat[j,d]（严格下三角 j<i）
        kk_raw = k_tilde @ k_hat.transpose(-1, -2)  # [B,HV,C,C]
        L = bc.unsqueeze(-1) * torch.tril(kk_raw, diagonal=-1)

        I = torch.eye(end - start, dtype=cdtype, device=v.device)
        Akk = torch.linalg.inv(I + L)               # [B,HV,C,C]

        w_seed = bc.unsqueeze(-1) * k_tilde         # [B,HV,C,K]
        u_seed = bc.unsqueeze(-1) * vc              # [B,HV,C,V]

        w = Akk @ w_seed                            # [B,HV,C,K]
        u = Akk @ u_seed                            # [B,HV,C,V]

        v_new = u - w @ h                           # [B,HV,C,V]

        attn_c = scale * (qg @ h) + Aqk @ v_new     # [B,HV,C,V]

        o[:, start:end] = attn_c.transpose(1, 2)    # [B,C,HV,V]

        # h_next[k,e] = E_last[k] * h[k,e] + (kg^T @ v_new)[k,e]
        kgT_vnew = kg.transpose(-1, -2) @ v_new     # [B,HV,K,V]
        h = E_last.unsqueeze(-1) * h + kgT_vnew     # [B,HV,K,V]

    o = o.to(dtype)
    S = h.to(dtype) if output_final_state else None
    return o, S
