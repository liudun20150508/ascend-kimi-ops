"""AttnRes 残差栈参考实现（正确性金标准）。

从 MoonshotAI/Kimi-K3 的 modeling_kimi_linear.py 中 `_apply_attn_res` 逐行提取。
本质是「mini-attention」：B+1 个残差向量做 RMS-norm + 学到的 score 点积 + softmax + 加权求和。
"""

import torch


def attn_res(
    block_residual: torch.Tensor,
    prefix_sum: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """AttnRes 融合算子。

    block_residual: (T, B, H)  bf16 —— 之前各 block 的残差
    prefix_sum:     (T, H)     bf16 —— 当前层残差
    score_weight:   (H,)       bf16 —— = norm.weight * proj.weight.squeeze(0)（学到的常量）
    eps:            float      —— rms_norm_eps
    返回: (T, H) bf16
    """
    # v = [block_residual; prefix_sum[:, None]]  -> (T, B+1, H)
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_float = v.float()
    # RMS norm（H 维）
    variance = v_float.pow(2).mean(-1, keepdim=True)      # (T, B+1, 1)
    k = v_float * torch.rsqrt(variance + eps)              # (T, B+1, H)
    # 学到的 score：sum_h k * score_weight
    scores = (k * score_weight.float()).sum(-1)            # (T, B+1)
    # softmax 在 B+1 维 + 加权求和
    probs = scores.softmax(-1).unsqueeze(1)                # (T, 1, B+1)
    out = torch.matmul(probs, v_float).squeeze(1)          # (T, H)
    return out.to(v.dtype)
