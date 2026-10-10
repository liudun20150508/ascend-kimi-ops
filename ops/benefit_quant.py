"""三项工作的收益量化（零依赖，本机可跑）。

    python3 benefit_quant.py
    python3 benefit_quant.py --d 11008 --h 4096 --B 8 --tokens 4096
    python3 benefit_quant.py --topk 8            # MoE 每 token 激活的专家数

量化三件事 —— 都是「华为没做 / 我做了」的部分：

  ① SiTU-GLU 融进上游 GEMM 的 epilogue（**待做**）
  ② AttnRes 内部融合：RMSNorm + proj 折成一次点积（**已做，且对拍验证过**）
  ③ LatentMoE 图接线（**使能类**，不能按带宽量化，单独说明）

★ 最重要的口径问题：**必须分 prefill / decode 两个阶段算**。
  · Prefill：一次算 N 个 token，省的访存是 N 倍 → 收益显著
  · Decode ：一次 1 个 token，而每步要读**整层权重**（几百 MB）→ 省的那点被淹没
  把两者混在一起说「融合快 X 倍」是错的，对方一追就露。

⚠️ 所有数字都是纸面推导（按公示规格 + 逐项字节数算），不是实测。
"""

import argparse

DTYPE_BYTES = {"bf16": 2, "fp16": 2}


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or u == "TB":
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


# ======================================================= ① SiTU-GLU epilogue 融合
def situ_fusion(H, d, tokens, dt, w_bytes=None):
    """返回 (每 token 省字节, prefill 省字节, prefill 原/新搬运, decode 对比)。"""
    d2 = 2 * d
    per_token = 4 * d * dt          # 省掉 gate_up 的「写 2d + 读 2d」

    # --- 该链路的搬运量（口径：gate_up GEMM + SiTU 这一段，不含 down_proj）---
    w = w_bytes if w_bytes else H * d2 * dt          # 权重 [W_gate;W_up]: H × 2d
    x_in = tokens * H * dt                            # 读输入 x
    gate_up = tokens * d2 * dt                        # gate_up 中间张量（写 + 读各一次）
    out = tokens * d * dt                             # 写输出
    before = x_in + w + gate_up + gate_up + out       # 原：多一次写 + 一次读
    after = x_in + w + out                            # 融合后：中间张量不落地
    return per_token, before, after


# ==================================================== ② AttnRes 内部融合（已做）
def attnres_fusion(H, B, tokens, dt):
    """RMSNorm + proj 折成点积：省掉 v_normed 的写+读 = 2H × (B+1) 每 token。"""
    nvec = B + 1                                      # B 个 block_residual + 1 个 prefix_sum
    per_token = 2 * H * nvec * dt
    # 原链路搬运：读 v + 写 v_normed + 读 v_normed + 写 score；融合后：读 v + 写 score
    v_read = tokens * nvec * H * dt
    v_normed = tokens * nvec * H * dt                 # 中间量（写+读各一次）
    score = tokens * nvec * 1 * dt
    before = v_read + v_normed + v_normed + score
    after = v_read + score
    return per_token, before, after, nvec


# ============================================== ③ LatentMoE：使能类，单独说明
def latentmoe_note():
    print("=" * 80)
    print("③ LatentMoE 图接线 —— 使能类收益，**不能按带宽量化**")
    print("=" * 80)
    print("""
  它在源码里的形态是「形状不符」，不是「性能不够」：
      down_proj(Linear H→moe_hidden) → MoE → [RMSNorm] → up_proj(Linear moe_hidden→H)
  MoE 那部分华为已有算子（npu_ops_transformer），两头两个 Linear 也是现成的 GEMM。

  所以它既不是「华为没做」，也不是「我优化了」—— 它是**接线**：
    · 不接线：模型形状对不上，根本跑不起来（0 → 1）
    · 接线：用现成算子拼，**不需要写任何新 kernel**

  收益怎么表述（表述用）：
    「这里最大的收益不是性能，是**避免返工**——如果把形状不符当成新算子去写，
      就是一个本来不需要的 AscendC 算子开发周期（参考本项目：一个算子从
      写 kernel 到对拍验证要几天）。判断出『这是接线问题』本身就是价值。」
""".rstrip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=4096, help="hidden_size")
    ap.add_argument("--d", type=int, default=11008, help="intermediate_size")
    ap.add_argument("--B", type=int, default=8, help="attn_res_block_size（残差块数）")
    ap.add_argument("--tokens", type=int, default=4096, help="prefill 一次处理的 token 数")
    ap.add_argument("--topk", type=int, default=8, help="MoE 每 token 激活专家数")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPE_BYTES))
    args = ap.parse_args()
    dt = DTYPE_BYTES[args.dtype]

    print(f"\n参数：H={args.h}  d={args.d}  B={args.B}  prefill tokens={args.tokens}  "
          f"topk={args.topk}  dtype={args.dtype}\n")

    # ---------------------------------------------------------------- ① SiTU
    per_tok1, before1, after1 = situ_fusion(args.h, args.d, args.tokens, dt)
    print("=" * 80)
    print("① SiTU-GLU 融进上游 GEMM 的 epilogue（待做）")
    print("=" * 80)
    print(f"  省掉的是 gate_up 中间张量（[.., 2d]，d={args.d} → 2d={2*args.d}）的「写 + 读」")
    print(f"  每 token 省 4d 个元素 = {human(per_tok1)}")
    print()
    print(f"  ▸ Prefill（tokens={args.tokens}）：")
    print(f"      原链路搬运 {human(before1)}  →  融合后 {human(after1)}")
    print(f"      省 {human(before1 - after1)}，占原搬运 {(before1-after1)/before1*100:.1f}%")
    print(f"  ▸ Decode（单 token）对比：")
    w_moe = args.h * 2 * args.d * dt * args.topk      # 激活 topk 个专家的权重读取（粗略：每专家 2 个矩阵）
    print(f"      本项省 {human(per_tok1)}；同层 MoE 要读的激活权重 ≈ {human(w_moe)}（topk={args.topk}）")
    print(f"      占比 {(per_tok1/w_moe)*100:.2f}%  ← **decode 基本可忽略**")
    print()

    # ------------------------------------------------------------- ② AttnRes
    per_tok2, before2, after2, nvec = attnres_fusion(args.h, args.B, args.tokens, dt)
    print("=" * 80)
    print("② AttnRes 内部融合：RMSNorm + proj 折成一次点积（已做 + 对拍验证）")
    print("=" * 80)
    print(f"  折叠公式 score_weight = norm.weight ⊙ proj.weight（离线算好），数学等价")
    print(f"  省掉 v_normed 的「写 + 读」= 2H × (B+1) 每 token，B={args.B} → {nvec} 个残差向量")
    print(f"  每 token 省 2×{args.h}×{nvec} = {human(per_tok2)}")
    print()
    print(f"  ▸ Prefill（tokens={args.tokens}）：")
    print(f"      原链路搬运 {human(before2)}  →  融合后 {human(after2)}")
    print(f"      省 {human(before2 - after2)}，占原搬运 {(before2-after2)/before2*100:.1f}%")
    print(f"  ▸ Decode：同 ①，省的量相对整层权重读取可忽略")
    print()
    print(f"  ★ 注意：这一项**已经做完了**，所以它不是「计划」，是「已实现的收益」。")
    print(f"    比 ① 少一步（① 要动 GEMM 的 tiling，风险更高）——先做这个是正确的顺序。")
    print()

    # ---------------------------------------------------------- ③ LatentMoE
    latentmoe_note()

    # ---------------------------------------------------------------- 汇总
    print("=" * 80)
    print("汇总：三项工作对照（怎么跟对方讲）")
    print("=" * 80)
    ratio = per_tok2 / per_tok1
    print(f"  {'工作':<28}{'类型':<12}{'每 token 省':<14}{'状态':<10}")
    print(f"  {'-'*64}")
    print(f"  {'SiTU-GLU → GEMM epilogue':<28}{'性能/省访存':<12}{human(per_tok1):<14}{'待做':<10}")
    print(f"  {'AttnRes 内部融合':<28}{'性能/省访存':<12}{human(per_tok2):<14}{'已做 ✅':<10}")
    print(f"  {'LatentMoE 图接线':<28}{'使能/防返工':<12}{'—':<14}{'待做':<10}")
    print()
    print(f"  AttnRes 每 token 省的比 SiTU 多 {ratio:.2f}x（因为要乘 (B+1) 个残差向量）")
    print()
    print("  ★ 表述口径（三句话）：")
    print("    1. 这两项省的**不是时间，是 HBM 往返**；换时间要除带宽。")
    print("    2. 收益**集中在 prefill**（一次 N 个 token，放大 N 倍）；")
    print("       decode 每步要读整层权重，省的那点占比不到 1% —— 别混着说。")
    print("    3. AttnRes 那项**已经做完了**，SiTU 那项是下一步；")
    print("       而 LatentMoE 的价值是**判断出它是接线不是算子**，避免白写一个 kernel。")


if __name__ == "__main__":
    main()
