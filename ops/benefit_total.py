"""全部工作的收益总账（零依赖，本机可跑）。

    python3 benefit_total.py
    python3 benefit_total.py --d 11008 --h 4096 --B 8 --tokens 4096 --torch-ops 6

把整个项目里「华为没做 / 我做了」的每一项都折算成**每 token 的 HBM 搬运量**，
再分 prefill / decode 两个阶段给结论。

★ 两个贯穿全文的口径，先说清楚（问答也照这个说）：

  口径 A：比较基准必须是「**等价功能的朴素实现**」，不能是「华为的算子」——
          SiTU-GLU / AttnRes 华为没有对应算子，跟它比是假命题。
  口径 B：省的**不是时间，是 HBM 往返**；换成时间要除带宽。
          而且收益**集中在 prefill**（一次 N 个 token 放大 N 倍），
          decode 每步要读整层权重（几百 MB），省的那点占比 <1%。

每个数都是「逐项字节数」推出来的，不是估的。凡是有估算成分的参数（如 torch 组合
用了几个算子）都做成命令行参数，你可以按真实统计改。
"""

import argparse

DTYPE_BYTES = {"bf16": 2, "fp16": 2}


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or u == "GB":
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} GB"


def section(title):
    print()
    print("=" * 82)
    print(title)
    print("=" * 82)


def per_token_line(label, elems, dt):
    print(f"    {label:<34} {elems:>12.1f} 元素 = {human(elems*dt):>10}")
    return elems * dt


# ===========================================================================
# ① SiTU-GLU：三种实现的搬运对比
# ===========================================================================
def scope_situ(d, tokens, dt, torch_ops, rows_note=""):
    section("① SiTU-GLU —— 三种实现的每 token 搬运对比")
    print(f"  输入：GEMM 已产出 gate_up[M, 2d]（d={d} → 2d={2*d}）")
    print()

    # 口径：都从「GEMM 该产出 gate_up」这一刻开始算，所以非融合方案都要算上 GEMM 的**写出**。
    # （融合的本质就是让这一步不发生 —— 所以 c 里没有它。漏掉这一项会让融合收益少算一半。）
    gemm_write = 2 * d
    # a) torch 算子组合：GEMM 写 gate_up + 读 gate_up + 中间往返 + 写 out
    a = gemm_write + 2 * d + 2 * torch_ops * d + d
    # b) 我的独立算子：GEMM 写 gate_up + 读 gate_up + 写 out
    b = gemm_write + 2 * d + d
    # c) 融合进 GEMM epilogue：gate_up 不落地，只有写 out
    c = d

    print(f"    {'实现':<26}{'元素/token':>14}{'字节/token':>13}   说明")
    print(f"    {'-'*70}")
    print(f"    {'a) torch 算子组合':<26}{a:>14.1f}{human(a*dt):>13}   {torch_ops} 个 elementwise")
    print(f"    {'b) 我的独立算子':<26}{b:>14.1f}{human(b*dt):>13}   1 个 kernel")
    print(f"    {'c) 融合(GEMM epilogue)':<26}{c:>14.1f}{human(c*dt):>13}   中间张量不落 HBM")
    print()
    print(f"    ⇒ b 相对 a 省 {a-b:.0f} 元素/token（{(1-b/a)*100:.0f}%，把 {torch_ops} 个算子合成 1 个）")
    print(f"    ⇒ c 相对 b 省 {b-c:.0f} 元素/token（{(1-c/b)*100:.0f}%，= GEMM 写 2d + 算子读 2d）")
    print(f"    ⇒ c 相对 a 省 {a-c:.0f} 元素/token（{(1-c/a)*100:.0f}%，累计）")
    print()
    print(f"    ⚠️ 这是「每 token **每执行一次 SiTU**」的账。MoE 下每 token 只激活 k 个专家，")
    print(f"       要乘 k；dense MLP 则是每层 1 次。")
    print()
    print(f"    Prefill（{tokens} token）：a→c 省 {human((a-c)*dt*tokens)}")
    print(f"    状态：b 已做并验证（max_rel 7.5e-3）；c 已设计未实现")
    return a, b, c


# ===========================================================================
# ② AttnRes：朴素实现 vs 单 kernel（内部已融合）
# ===========================================================================
def scope_attnres(H, B, tokens, dt, naive_ops):
    section("② AttnRes —— 朴素多算子实现 vs 单 kernel（内部已融合）")
    nvec = B + 1
    print(f"  输入：v = [T, B+1, H]（B={B} → {nvec} 个残差向量，H={H}）")
    print(f"  AttnRes 内部嵌了 RMSNorm + Linear(proj)，外包一层 softmax + 加权和")
    print()

    # a) 朴素：RMSNorm + proj + softmax + 加权和，每个算子对 (B+1)×H 做读写
    a = nvec * H + naive_ops * 2 * nvec * H + H      # 读 v + 各算子往返 + 写 out
    # b) 单 kernel：读 v 一遍（顺带算 rms 与点积），写 out
    b = nvec * H + H

    label_a = f"a) 朴素（{naive_ops} 个算子）"
    print(f"    {'实现':<30}{'元素/token':>14}{'字节/token':>13}")
    print(f"    {'-'*60}")
    print(f"    {label_a:<30}{a:>14.1f}{human(a*dt):>13}")
    print(f"    {'b) 我的单 kernel':<30}{b:>14.1f}{human(b*dt):>13}")
    print()
    print(f"    ⇒ 省 {a-b:.0f} 元素/token（{(1-b/a)*100:.0f}%）")
    print(f"       其中「norm + proj 折成一次点积」这一项（已做）= 2×H×{nvec} = {2*H*nvec} 元素")
    print(f"    Prefill（{tokens} token）：省 {human((a-b)*dt*tokens)}")
    print(f"    状态：**已做并验证**（max_rel 7.7e-3）")
    return a, b


# ===========================================================================
# ③ LatentMoE：使能类
# ===========================================================================
def scope_latentmoe():
    section("③ LatentMoE 图接线 —— 使能类，不按带宽量化")
    print("""
  形态：down_proj(Linear H→moe_hidden) → MoE → [RMSNorm] → up_proj(Linear moe_hidden→H)
        两个 Linear 是现成 GEMM，MoE 是华为已有算子 → **不需要写任何新 kernel**

  不接线：模型形状对不上，**跑不起来**（0 → 1）
  接线后：用现成算子拼，改动量 = 模型文件里加两个 Linear + 一个 norm

  收益表述（不能编成字节数）：
    「最大的收益是**避免返工** —— 把形状不符误判成新算子，就是一个本来不需要的
      AscendC 开发周期（参考本项目：一个算子从 kernel 到对拍验证要几天）。
      判断出『这是接线问题』本身就是价值。」
""")


# ===========================================================================
# ④ kernel 流水优化（double buffer / tiling 参数化）
# ===========================================================================
def scope_pipeline():
    section("④ kernel 流水与 tiling —— 收益是「性能」不是「访存」，且未实测")
    print("""
  改动项（都在 SituAndMul 上做的）：
    · TQue depth 1→2（double buffer）：原来 MTE2 / V / MTE3 **完全串行**，
      改成可重叠。理想情况三段取 max 而非 sum → 理论上限提升，但实际取决于哪段最长。
    · tileElem 由 host 按 UB 容量反推（不再硬编码 4096）
    · fp32 缓冲 4→3 块（acc 复用）
    · 任务粒度从「整行」改「(row,tile) 分块」均分

  ⚠️ 这一项的收益**不能凭理论给数字**：
      访存受限算子的下限是「搬运量 ÷ 带宽」，double buffer **不减少搬运量**，
      它只是让搬运与计算并行 —— 所以省的是「没重叠掉的那部分时间」。
      实测方法见 kimi3_ops/docs/04：msprof 看 MTE2/V/MTE3 是否相加。
      「相加 → 没重叠；取 max → 重叠了」。

  表述口径：
    「double buffer 不省访存，它省的是**没被重叠的那段时间**。判据不是加速比，
      是 msprof 里 MTE2+vec+MTE3 的时间从『相加』变成『取 max』。」
""")


# ===========================================================================
# 总账
# ===========================================================================
def summary(d, H, B, tokens, dt, torch_ops, naive_ops):
    section("总账：所有工作项对照（**阶梯口径，跨行不可相加**）")
    nvec = B + 1

    # 阶梯口径：每一行都是「相对上一级实现」省的，所以不能把各行的数字直接加起来。
    # 真正能相加的是「最终版 vs 原始实现」，见下面的「阶梯合计」。
    situ_step1 = 2 * torch_ops * d            # torch 组合 → 我的独立算子
    situ_step2 = 4 * d                        # 独立算子 → GEMM epilogue 融合
    situ_total = situ_step1 + situ_step2      # torch 组合 → 融合版
    attn_total = naive_ops * 2 * nvec * H     # 朴素多算子 → 我的单 kernel（含内部融合）

    rows = [
        # (工作项, 性质, 每 token 省(元素), 相对基准, 状态)
        ("SiTU ①：torch组合 → 独立算子", "补缺", situ_step1, "torch 组合", "已做+验证"),
        ("SiTU ②：独立算子 → GEMM 融合", "优化", situ_step2, "独立算子", "已设计"),
        ("AttnRes：朴素 → 单kernel(含融合)", "补缺+融合", attn_total, "朴素多算子", "已做+验证"),
        ("LatentMoE 图接线", "使能", None, "—", "已落地未验证"),
        ("kernel 流水 double buffer", "性能(不省访存)", None, "—", "已改未验证"),
    ]

    print(f"  {'工作项':<34}{'性质':<14}{'每 token 省':<11}{'相对基准':<12}{'状态'}")
    print(f"  {'-'*92}")
    for name, kind, elems, base, status in rows:
        val = "—" if elems is None else human(elems * dt)
        print(f"  {name:<34}{kind:<14}{val:<11}{base:<12}{status}")
    print(f"  {'-'*92}")
    print()
    print("  阶梯合计（最终版 vs 原始实现，不重复计）：")
    print(f"    SiTU-GLU : torch组合 → 融合版  = {situ_step1:.0f} + {situ_step2:.0f} = "
          f"{situ_total:.0f} 元素/token = {human(situ_total*dt)}")
    print(f"    AttnRes  : 朴素     → 单kernel = {attn_total:.0f} 元素/token = "
          f"{human(attn_total*dt)}")
    grand = situ_total + attn_total
    print("    " + "-" * 62)
    print(f"    合计 = {grand:.0f} 元素/token = {human(grand*dt)}")
    print()
    print(f"  Prefill 一次 {tokens} token：省 {human(grand*dt*tokens)}")
    layer_w = H * 2 * d * dt * 8       # 整层 MoE 激活权重的粗略量（按 8 个专家估）
    print(f"  Decode 单步：{human(grand*dt)}，相对整层权重读取（≈{human(layer_w)}）"
          f" = {grand*dt/layer_w*100:.2f}%  ← **可忽略**")
    print()
    print("  ★ 怎么读这张表（三句话）：")
    print("    1. 前两行是**阶梯**（① 然后 ②），不能相加；能相加的是「最终版 vs 原始」。")
    print("    2. 所有数字都是**每 token 的 HBM 往返**，换时间要除带宽。")
    print("    3. 收益集中在 **prefill**（乘 token 数）；decode 那点被整层权重读取淹没。")
    print("    4. LatentMoE 是**使能**、double buffer 是**性能**——性质不同，别混进合计。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=4096, help="hidden_size")
    ap.add_argument("--d", type=int, default=11008, help="intermediate_size")
    ap.add_argument("--B", type=int, default=8, help="attn_res_block_size")
    ap.add_argument("--tokens", type=int, default=4096, help="prefill 一次处理的 token 数")
    ap.add_argument("--torch-ops", type=int, default=6,
                    help="torch 版 SiTU-GLU 用了几个 elementwise 算子（可调，按真实统计改）")
    ap.add_argument("--naive-ops", type=int, default=4,
                    help="朴素 AttnRes 用了几个算子（RMSNorm/proj/softmax/加权和）")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPE_BYTES))
    args = ap.parse_args()
    dt = DTYPE_BYTES[args.dtype]

    print(f"\n参数：H={args.h} d={args.d} B={args.B} prefill tokens={args.tokens} "
          f"torch-ops={args.torch_ops} naive-ops={args.naive_ops} dtype={args.dtype}")

    scope_situ(args.d, args.tokens, dt, args.torch_ops)
    scope_attnres(args.h, args.B, args.tokens, dt, args.naive_ops)
    scope_latentmoe()
    scope_pipeline()
    summary(args.d, args.h, args.B, args.tokens, dt, args.torch_ops, args.naive_ops)


if __name__ == "__main__":
    main()
