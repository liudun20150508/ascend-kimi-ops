"""融合算子（GEMM + SiTU epilogue）的收益与 tiling 参数计算器（零依赖）。

    python3 fusion_tiling_calc.py
    python3 fusion_tiling_calc.py --d 11008 --h 4096 --tokens 3584
    python3 fusion_tiling_calc.py --show-all
    python3 fusion_tiling_calc.py --nbuf 2        # epilogue 只用 2 块 UB 缓冲

算四件事（上机前把参数备好，别到机器上现推）：

  ① 融合收益      —— 省掉 2d 中间张量的写+读，多少字节、多少比例
  ② 两套粒度      —— **GEMM 的 (BaseM,BaseN,BaseK) 受 L0C/L0A/L0B 约束；
                     epilogue 每次处理 (BaseM, subN) 受 UB 约束。**
                     subN < BaseN 时 epilogue 必须分片 —— 这是融合最容易卡住的地方
  ③ epilogue 压力 —— 独立 elementwise 算子的 tile 只受 UB 约束（可以取到几千元素），
                     融合后 tile 被 GEMM 的 BaseM 绑住，UB 反而成了最紧那一级
  ④ 边界对齐      —— d 能否被 BaseN 整除，决定 epilogue 能否"按 tile 二分"

⚠️ 容量常数取 910_93 量级的工程常用值。上机请用
   `PlatformAscendC::GetCoreMemSize(...)` 打印真实值替换 CAP。
⚠️ 本脚本产出的参数是**纸面推导，未编译验证**。
"""

import argparse

# 各级片上容量（字节）—— 910_93 量级，上机请用 GetCoreMemSize 核对
CAP = {
    "L0A": 64 * 1024,
    "L0B": 64 * 1024,
    "L0C": 128 * 1024,
    "L1": 512 * 1024,
    "UB": 192 * 1024,
}
DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}
ACC_BYTES = 4            # L0C 累加永远是 fp32
UB_SAFE_FRACTION = 0.5   # UB 只用一半，余量留给框架同步
ALIGN = 16               # bf16 下 32B = 16 元素


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024 or u == "GB":
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} GB"


# ---------------------------------------------------------------- ① 融合收益
def benefit(d, tokens, dt_bytes):
    print("=" * 78)
    print("① 融合收益：SiTU-GLU 融进上游 GEMM 的 epilogue")
    print("=" * 78)
    # 不融合：GEMM 写 gate_up(2d) → SiTU 读(2d) + 写 out(d)  = 5d 元素
    # 融合后：epilogue 直接算，只写 out(d)                    = 1d 元素
    before, after = 5 * d, 1 * d
    saved_elem = before - after                       # = 4d
    per_token = saved_elem * dt_bytes
    total = per_token * tokens
    print(f"  d={d}  2d={2*d}  tokens/层={tokens}  dtype={dt_bytes}B")
    print(f"  不融合（元素/token）: 写 2d + 读 2d + 写 d = {before}")
    print(f"  融合后（元素/token）: 只写 d              = {after}")
    print(f"  ──────────────────────────────────────────────")
    print(f"  每 token 省 {saved_elem} 元素 = {human(per_token)}")
    print(f"  一层 MLP（{tokens} token）省 {human(total)}")
    print(f"  搬运量降低 {saved_elem / before * 100:.0f}% —— 这是**确定收益**，不是估计")
    print()
    return per_token


# ------------------------------------------- ② GEMM 粒度（L0C/L0A/L0B 约束）
def gemm_candidates(dt_bytes):
    """只受片上矩阵级约束，与 epilogue 无关。返回 [(BaseM, BaseN, BaseK, L0C%)]。"""
    out = []
    for bm in (64, 128, 256):
        for bn in (64, 128, 256):
            if bm * bn * ACC_BYTES > CAP["L0C"]:
                continue
            for bk in (16, 32, 64, 128, 256):
                if bm * bk * dt_bytes > CAP["L0A"]:
                    continue
                if bk * bn * dt_bytes > CAP["L0B"]:
                    continue
                l0c = bm * bn * ACC_BYTES / CAP["L0C"]
                out.append((bm, bn, bk, l0c))
    return out


def epilogue_subn(bm, nbuf):
    """epilogue 一次能从 L0C 取多少列进 UB（受 UB 额度约束，对齐到 ALIGN）。"""
    budget = int(CAP["UB"] * UB_SAFE_FRACTION)      # 取整，避免下游打印出 64.0 这种浮点
    raw = budget // (bm * ACC_BYTES * nbuf)
    return max(ALIGN, raw // ALIGN * ALIGN)


def search_tiling(d2, dt_bytes, nbuf, show_all):
    print("=" * 78)
    print("② 两套粒度：GEMM 的 L0C 分块 + epilogue 的 UB 分片")
    print("=" * 78)
    print(f"  L0A {CAP['L0A']//1024}KB | L0B {CAP['L0B']//1024}KB | L0C {CAP['L0C']//1024}KB | "
          f"L1 {CAP['L1']//1024}KB | UB {CAP['UB']//1024}KB（epilogue 额度 "
          f"{human(CAP['UB']*UB_SAFE_FRACTION)}，{nbuf} 块缓冲）")
    print()
    print("  ★ 关键：这两套粒度不是一回事 ——")
    print("    GEMM 的 (BaseM, BaseN) 由 L0C 定，一次累加一整块；")
    print("    epilogue 从 L0C 里**分片**取到 UB 做向量运算，一次能取多宽由 UB 定。")
    print()

    cands = []
    for bm, bn, bk, l0c in gemm_candidates(dt_bytes):
        subn = epilogue_subn(bm, nbuf)
        split = -(-bn // subn)          # 向上取整 = 分片次数
        cands.append((bm, bn, bk, l0c, subn, split))
    if not cands:
        print("  ❌ 无可行 GEMM 组合（连 L0A/L0B/L0C 都过不了），检查 dtype 或容量常数")
        return None

    # 推荐：L0C 填得满（复用率高）→ 分片次数少（循环开销小）→ BaseK 大（K 迭代少）
    def rank(c):
        bm, bn, bk, l0c, subn, split = c
        return (round(l0c, 2), -split, bk)

    best = max(cands, key=rank)
    rows = sorted(cands, key=rank, reverse=True)
    show = rows if show_all else rows[:6]

    print(f"  {'BaseM':>6}{'BaseN':>7}{'BaseK':>7}{'L0C':>8}{'epi subN':>10}{'epi 分片':>10}{'N向块数':>9}")
    for bm, bn, bk, l0c, subn, split in show:
        nblk = -(-d2 // bn)
        mark = "  ← 推荐" if (bm, bn, bk, l0c, subn, split) == best else ""
        print(f"  {bm:>6}{bn:>7}{bk:>7}{l0c*100:>7.0f}%{subn:>10}{split:>10}{nblk:>9}{mark}")
    if not show_all and len(rows) > len(show):
        print(f"  …（共 {len(rows)} 个可行组合，加 --show-all 看全部）")
    print()
    if best[5] > 1:
        print(f"  ⚠️ 推荐组合里 epilogue 需要**分 {best[5]} 片**（subN={best[4]} < BaseN={best[1]}）——")
        print("     这不是缺陷，是必然：L0C 一个块 128×256，UB 一次吃不下这么大。")
        print("     实现上就是 `Iterate()` 拿一小块 → 向量算 → 写回，循环 BaseN/subN 次。")
    else:
        print("  ✅ 推荐组合里 epilogue 一次能吃完整个 BaseN，不用分片")
    print()
    return best


# ------------------------------------------------------ ③ epilogue UB 压力
def epilogue_pressure(d, best, nbuf):
    print("=" * 78)
    print("③ epilogue 的 UB 压力 —— 融合和独立 elementwise 算子的真正差别")
    print("=" * 78)
    if not best:
        print("  无可行组合，跳过")
        return
    bm, bn, bk, l0c, subn, split = best
    standalone_tile = int(CAP["UB"] * UB_SAFE_FRACTION / 24)   # 独立版：UB×0.5/24B
    print(f"  独立 SiTU-GLU 的 tile：{standalone_tile} 个元素 "
          f"（UB×0.5/24B —— 只受 UB 约束，想取多大取多大）")
    print(f"  融合后的 tile：一次只能处理 {bm}×{subn} = {bm*subn} 个元素 "
          f"（被 GEMM 的 BaseM={bm} 绑住）")
    print(f"  → 融合后单次处理量是独立版的 {bm*subn/standalone_tile:.2f} 倍")
    print()
    for n in (2, 3, 4):
        s = epilogue_subn(bm, n)
        need = bm * s * ACC_BYTES * n
        print(f"    nbuf={n}: subN={s:>4}  占用 {human(need):>9}  "
              f"{'✅' if need <= CAP['UB']*UB_SAFE_FRACTION else '❌'}")
    print()
    print("  ★ 结论：融合**不是免费**的。它省的是 HBM 往返（收益），")
    print("    代价是 tile 粒度被 GEMM 绑住、epilogue 必须分片、UB 变得很紧。")
    print("    所以追问里讲融合，不能只说省了多少带宽，还要说清这个约束 ——")
    print("    这才是「融合不一定更快」那条反直觉结论的同一套逻辑。")
    print()


# ------------------------------------------------------------ ④ 边界对齐
def alignment_check(d, best):
    print("=" * 78)
    print("④ 边界对齐：d 能否被 BaseN 整除（决定 epilogue 能否按 tile 二分）")
    print("=" * 78)
    bn = best[1] if best else 128
    print(f"  d={d}  BaseN={bn}  ⇒  d/BaseN = {d/bn:.4f}")
    if d % bn == 0:
        print(f"  ✅ 整除（{d} = {bn} × {d//bn}）—— 每个 tile 要么全在 gate 区、要么全在 up 区，")
        print("     epilogue 只需按 tile 索引选分支，无跨界处理。")
    else:
        print(f"  ⚠️ 不整除，第 {d//bn} 块会横跨 gate/up 边界。三种处理：")
        print("     ① 该块内做掩码分两次算  ② 换 BaseN 使其整除  ③ 输入侧 pad 到整除")
    print()
    d2 = 2 * d
    print(f"  参考（2d = {d2}）：")
    for cand in (64, 128, 256):
        nblk = -(-d2 // cand)
        tail = "末块不满" if d2 % cand else "正好整除"
        print(f"    BaseN={cand:>4} → {nblk:>3} 块（{tail}），d%BaseN={d % cand}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=4096, help="输入特征维（KIMI3 hidden_size）")
    ap.add_argument("--d", type=int, default=11008, help="intermediate_size（输出通道的一半）")
    ap.add_argument("--tokens", type=int, default=3584, help="一层 MLP 处理的 token 数")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPE_BYTES))
    ap.add_argument("--nbuf", type=int, default=3, help="epilogue 的 UB 缓冲块数")
    ap.add_argument("--show-all", action="store_true")
    args = ap.parse_args()

    dt = DTYPE_BYTES[args.dtype]
    print(f"\n参数：H={args.h}  d={args.d}  2d={2*args.d}  tokens={args.tokens}  dtype={args.dtype}\n")

    benefit(args.d, args.tokens, dt)
    best = search_tiling(2 * args.d, dt, args.nbuf, args.show_all)
    epilogue_pressure(args.d, best, args.nbuf)
    alignment_check(args.d, best)

    if best:
        bm, bn, bk, l0c, subn, split = best
        print("=" * 78)
        print("上机可直接用的参数（复制过去，别现场推）")
        print("=" * 78)
        print(f"  BaseM={bm}  BaseN={bn}  BaseK={bk}    (L0C 占用 {l0c*100:.0f}%)")
        print(f"  epilogue 每次处理 BaseM×subN = {bm}×{subn}，分片 {split} 次")
        print(f"  K 方向迭代 = ceil(H/BaseK) = {-(-args.h//bk)}")
        print(f"  N 方向块数 = ceil(2d/BaseN) = {-(-(2*args.d)//bn)}")
        l1_need = (bm * bk + bk * bn) * dt * 2       # A/B 各一块 × 双缓冲
        print(f"  L1 需求（stepM={bm} stepN={bn}，双缓冲）= {human(l1_need)} "
              f"{'✅' if l1_need <= CAP['L1'] else '❌ 超 L1，减小 step 或降缓冲深度'}")
        print()
        print("  ⚠️ 纸面推导，未编译验证。上机第一步用 GetCoreMemSize 核对真实容量。")


if __name__ == "__main__":
    main()
