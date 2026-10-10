"""量化实验台：PTQ 的 scale 怎么选 / outlier 为什么毁精度 / SmoothQuant 在算什么。

零第三方依赖（纯标准库），任何有 Python3 的机器都能跑：

    python3 quant_lab.py
    python3 quant_lab.py --in-ch 256 --tokens 64 --out-ch 128 --outliers 3

回答四个高频追问（都是"怎么做"，不是"有哪些"）：
  1. scale 用 MinMax / Percentile / MSE，差别有多大？
  2. per-tensor 和 per-channel 差在哪、为什么激活不能 per-channel？
  3. 激活里的 outlier 为什么会把整层拖垮？
  4. SmoothQuant 的 s = max|X|^α / max|W|^(1-α) 到底把什么搬到了哪里？

核心结论（跑完会在最后打印）：
  · 权重量化用 per-channel，几乎没有代价 —— 权重是**静态**的，scale 可以离线算好
  · 激活量化只能 per-tensor —— 激活是**动态**的，per-channel scale 会让 GEMM 无法做矩阵化
  · 所以激活的 channel-wise outlier 就是致命的 → SmoothQuant 把它"搬"到权重侧
"""

import argparse
import math
import random


# ------------------------------------------------------------------ 基础量化
def qmax_of(bits):
    return (1 << (bits - 1)) - 1        # 对称 INT8 → 127


def quant_sym(x, scale, bits):
    qm = qmax_of(bits)
    q = int(round(x / scale))
    return max(-qm - 1, min(qm, q))


def dequant_sym(q, scale):
    return q * scale


def quant_dequant_sym(x, scale, bits):
    return dequant_sym(quant_sym(x, scale, bits), scale)


# ---------------------------------------------------------- scale 计算三种算法
def scale_minmax(values, bits):
    """最粗暴：取绝对值最大。一个 outlier 就能把 scale 撑大、把其余值压扁。"""
    m = max(abs(v) for v in values)
    return m / qmax_of(bits) if m > 0 else 1e-12


def scale_percentile(values, bits, pct=99.9):
    """按分位数裁剪：丢掉最大的 0.1%，换来其余 99.9% 的分辨率。"""
    s = sorted(abs(v) for v in values)
    idx = min(len(s) - 1, int(len(s) * pct / 100.0))
    m = s[idx]
    return m / qmax_of(bits) if m > 0 else 1e-12


def scale_mse(values, bits, steps=40):
    """在 [minmax/8, minmax] 里网格搜一个使量化误差平方和最小的 scale。"""
    hi = max(abs(v) for v in values)
    if hi <= 0:
        return 1e-12
    best_s, best_err = hi / qmax_of(bits), float("inf")
    for i in range(steps + 1):
        cand = hi * (1.0 / 8.0 + (7.0 / 8.0) * i / steps) / qmax_of(bits)
        err = 0.0
        for v in values:
            d = v - quant_dequant_sym(v, cand, bits)
            err += d * d
        if err < best_err:
            best_err, best_s = err, cand
    return best_s


def snr_db(orig, approx):
    sig = sum(o * o for o in orig)
    noise = sum((o - a) ** 2 for o, a in zip(orig, approx))
    return float("inf") if noise <= 0 else 10.0 * math.log10(sig / noise)


def max_rel(orig, approx, atol=1e-3):
    return max(abs(o - a) / (abs(o) + atol) for o, a in zip(orig, approx))


# ------------------------------------------------------------------ 数据生成
def gen_weight(out_ch, in_ch, seed=0):
    """权重：窄高斯（真实权重近似如此），按 output channel 一行。"""
    rng = random.Random(seed)
    return [[rng.gauss(0.0, 0.02) for _ in range(in_ch)] for _ in range(out_ch)]


def gen_activation(tokens, in_ch, outliers, seed=1, boost=30.0):
    """激活：高斯 + 少量**通道级** outlier —— 这就是 LLM 激活的真实形态。

    注意 outlier 是「固定几列特别大」，不是随机散布的尖峰 —— 这个区别决定了解法。
    """
    rng = random.Random(seed)
    hot = set(rng.sample(range(in_ch), min(outliers, in_ch)))
    X = []
    for _ in range(tokens):
        row = []
        for j in range(in_ch):
            v = rng.gauss(0.0, 1.0)
            if j in hot:
                v *= boost
            row.append(v)
        X.append(row)
    return X, hot


def matmul(A, B, in_ch):
    """A: m×in_ch, B: out_ch×in_ch（按行存）→ 返回 m×out_ch 且做转置乘。"""
    out = []
    for row in A:
        o = []
        for b in B:
            o.append(sum(row[k] * b[k] for k in range(in_ch)))
        out.append(o)
    return out


# --------------------------------------------------------------- 实验一：scale 算法
def exp_scale_algorithms(args):
    print("=" * 78)
    print("实验一：scale 三种算法的差别（权重量化，per-tensor，INT8）")
    print("=" * 78)
    W = gen_weight(args.out_ch, args.in_ch)
    flat = [v for row in W for v in row]
    for name, fn in (("MinMax", scale_minmax),
                     ("Percentile(99.5)", lambda v, b: scale_percentile(v, b, 99.5)),
                     ("MSE 搜索", scale_mse)):
        s = fn(flat, 8)
        approx = [quant_dequant_sym(v, s, 8) for v in flat]
        print(f"  {name:<18} scale={s:.6e}  SNR={snr_db(flat, approx):6.2f} dB  "
              f"max_rel={max_rel(flat, approx):.3e}")
    print("  → 权重分布窄且无 outlier 时，三种算法差别很小（MinMax 就够）。")
    print("    真正的分水岭出现在有 outlier 的激活上（实验二）。")
    print()


# ----------------------------------------------------- 实验二/三：outlier 与粒度
def exp_outlier_and_granularity(args):
    print("=" * 78)
    print(f"实验二：激活 outlier × 量化粒度（{args.tokens} tokens × {args.in_ch} ch，"
          f"{args.outliers} 个 outlier 通道 ×{args.boost:.0f}）")
    print("=" * 78)
    X, hot = gen_activation(args.tokens, args.in_ch, args.outliers, boost=args.boost)
    flat = [v for row in X for v in row]

    # ① per-tensor：全张量一个 scale
    s_t = scale_minmax(flat, 8)
    approx_t = [quant_dequant_sym(v, s_t, 8) for v in flat]

    # ② per-channel（按列，即同一个特征维一个 scale）—— 硬件不支持的理想方案
    chan_scale = []
    for j in range(args.in_ch):
        col = [X[t][j] for t in range(args.tokens)]
        chan_scale.append(scale_minmax(col, 8))
    approx_c = [quant_dequant_sym(X[t][j], chan_scale[j], 8)
                for t in range(args.tokens) for j in range(args.in_ch)]

    # ③ per-token（按行）—— 能治"个别 token 特别大"，但治不了通道 outlier
    tok_scale = [scale_minmax(row, 8) for row in X]
    approx_k = [quant_dequant_sym(X[t][j], tok_scale[t], 8)
                for t in range(args.tokens) for j in range(args.in_ch)]

    print(f"  outlier 通道 = {sorted(hot)[:12]}{' ...' if len(hot) > 12 else ''}")
    print(f"  {'粒度':<28}{'SNR(dB)':>10}{'max_rel':>12}")
    for name, ap in (("per-tensor（硬件默认）", approx_t),
                     ("per-token（按行）", approx_k),
                     ("per-channel（按列，理想）", approx_c)):
        print(f"  {name:<28}{snr_db(flat, ap):10.2f}{max_rel(flat, ap):12.3e}")
    print()
    print("  → 关键区别：per-token 几乎没用（outlier 是**固定几列**，每个 token 都中招）；")
    print("    per-channel 有效，但**激活是动态的**，per-channel scale 会让 GEMM 无法矩阵化 ——")
    print("    所以硬件上激活只能 per-tensor，这就是 SmoothQuant 要解决的那个矛盾。")
    print()


# --------------------------------------------------------------- 实验四：SmoothQuant
def exp_smoothquant(args, alpha=0.5):
    print("=" * 78)
    print(f"实验三：SmoothQuant 把 outlier 从激活搬到权重（α={alpha}）")
    print("=" * 78)
    W = gen_weight(args.out_ch, args.in_ch)
    X, hot = gen_activation(args.tokens, args.in_ch, args.outliers, boost=args.boost)
    P = args.in_ch

    Y_ref = matmul(X, W, P)                        # 未量化的真值

    def quant_forward(Xq_src, Wq_src, per_channel_w=True):
        """激活 per-tensor + 权重 per-channel，这是硬件的真实约束。"""
        flatx = [v for row in Xq_src for v in row]
        sx = scale_minmax(flatx, 8)
        Xq = [[quant_dequant_sym(v, sx, 8) for v in row] for row in Xq_src]
        sW = [scale_minmax(row, 8) for row in Wq_src]
        Wq = [[quant_dequant_sym(v, sW[i], 8) for v in row] for i, row in enumerate(Wq_src)]
        return matmul(Xq, Wq, P)

    # ① 不迁移：直接量化
    Y1 = quant_forward(X, W)

    # ② 迁移：s_j = max|X_j|^α / max|W_j|^(1-α)
    s = []
    for j in range(P):
        mx = max(abs(X[t][j]) for t in range(args.tokens))
        mw = max(abs(W[i][j]) for i in range(args.out_ch))
        s.append((mx ** alpha) / (mw ** (1 - alpha)) if mx > 0 and mw > 0 else 1.0)
    X_s = [[X[t][j] / s[j] for j in range(P)] for t in range(args.tokens)]
    W_s = [[W[i][j] * s[j] for j in range(P)] for i in range(args.out_ch)]
    Y2 = quant_forward(X_s, W_s)

    flat_ref = [v for row in Y_ref for v in row]
    print(f"  {'方案':<34}{'SNR(dB)':>10}{'max_rel':>12}")
    print(f"  {'① 直接量化（激活被 outlier 撑爆）':<34}"
          f"{snr_db(flat_ref, [v for r in Y1 for v in r]):10.2f}"
          f"{max_rel(flat_ref, [v for r in Y1 for v in r]):12.3e}")
    print(f"  {'② SmoothQuant 迁移后':<34}"
          f"{snr_db(flat_ref, [v for r in Y2 for v in r]):10.2f}"
          f"{max_rel(flat_ref, [v for r in Y2 for v in r]):12.3e}")

    mx_ratio = max(max(abs(X[t][j]) for t in range(args.tokens)) for j in range(P)) / \
        max(abs(v) for row in X for v in row)
    print(f"\n  数学等价性检查: X_s @ W_s 与 X @ W 应完全相等（只是重新分配了动态范围）")
    chk = matmul(X_s, W_s, P)
    d = max(abs(a - b) for ra, rb in zip(Y_ref, chk) for a, b in zip(ra, rb))
    print(f"    max|Δ| = {d:.3e}  {'等价 ✅' if d < 1e-9 else '❌ 不等价，检查公式'}")
    print()
    print("  → 为什么有效：激活的 outlier 是「列方向」的（固定几个特征维特别大），")
    print("    乘以 1/s_j 把这些列压回正常范围；代价是权重对应列被放大 s_j 倍，")
    print("    而**权重是静态的、可以 per-channel 量化**，放大它几乎不损失精度。")
    print("    α 控制搬运力度：α=0 全压在权重，α=1 全留在激活，0.5 是常用折中。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-ch", type=int, default=128)
    ap.add_argument("--out-ch", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--outliers", type=int, default=3, help="outlier 通道数")
    ap.add_argument("--boost", type=float, default=30.0, help="outlier 放大倍数")
    args = ap.parse_args()

    exp_scale_algorithms(args)
    exp_outlier_and_granularity(args)
    exp_smoothquant(args)

    print("=" * 78)
    print("问答可以直接说的四句")
    print("=" * 78)
    print("  1. 权重 per-channel、激活 per-tensor —— 不是不想，是激活动态、per-channel 破坏矩阵化。")
    print("  2. outlier 是通道级的（固定几列），所以 per-token 救不了，只有换粒度或搬走。")
    print("  3. SmoothQuant 搬、AWQ 保、GPTQ 重建 —— 三条路都在绕开同一个矛盾。")
    print("  4. scale 算法只有在有 outlier 时才拉开差距；激活上 MinMax 往往不够，")
    print("     要么 Percentile/MSE，要么先做 SmoothQuant 再 MinMax。")


if __name__ == "__main__":
    main()
