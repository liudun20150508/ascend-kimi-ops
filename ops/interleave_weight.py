"""权重交错重排工具 —— SituGluFused 融合算子的必要前置。

    python3 interleave_weight.py                    # 自检（往返一致 + 布局示意）
    python3 interleave_weight.py --mode interleave --in w.pt --out wp.pt --subn 64
    python3 interleave_weight.py --mode deinterleave --in wp.pt --out w.pt --subn 64

为什么需要它：
    融合算子要求权重是**交错重排**过的 ——
        W' = [gate_0, up_0, gate_1, up_1, ...]     每片 subN 行
    这样取 BaseN = 2 × subN × kPairs 时，每个 L0C tile 内同时含配对的 gate/up，
    epilogue 才能就地算 situ(gate) × tanh(up) 而不用暂存整行。

⚠️ **忘了重排 → 结果完全错，而且不会报错**（gate 和 up 配不上对）。
    这是本方案最危险的一步，所以这个工具自带往返自校验。

不变量（kernel 与 host 都依赖）：
    W' 的行数 == 2 × d，且 d % subN == 0，baseN == 2 × subN × kPairs
"""

import argparse
import sys

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False


# ---------------------------------------------------------------- 纯 Python 实现
def interleave_plain(gate, up, subn):
    """两个 [d, H] 的 list-of-list → 交错后的 [2d, H]。"""
    d = len(gate)
    out = []
    for j in range(0, d, subn):
        out.extend(gate[j:j + subn])
        out.extend(up[j:j + subn])
    return out


def deinterleave_plain(wp, subn):
    """交错权重 [2d, H] → (gate, up) 两个 [d, H]。"""
    gate, up = [], []
    for j in range(0, len(wp), 2 * subn):
        gate.extend(wp[j:j + subn])
        up.extend(wp[j + subn:j + 2 * subn])
    return gate, up


# --------------------------------------------------------------- numpy 实现
def interleave_np(gate, up, subn):
    d = gate.shape[0]
    assert d % subn == 0, f"d={d} 不是 subN={subn} 的整数倍"
    blocks = []
    for j in range(0, d, subn):
        blocks.append(gate[j:j + subn])
        blocks.append(up[j:j + subn])
    return np.concatenate(blocks, axis=0)


def deinterleave_np(wp, subn):
    d2 = wp.shape[0]
    gate_blocks, up_blocks = [], []
    for j in range(0, d2, 2 * subn):
        gate_blocks.append(wp[j:j + subn])
        up_blocks.append(wp[j + subn:j + 2 * subn])
    return (np.concatenate(gate_blocks, axis=0),
            np.concatenate(up_blocks, axis=0))


# ---------------------------------------------------------------- 自检
def selfcheck(subn=64, d=256, h=8):
    print("=" * 74)
    print(f"自检：interleave / deinterleave 往返一致（d={d}, H={h}, subN={subn}）")
    print("=" * 74)

    if HAVE_NUMPY:
        print("（用 numpy）")
        rng = np.random.default_rng(0)
        gate = rng.standard_normal((d, h)).astype(np.float32)
        up = rng.standard_normal((d, h)).astype(np.float32)
        wp = interleave_np(gate, up, subn)
        g2, u2 = deinterleave_np(wp, subn)
        ok = np.array_equal(gate, g2) and np.array_equal(up, u2)
    else:
        print("（没有 numpy，用纯 Python 小规模自检）")
        import random
        rng = random.Random(0)
        gate = [[rng.random() for _ in range(h)] for _ in range(d)]
        up = [[rng.random() for _ in range(h)] for _ in range(d)]
        wp = interleave_plain(gate, up, subn)
        g2, u2 = deinterleave_plain(wp, subn)
        ok = (gate == g2) and (up == u2)

    print(f"  往返一致: {'✅ PASS' if ok else '❌ FAIL'}")
    print()

    # 布局示意（只看行号）
    print(f"  重排后的行顺序（d={d}, subN={subn}，示意前 3 片）：")
    for j in range(0, min(d, 3 * subn), subn):
        k = j // subn
        tagname = "gate" if k % 2 == 0 else "up"
        print(f"    行 {j:>5} .. {j + subn - 1:>5}   ← {tagname}_{k // 2}")
    print()
    print("  等价说法：W'[2k*subN + i] = gate[k*subN + i]"
          "   W'[(2k+1)*subN + i] = up[k*subN + i]")
    print()
    print("  ★ 记牢：**忘了重排不会报错，只会算错**。")
    print(f"     d={d} 能被 subN={subn} 整除 是前提（KIMI3: d=11008 = 64×172 ✅）")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["selfcheck", "interleave", "deinterleave"],
                    default="selfcheck")
    ap.add_argument("--in", dest="inp", help="输入张量（.pt 需 torch，.npy 需 numpy）")
    ap.add_argument("--out", dest="out", help="输出路径（.npy）")
    ap.add_argument("--gate", help="interleave 模式的 gate 权重路径")
    ap.add_argument("--up", help="interleave 模式的 up 权重路径")
    ap.add_argument("--subn", type=int, default=64)
    args = ap.parse_args()

    if args.mode == "selfcheck":
        sys.exit(0 if selfcheck(args.subn) else 1)

    if not HAVE_NUMPY:
        print("文件模式需要 numpy（本机没有；自检模式纯 Python 也能跑）")
        sys.exit(1)

    if args.mode == "interleave":
        if not (args.gate and args.up and args.out):
            print("interleave 模式需要 --gate / --up / --out")
            sys.exit(1)
        gate, up = np.load(args.gate), np.load(args.up)
        np.save(args.out, interleave_np(gate, up, args.subn))
        print(f"已写出 {args.out}（{gate.shape[0]}×2 行）")
    else:
        if not (args.inp and args.out):
            print("deinterleave 模式需要 --in / --out")
            sys.exit(1)
        g, u = deinterleave_np(np.load(args.inp), args.subn)
        np.save(args.out + ".gate.npy", g)
        np.save(args.out + ".up.npy", u)
        print(f"已写出 {args.out}.gate.npy / {args.out}.up.npy")


if __name__ == "__main__":
    main()
