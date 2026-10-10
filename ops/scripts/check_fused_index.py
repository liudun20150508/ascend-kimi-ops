"""纯 CPU 自检：**重排约定 ↔ 新 epilogue 取数公式 ↔ golden** 三者是否自洽。

为什么需要它
------------
融合算子最容易的错法是「gate/up 配对错了」，而且**不会报错、只会算错**
（见 ../kernels/SituGluFused/README.md §5 的「最危险的一点」）。
重建版（docs/14 §16）把 epilogue 从「在 UB 里按 L0C tile 切片」改成了
「在 GM 里按 (row, pairIdx) 切片」，**索引公式变了** —— 而整条链路在开发容器里跑不了
（宿主沙箱拒绝 bash，见仓库根 reasonix.toml 的注释）。

所以这里用纯 Python 把**索引公式**单独验一遍：
它不测 AscendC、不测 Matmul，只用小矩阵模拟，
验证「kernel 的取数公式 + 交错重排规则 + 参考实现」三者互相自洽。

跑法（30 秒，零依赖，不需要 torch/numpy）
----------------------------------------
    python3 scripts/check_fused_index.py
    python3 scripts/check_fused_index.py --M 3 --H 5 --d 12 --subN 4

它做两件事：
    ① 正向：按 kernel 的公式取数 → 必须与 golden **完全一致**（浮点误差量级 1e-16）
    ② 反向（自证）：故意**不做重排** → 必须 FAIL，否则说明这个检查恒过、没有意义

⚠️ 它证明不了一件事：AscendC 的 `DataCopyPad` / `Matmul.IterateAll` 在真机上的实际行为。
   那些只能上机（见 README §6 的 11 条）。
"""

import argparse
import math
import random


# --------------------------------------------------------------------------
# 数学（与 reference/situ.py、SituAndMul kernel 一字不差）
# --------------------------------------------------------------------------
def situ(gate, beta):
    """situ(g) = β·tanh(g/β)·σ(g)"""
    return beta * math.tanh(gate / beta) * (1.0 / (1.0 + math.exp(-gate)))


def tanh_scale(up, linear_beta):
    """up' = linear_beta · tanh(up / linear_beta)"""
    return linear_beta * math.tanh(up / linear_beta)


# --------------------------------------------------------------------------
# 重排（与 interleave_weight.py 的规则一致）
#   W'[2k·subN + i] = gate[k·subN + i]      W'[(2k+1)·subN + i] = up[k·subN + i]
# --------------------------------------------------------------------------
def interleave(gate, up, d, subN):
    assert d % subN == 0, f"d={d} 不是 subN={subN} 的整数倍（host 侧会直接 GRAPH_FAILED）"
    wp = [None] * (2 * d)
    for k in range(d // subN):
        for i in range(subN):
            wp[2 * k * subN + i] = gate[k * subN + i]
            wp[(2 * k + 1) * subN + i] = up[k * subN + i]
    assert all(v is not None for v in wp), "重排后有空洞 —— 规则写错了"
    return wp


# --------------------------------------------------------------------------
# GEMM 模拟：C[M, N] = x[M, H] · W[N, H]^T   （W 的行就是 C 的列）
# --------------------------------------------------------------------------
def gemm(x, w, M, H, N):
    return [[sum(x[r][k] * w[c][k] for k in range(H)) for c in range(N)] for r in range(M)]


# --------------------------------------------------------------------------
# 被测路径：完全照 op_kernel/situ_glu_fused.cpp 的 epilogue 公式搬过来
#   colGate = pairIdx*2*subN ; colUp = colGate + subN
#   写回位置 = y[row][pairIdx*subN + i]
#   （行分核只影响性能，不影响数值 —— 所以这里逐行算就够了）
# --------------------------------------------------------------------------
def kernel_path(C, M, d, subN, beta, linear_beta):
    n2d = 2 * d
    nPairs = d // subN
    y = [[0.0] * d for _ in range(M)]
    for row in range(M):
        for pairIdx in range(nPairs):
            colGate = pairIdx * 2 * subN
            colUp = colGate + subN
            for i in range(subN):
                g = C[row][colGate + i]          # ← 与 kernel 的 cGm[row0*n2d + colGate + i] 对应
                u = C[row][colUp + i]            # ← 对应 cGm[row0*n2d + colUp + i]
                y[row][pairIdx * subN + i] = situ(g, beta) * tanh_scale(u, linear_beta)
    return y


# --------------------------------------------------------------------------
# golden：不做融合 —— 直接对 [W_gate; W_up] 的 GEMM 结果算 situ
#   与 kernels/SituAndMul + reference/situ.py 等价
# --------------------------------------------------------------------------
def golden_path(x, gate, up, M, H, d, beta, linear_beta):
    cg = gemm(x, gate, M, H, d)
    cu = gemm(x, up, M, H, d)
    return [[situ(cg[r][j], beta) * tanh_scale(cu[r][j], linear_beta) for j in range(d)]
            for r in range(M)]


def max_abs_diff(a, b):
    return max(abs(a[r][c] - b[r][c]) for r in range(len(a)) for c in range(len(a[0])))


def first_mismatch(a, b, tol=1e-12):
    for r in range(len(a)):
        for c in range(len(a[0])):
            if abs(a[r][c] - b[r][c]) > tol:
                return r, c, a[r][c], b[r][c]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=3, help="token 行数（小规模即可）")
    ap.add_argument("--H", type=int, default=4, help="hidden_size（K 维）")
    ap.add_argument("--d", type=int, default=8, help="intermediate_size")
    ap.add_argument("--subN", type=int, default=4, help="epilogue 片宽；必须整除 d")
    ap.add_argument("--beta", type=float, default=4.0)
    ap.add_argument("--linear-beta", dest="linear_beta", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    M, H, d, subN = args.M, args.H, args.d, args.subN
    print("=" * 78)
    print(f"融合算子索引自检：M={M} H={H} d={d} subN={subN} "
          f"beta={args.beta} linear_beta={args.linear_beta}")
    print("=" * 78)

    if d % subN != 0:
        print(f"❌ 前提不成立：d={d} 必须被 subN={subN} 整除（权重重排工具的前提，host 侧会拒）")
        return 1
    if subN % 16 != 0:
        print(f"⚠️  注意：subN={subN} 不是 16 的倍数。真机上 host 会另选 subN（UB/对齐约束），"
              f"这里只验索引逻辑。")

    rng = random.Random(args.seed)
    x = [[rng.uniform(-2, 2) for _ in range(H)] for _ in range(M)]
    gate = [[rng.uniform(-2, 2) for _ in range(H)] for _ in range(d)]
    up = [[rng.uniform(-2, 2) for _ in range(H)] for _ in range(d)]

    # ---- ① 正向：重排 → GEMM → kernel 公式，应与 golden 一致 ----
    wp = interleave(gate, up, d, subN)
    C = gemm(x, wp, M, H, 2 * d)                 # 模拟融合版的 Matmul 输出 C[M, 2d]
    y_kernel = kernel_path(C, M, d, subN, args.beta, args.linear_beta)
    y_golden = golden_path(x, gate, up, M, H, d, args.beta, args.linear_beta)

    diff = max_abs_diff(y_kernel, y_golden)
    print(f"① 正向（重排 + kernel 公式 vs golden）：max|Δ| = {diff:.3e}")
    if diff > 1e-12:
        bad = first_mismatch(y_kernel, y_golden)
        print(f"❌ FAIL —— 第一个不符的位置 (row={bad[0]}, col={bad[1]}): "
              f"kernel={bad[2]:.6f} golden={bad[3]:.6f}")
        print("   ⇒ 取数公式或重排规则不自洽，**上机前必须先修**（否则真机上会静默算错）")
        return 1
    print("✅ PASS —— 索引公式与重排约定自洽（上机前的必要条件已满足）")

    # ---- ② 反向自证：故意不重排，必须算错（否则这个检查恒过，无意义） ----
    wp_bad = [row[:] for row in gate] + [row[:] for row in up]     # 原始布局：gate 在前、up 在后
    C_bad = gemm(x, wp_bad, M, H, 2 * d)
    y_bad = kernel_path(C_bad, M, d, subN, args.beta, args.linear_beta)
    diff_bad = max_abs_diff(y_bad, y_golden)
    print(f"② 反向（**故意不做重排**）：max|Δ| = {diff_bad:.3e}")
    if diff_bad <= 1e-12:
        print("❌ 自证失败 —— 不重排也能过，说明这个检查测不出问题，别信它的 PASS")
        return 1
    print("✅ 反向确认为 FAIL（说明检查有效）")

    print()
    print("结论：索引/配对逻辑没问题。真机剩下的不确定性只有 AscendC 行为本身 ——")
    print("      DataCopyPad 的单位与 UB 行距、Matmul 的 C 行距、SyncAll 的可见性、")
    print("      GetBlockIdx() 在 AIV 上的取值范围（见 README §6 的 11 条）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
