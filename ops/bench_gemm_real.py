"""GEMM 算力达标率（MFU）基准：把「快多少」变成一个可复算的百分比。

对标对象说明：NPU 上 `torch.matmul` 底层就是 `aclnnMatmul`，所以它是**基线**不是对手。
要证明自己的 kernel 有优势，应该比的是「融合算子」（GEMM + epilogue 一起吃进去）。

用法：
    python3 bench_gemm_real.py                    # 默认 shape 组
    python3 bench_gemm_real.py --bench-peak-only  # 先用一个大 GEMM 测本机 bf16 峰值算力
    python3 bench_gemm_real.py --peak-tflops 400  # 用实测峰值算 MFU
    python3 bench_gemm_real.py --m 4096 --n 4096 --k 4096

判断口径（详见 docs/05-GEMM-tiling推导与达标率.md）：
    MFU >= 70%  显式 GEMM 到位
    40~70%      正常，看是哪一维太小
    < 40%       tiling / 流水 / L2 复用有问题
    M=1（decode 的 GEMV）→ 必然访存受限，不能用 MFU 评判，要换带宽口径
"""

import argparse
import os
import sys

import torch
import torch_npu

DEFAULT_PEAK_TFLOPS = 400.0  # 占位值，务必用 --bench-peak-only 的实测替换


def timeit(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    s = torch.npu.Event(enable_timing=True)
    e = torch.npu.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.npu.synchronize()
    return s.elapsed_time(e) / iters  # ms


def flops(m, n, k):
    return 2.0 * m * n * k


def bench_one(m, n, k, peak_tflops, iters):
    a = torch.randn(m, k, dtype=torch.bfloat16).npu()
    b = torch.randn(k, n, dtype=torch.bfloat16).npu()
    ms = timeit(lambda: a @ b, iters)
    f = flops(m, n, k)
    tflops = f / (ms * 1e-3) / 1e12
    if peak_tflops > 1.0:
        mfu = tflops / peak_tflops * 100.0
        lower_ms = f / (peak_tflops * 1e12) * 1e3
        print(f"  M={m:>6} N={n:>6} K={k:>6}  {ms:8.4f} ms  {tflops:8.1f} TFLOPS  "
              f"MFU {mfu:5.1f}%  理论下限 {lower_ms:8.4f} ms")
    else:
        # 测峰值模式（peak=1.0 占位）：此时 MFU/理论下限无意义，只报绝对吞吐
        mfu = 0.0
        print(f"  M={m:>6} N={n:>6} K={k:>6}  {ms:8.4f} ms  {tflops:8.1f} TFLOPS")
    return tflops, mfu


def bench_peak(iters=20):
    """用大 GEMM 测本机可达到的 bf16 峰值（MFU 的分母）。"""
    print("测本机 bf16 GEMM 峰值（8192³，取多次）:")
    best = 0.0
    for _ in range(3):
        t, _ = bench_one(8192, 8192, 8192, 1.0, iters)  # peak=1.0 只看 TFLOPS
        best = max(best, t)
    print(f"\n→ 本机实测峰值参考值 ≈ {best:.1f} TFLOPS（填给 --peak-tflops）")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak-tflops", type=float,
                    default=float(os.environ.get("GEMM_PEAK_TFLOPS", DEFAULT_PEAK_TFLOPS)))
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--m", type=int)
    ap.add_argument("--n", type=int)
    ap.add_argument("--k", type=int)
    ap.add_argument("--bench-peak-only", action="store_true")
    args = ap.parse_args()

    print(f"device: {torch.npu.get_device_name(0)}")

    if args.bench_peak_only:
        bench_peak(args.iters)
        return

    if args.peak_tflops == DEFAULT_PEAK_TFLOPS:
        print("!! --peak-tflops 未覆盖，MFU 基于占位峰值；正式数字先跑 --bench-peak-only")

    if args.m and args.n and args.k:
        print(f"\n单点 shape（峰值 {args.peak_tflops:.0f} TFLOPS）:")
        bench_one(args.m, args.n, args.k, args.peak_tflops, args.iters)
        return

    print(f"\nA) 固定 M=N=4096，扫 K —— 看「K 越大越吃满算力」（峰值 {args.peak_tflops:.0f} TFLOPS）:")
    for k in (256, 1024, 4096, 16384):
        bench_one(4096, 4096, k, args.peak_tflops, args.iters)

    print(f"\nB) 固定 K=4096，扫 M —— 看「M 太小必然掉」（峰值 {args.peak_tflops:.0f} TFLOPS）:")
    for m in (1, 16, 128, 1024, 4096):
        bench_one(m, 4096, 4096, args.peak_tflops, args.iters)

    print("""
结论怎么读：
  * A 组：K 从 256 涨到 16384，MFU 应明显上升 —— L1 的 K 方向分块打满后 MTE1 与 MAD 才能重叠。
  * B 组：M=1 那行是 decode 阶段的 GEMV，必然访存受限（读整个权重只算一次），
          它的「达标率」要用带宽口径算，不是 MFU —— 这一条是常设的陷阱题。
  * 若 A/B 两组整体都低于 40%，先查 tiling（BaseM/BaseN/BaseK 与 stepKa/stepKb）与流水级数。""")


if __name__ == "__main__":
    sys.exit(main())
