"""SiTU-GLU 性能基准：自定义 AscendC 算子 vs torch 原生算子组合。

测三件事（这是追问里「快多少」那句话的直接证据）：
  1. 延时      —— NPU Event 计时，warmup + 多次取中位
  2. 有效带宽  —— 搬运字节数 / 延时
  3. 带宽达标率 —— 有效带宽 / 该硬件实测 HBM 峰值

为什么 elementwise 算子要按带宽判合格：
  SiTU-GLU 每个输出元素只做几次 FMA（AI ≈ 0.5 FLOP/B），必然访存受限，
  理论上限时间 = 搬运字节数 / HBM 带宽。做到 80% 以上才算「优化到位」，
  达不到就是流水没重叠（TQue depth=1）、分块不合理或访存未对齐。

用法：
    python bench_situ_real.py                       # 默认几组 KIMI3 真实 shape
    python bench_situ_real.py --peak-bw 1600        # 填入真机实测 HBM 带宽(GB/s)
    SITU_PEAK_BW_GBPS=1600 python bench_situ_real.py

拿实测峰值带宽的办法（A3 上，跑一次就够，填进上面参数）：
    msprof --application="python bench_situ_real.py --bench-bw-only" --output=./prof
    # 或直接看带宽压测：无自带工具时用一个大 copy kernel 计时，字节数/时间即为实测峰值
    # 通用做法：python -c "import torch,torch_npu; ..."  见 --bench-bw-only 分支

精确到单 kernel 的拆解（可选，需 msprof）：
    msprof --application="python bench_situ_real.py" --output=./prof --ai-core=on
    # 看 op_summary 里 situ_and_mul 的 aiv_time 与 MTE2/V/MTE3 占比
"""

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch_npu

from reference.situ import situ_and_mul as situ_golden  # noqa: E402
import aclnn_ops  # noqa: E402  复用 aclnn 运行时（唯一实现，含自定义算子环境变量设置）

# A3（ascend910_93）单卡 HBM 峰值带宽的**占位值**：必须用 --peak-bw 覆盖为实测值，
# 否则达标率没有意义。默认取 1600 GB/s 只是为了让脚本能跑出数来。
DEFAULT_PEAK_BW_GBPS = 1600.0


def bytes_moved(shape, elem_bytes=2):
    """SiTU-GLU 的访存字节数：读 gate(2d) + 读 up(2d) + 写 out(d)，bf16 = 2B。

    注意这里只算「必要流量」——如果 kernel 里多做了一次 Cast 往返、
    或 MTE2/MTE3 没有重叠导致实际搬运被放大，实测比值会低于 100%。
    """
    d = shape[-1] // 2
    rows = 1
    for s in shape[:-1]:
        rows *= s
    return rows * (2 * d + 2 * d + d) * elem_bytes


def timeit(fn, iters, warmup=10):
    """NPU 事件计时，返回单次耗时（毫秒）。取多次迭代的平均，另跑一遍取中位数更稳。"""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def torch_reference(x, beta, linear_beta):
    """torch 原生算子组合（非融合）：模拟「没有自定义算子时」要走的路。

    这正是自定义 kernel 要打败的基线 —— 它至少要落 3~4 个中间张量到 HBM。
    """
    gate, up = x.chunk(2, dim=-1)
    situ = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    up_t = linear_beta * torch.tanh(up / linear_beta)
    return situ * up_t


def bench_bw_only(iters=50):
    """测本机 HBM 峰值带宽（GB/s）：一个大张量 copy，字节数 = 2 × 张量大小。

    结果填给 --peak-bw。注意 device-to-device copy 走 HBM 读写各一遍。
    """
    n = 256 * 1024 * 1024  # 256M 元素 = 512MB(bf16)
    src = torch.empty(n, dtype=torch.bfloat16).npu().normal_()
    dst = torch.empty_like(src)
    ms = timeit(lambda: dst.copy_(src), iters)
    gbps = (n * 2 * 2) / (ms * 1e-3) / 1e9  # 读+写，bf16=2B
    print(f"[峰值带宽] copy 512MB×2: {ms:.3f} ms → {gbps:.1f} GB/s")
    return gbps


def check_and_bench(shape, beta, linear_beta, peak_bw_gbps, iters):
    torch.manual_seed(0)
    x = (torch.randn(shape, dtype=torch.bfloat16) * 6.0).npu()

    ref = situ_golden(x.cpu(), beta=beta, linear_beta=linear_beta).to(torch.bfloat16)
    out = aclnn_ops.situ_and_mul(x, beta, linear_beta)
    err = (out.cpu().float() - ref.float()).abs()
    rel = (err / (ref.float().abs() + 1e-3)).max().item()

    t_op = timeit(lambda: aclnn_ops.situ_and_mul(x, beta, linear_beta), iters)
    t_ref = timeit(lambda: torch_reference(x, beta, linear_beta), iters)

    nbytes = bytes_moved(shape)
    bw_op = nbytes / (t_op * 1e-3) / 1e9          # GB/s
    bw_ref = nbytes / (t_ref * 1e-3) / 1e9
    utilization = bw_op / peak_bw_gbps * 100.0     # 达标率（%）
    lower_bound_ms = nbytes / (peak_bw_gbps * 1e9) * 1e3

    print(f"\nshape={shape}  d={shape[-1]//2}  "
          f"搬运={nbytes/1e6:.2f} MB  max_rel={rel:.2e} {'OK' if rel < 1e-2 else 'FAIL'}")
    print(f"  理论下限 (峰值 {peak_bw_gbps:.0f} GB/s) : {lower_bound_ms:.4f} ms")
    print(f"  自定义算子                            : {t_op:.4f} ms  {bw_op:7.1f} GB/s  达标率 {utilization:5.1f}%")
    print(f"  torch 算子组合(基线)                  : {t_ref:.4f} ms  {bw_ref:7.1f} GB/s")
    print(f"  加速比 vs torch 组合                   : {t_ref/t_op:.2f}x")
    return utilization


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak-bw", type=float,
                    default=float(os.environ.get("SITU_PEAK_BW_GBPS", DEFAULT_PEAK_BW_GBPS)),
                    help="真机实测 HBM 峰值带宽 GB/s（默认占位值，务必覆盖）")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--bench-bw-only", action="store_true", help="只测本机峰值带宽并退出")
    ap.add_argument("--assert-util", default="",
                    help="判据：搬运量 ≥ MB 的 case 其达标率必须 ≥ pct，格式 'MB:pct'（如 '100:80'）。"
                         "只对**大搬运**的 case 判定 —— 小 shape 被启动开销主导，不在判据适用域内")
    args = ap.parse_args()

    print(f"device: {torch.npu.get_device_name(0)}")

    if args.bench_bw_only:
        bench_bw_only(args.iters)
        return

    if args.peak_bw == DEFAULT_PEAK_BW_GBPS:
        print("!! --peak-bw 未覆盖，达标率基于占位值，仅用于验证脚本通路；"
              "正式数字请先跑 --bench-bw-only 得到实测峰值")

    # KIMI3 真实 intermediate_size = 11008；再补几组控制变量 shape
    cases = [
        ((1, 1, 22016), 4.0, 25.0),      # d=11008（KIMI3 真实）
        ((2, 7, 22016), 4.0, 25.0),
        ((8, 512, 22016), 4.0, 25.0),    # 大 batch：访存受限应更充分
        ((2, 7, 4096), 4.0, 25.0),       # d=2048
        ((2, 7, 256), 4.0, 25.0),        # d=128：小块，启动开销占比高
    ]

    utils = []
    for shape, b, lb in cases:
        utils.append(check_and_bench(shape, b, lb, args.peak_bw, args.iters))

    print("\n==== 汇总 ====")
    for (shape, _, _), u in zip(cases, utils):
        verdict = "达标(>=80%)" if u >= 80 else ("接近(60~80%)" if u >= 60 else "未达标(<60%)")
        print(f"  {str(shape):>22}  达标率 {u:5.1f}%  {verdict}")
    print(f"\n中位达标率 {statistics.median(utils):.1f}%  "
          f"（峰值取值 {args.peak_bw:.0f} GB/s，非 80% 以上先查 double buffer / 分块）")

    # ---- 判据断言（只对适用域内的 case 生效）----
    if args.assert_util:
        mb_min, pct_min = (float(v) for v in args.assert_util.split(':'))
        checked, failed = 0, []
        for (shape, _, _), u in zip(cases, utils):
            mb = bytes_moved(shape) / 1e6
            if mb >= mb_min:
                checked += 1
                if u < pct_min:
                    failed.append((tuple(shape), u))
        print(f"\n[判据] 搬运量 ≥ {mb_min:g} MB 的 case 共 {checked} 个，要求达标率 ≥ {pct_min:g}%")
        if failed:
            for shape, u in failed:
                print(f"  ❌ {tuple(shape)} 达标率 {u:.1f}%")
            print("  ⇒ 判据未满足（注意：判据只对访存受限的大 case 成立）")
            sys.exit(1)
        print("  ✅ 全部满足（适用域：访存受限 & 大搬运）")
        print("  注：小 shape 不进判据 —— 它们的达标率被启动开销（15~20 μs）主导，"
              "正确判据是「启动开销 + 搬运」，见 docs/04")


if __name__ == "__main__":
    main()
