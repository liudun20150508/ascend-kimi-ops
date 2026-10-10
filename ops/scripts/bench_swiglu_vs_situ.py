#!/usr/bin/env python3
"""「相对华为提升多少」那一格：把 CANN **内置 SwiGLU** 和自研 **SiTU-GLU** 放在
同卡、同 shape、同判据下对测。

为什么只能这样比（**读这一节，别答错**）
----------------------------------------
华为的 GLU 家族里有 `SwiGlu` / `SwiGluQuant` / `ClippedSwiglu` / `GeGlu` / `Glu`
（CANN 910_93 内置算子清单，见 `docs/17` §1.1），**没有 SiTU 变体** —— 两者数学不同：

    SwiGLU : silu(gate) · up
    SiTU   : β·tanh(gate/β)·σ(gate) · linear_beta·tanh(up/linear_beta)

⇒ **不能比"谁算得对"**，只能比**带宽达标率**（同一个 shape、同一张卡、同一套计时口径）。
这就是 `docs/17` §5 里那格"还没填"的格子，本脚本填它。

用法
----
    # ① 先测本卡实测峰值带宽（不给进去，达标率没有意义）
    /usr/local/python3.11.15/bin/python3 scripts/bench_swiglu_vs_situ.py --bench-bw-only
    # ② 对测（把上一步的峰值填进 --peak-bw）
    /usr/local/python3.11.15/bin/python3 scripts/bench_swiglu_vs_situ.py --peak-bw 1208.7
    # 单组自定义 shape：rows=3584、d=11008
    /usr/local/python3.11.15/bin/python3 scripts/bench_swiglu_vs_situ.py --rows 3584 --d 11008 --peak-bw 1208.7

口径说明（两种会同时打印，别混用）
----------------------------------
  口径 A（**读 gate 2d + 读 up 2d + 写 out d = 5d 元素**）：与 `bench_situ_real.py`、
         `scripts/bench_situ_aclnn.cpp` **完全一致** ⇒ 只有它打印的达标率能和已记录的
         **81.5% / 77.6%** 并排比较（它把单输入张量按"读两遍"算，分子偏大 ⇒ 偏乐观）。
  口径 B（**读 2d + 写 out d = 3d 元素**）：单输入张量的**严格下限**，用来说"理论最短路径"。
  两个算子**同口径**，所以横向比较始终公平。

  ★ 口径陷阱（**实测踩到过，别再犯**）：拿 copy 测的峰值带宽（**1:1 读写**）当分母，
     对**读多写少**的算子会算出 **>100% 的达标率** —— 实测内置 `npu_swiglu` 在 `(8,512,22016)`
     上口径 A 算出 **179%**。⇒ **达标率 >100% 不是好成绩，是判据有 bug**；
     脚本会自动打警告。**对外只报“两算子同口径的时间/带宽比”，不报绝对达标率。**

读结果（三种情况的话术见 `docs/18` §2.3）
------------------------------------------
  我的 ≥ 内置       → 同类 GLU 算子同卡同判据不落下风（再补一句：内置还能吃量化 SwiGluQuant）
  我的略低(3~10 点) → 差在 tiling 硬编码 + 多余 Cast，改进项写清楚，比编数字强
  内置明显更高(>20) → 如实说没吃满带宽；**不许改判据、不许挑小 shape 糊弄**

⚠️ 本脚本**必须真机跑**（要 torch + torch_npu + 已安装的自定义算子）。
    没有 NPU 时它会打印 SKIP 而不是报错。
"""
import argparse
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIMI3_ROOT = os.path.dirname(HERE)
sys.path.insert(0, KIMI3_ROOT)

# ------------------------------------------------ 环境（自定义算子 + aclnn_common.h）
for _cand in ('/workspace/ops-transformer/torch_extension/cann_ops_transformer',
              '/tmp/ops-transformer/torch_extension/cann_ops_transformer'):
    if os.path.isdir(_cand):
        os.environ.setdefault('OPS_TRANSFORMER_PKG', _cand)
        break
os.environ.setdefault('ASCEND_HOME_PATH', '/usr/local/Ascend/cann-9.0.0')
os.environ.setdefault('KIMI3_VENDOR', os.path.join(os.environ['ASCEND_HOME_PATH'], 'opp/vendors/customize'))
os.environ.setdefault('ASCEND_CUSTOM_OPP_PATH', os.environ['KIMI3_VENDOR'])

try:
    import torch
    import torch_npu  # noqa: F401
except Exception as _e:  # noqa: BLE001
    print(f"[SKIP] 需要 torch + torch_npu（真机才能跑）：{_e}")
    sys.exit(0)

import aclnn_ops  # noqa: E402  复用 aclnn 运行时（三个算子共用，含环境变量设置）

# 占位峰值带宽：**必须用 --peak-bw 覆盖为实测值**，否则达标率没有意义（同 bench_situ_real.py）
DEFAULT_PEAK_BW_GBPS = 1600.0


def bytes_moved(shape, elem_bytes=2, read_gate_twice=True):
    """两个算子**统一**的"必要流量"口径。

    read_gate_twice=True  → 5d 元素（口径 A，与 docs/14 的 81.5%/77.6% 同口径）
    read_gate_twice=False → 3d 元素（口径 B，单输入张量的严格下限）
    """
    d = shape[-1] // 2
    rows = 1
    for s in shape[:-1]:
        rows *= s
    reads = 2 * d * (2 if read_gate_twice else 1)
    return rows * (reads + d) * elem_bytes


def timeit(fn, iters, warmup=10):
    """NPU Event 计时，返回单次耗时（ms）。"""
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


def bench_peak_bw(iters=50):
    """大张量 copy 测峰值带宽（GB/s）；结果填给 --peak-bw。"""
    n = 256 * 1024 * 1024  # 256M 元素 = 512MB(bf16)
    src = torch.empty(n, dtype=torch.bfloat16).npu().normal_()
    dst = torch.empty_like(src)
    ms = timeit(lambda: dst.copy_(src), iters)
    gbps = (n * 2 * 2) / (ms * 1e-3) / 1e9
    print(f"[峰值带宽] copy 512MB×2: {ms:.3f} ms → {gbps:.1f} GB/s")
    return gbps


def find_builtin_swiglu():
    """探测 CANN 内置 SwiGLU 的 Python 入口（不同 torch_npu 版本位置不同）。"""
    cands = [
        ('torch_npu.npu_swiglu', lambda: torch_npu.npu_swiglu),
        ('torch.ops.npu.npu_swiglu', lambda: torch.ops.npu.npu_swiglu),
        ('torch_npu.npu.npu_swiglu', lambda: torch_npu.npu.npu_swiglu),
    ]
    for name, get in cands:
        try:
            return name, get()
        except Exception:  # noqa: BLE001
            continue
    return None, None


def call_swiglu(fn, x):
    """内置 SwiGLU 的签名随版本略有差异（有的要 dim）⇒ 两种都试。"""
    try:
        return fn(x)
    except TypeError:
        return fn(x, -1)


def bench_case(shape, peak_bw, iters, situ_fn, swiglu_fn, swiglu_name):
    torch.manual_seed(0)
    x = (torch.randn(shape, dtype=torch.bfloat16) * 6.0).npu()

    t_situ = timeit(lambda: situ_fn(x), iters)
    t_swi = timeit(lambda: call_swiglu(swiglu_fn, x), iters)

    out = {}
    for tag, read_twice in (('A', True), ('B', False)):
        nb = bytes_moved(shape, read_gate_twice=read_twice)
        bw_situ = nb / (t_situ * 1e-3) / 1e9
        bw_swi = nb / (t_swi * 1e-3) / 1e9
        out[tag] = (nb, bw_situ / peak_bw * 100.0, bw_swi / peak_bw * 100.0)

    nb_a, u_situ_a, u_swi_a = out['A']
    _, u_situ_b, u_swi_b = out['B']

    print(f"\nshape={shape}  d={shape[-1] // 2}  搬运(口径A 5d)={nb_a / 1e6:.2f} MB  "
          f"理论下限 {nb_a / (peak_bw * 1e9) * 1e3:.4f} ms")
    print(f"  内置 {swiglu_name:<24} : {t_swi:.4f} ms   达标率 {u_swi_a:5.1f}% (口径A) / {u_swi_b:5.1f}% (口径B)")
    print(f"  自研 SiTU-GLU                : {t_situ:.4f} ms   达标率 {u_situ_a:5.1f}% (口径A) / {u_situ_b:5.1f}% (口径B)")
    print(f"  耗时比（自研 / 内置）         : {t_situ / t_swi:.2f}x   (<1 表示我的更快)")
    for _tag, _u in (('自研', u_situ_a), ('内置', u_swi_a)):
        if _u > 100.0:
            print(f"  ⚠️ 口径自检：{_tag} 达标率 {_u:.0f}% > 100% ⇒ **分母口径不适用于本算子**"
                  f"（copy 峰值是 1:1 读写，读数多于写数的算子真实上限高于它）。"
                  f"该格只能用于「两算子同口径横比」，不能对外报绝对达标率。")
    return u_situ_a, u_swi_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--peak-bw', type=float,
                    default=float(os.environ.get('SITU_PEAK_BW_GBPS', DEFAULT_PEAK_BW_GBPS)),
                    help='真机实测 HBM 峰值带宽 GB/s（默认占位值，务必覆盖）')
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--bench-bw-only', action='store_true', help='只测本机峰值带宽并退出')
    ap.add_argument('--rows', type=int, default=0, help='只测单组 shape：rows（配合 --d）')
    ap.add_argument('--d', type=int, default=11008, help='单组 shape 的 d（KIMI3 真实 11008）')
    ap.add_argument('--beta', type=float, default=4.0)
    ap.add_argument('--linear-beta', type=float, default=25.0)
    args = ap.parse_args()

    print(f'device: {torch.npu.get_device_name(0)}')

    if args.bench_bw_only:
        bench_peak_bw(args.iters)
        return

    if args.peak_bw == DEFAULT_PEAK_BW_GBPS:
        print('!! --peak-bw 未覆盖，达标率基于占位值，仅用于验证脚本通路；'
              '正式数字请先跑 --bench-bw-only 得到实测峰值')

    swiglu_name, swiglu_fn = find_builtin_swiglu()
    if swiglu_fn is None:
        print('[SKIP] 没找到内置 SwiGLU 的 Python 入口。请先核对本机 torch_npu 的接口名：\n'
              '    python3 -c "import torch,torch_npu; print([n for n in dir(torch_npu) if \'glu\' in n.lower()]);'
              ' print([n for n in dir(torch.ops.npu) if \'glu\' in n.lower()])"\n'
              '  或直接走 C++ 口径：用 `aclnnSwiGlu`（头文件 `aclnn_swi_glu.h`），\n'
              '  抄 `scripts/bench_situ_aclnn.cpp` 的骨架，把算子名换掉即可。')
        return
    print(f'内置 SwiGLU 入口: {swiglu_name}')

    situ_fn = lambda x: aclnn_ops.situ_and_mul(x, args.beta, args.linear_beta)  # noqa: E731

    if args.rows:
        cases = [(args.rows, args.d)]
    else:
        # 与 bench_situ_real.py 对齐的前三组（大 / 中 / 小）
        cases = [(8, 512, 22016), (2, 7, 22016), (2, 7, 4096)]

    pairs = []
    for shape in cases:
        pairs.append(bench_case(shape, args.peak_bw, args.iters, situ_fn, swiglu_fn, swiglu_name))

    print('\n==== 汇总（口径 A，与 docs/14 的 81.5%/77.6% 同口径）====')
    for shape, (u_situ, u_swi) in zip(cases, pairs):
        print(f'  {str(shape):>22}  自研 {u_situ:5.1f}%  vs  内置 {u_swi:5.1f}%   '
              f'差 {u_situ - u_swi:+.1f} 点')
    med_situ = statistics.median([p[0] for p in pairs])
    med_swi = statistics.median([p[1] for p in pairs])
    print(f'\n中位：自研 {med_situ:.1f}%  内置 {med_swi:.1f}%（峰值取值 {args.peak_bw:.0f} GB/s）')
    print('读法见文件头 docstring 与 docs/18 §2.3；'
          '注意小 shape 的达标率被启动开销主导，判据应换成「启动开销 + 搬运」。')


if __name__ == '__main__':
    main()
