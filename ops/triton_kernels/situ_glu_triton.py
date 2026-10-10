"""SiTU-GLU 的 Triton 实现 —— 同一算子的第三种实现（三实现对比的第三极）。

三实现：CANN 内置算子（只有 SiLU，SiTU 缺失）→ AscendC 自研 kernel → Triton（本文件）。
对比结论见 ../docs/06-Triton-三实现对比.md。

★ 实测状态（2026-09-19，本机 A3）——**"装了"≠"能用"**，三件事分开看：

| 问题 | 答案 | 证据 |
|---|---|---|
| triton 装了吗？ | ✅ **装了** | `pip show triton-ascend` → `Version: 3.2.0` |
| `import triton` 行吗？ | ✅ 行（**但这是懒加载，什么都没证明**） | `import triton` → `3.2.0` |
| **能跑 kernel 吗？** | ❌ **不能** | driver 初始化时编 `npu_utils.cpp` 失败：`error: 'RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE' is not a member of 'rtLimitType_t'` |

⇒ **`triton-ascend 3.2.0` 与本机 CANN 9.0.0 驱动头不配套** ⇒ 本机**无法运行 Triton kernel**。
而 `vllm-ascend` 要的是 **3.2.2**（0.23.0）/ **3.2.1**（0.20~0.22），公开源（pypi + 华为镜像）**最高只有 3.2.0**
⇒ **两条路都断**：3.2.0 跑不了、3.2.1+ 拿不到。

▶ 本文件已改成**双设备兼容**（昇腾 / CUDA 共用一份代码），换到配套环境（或拿到 3.2.1+）可直接跑。

跑法（需要 GPU + triton；昇腾侧对应 triton-ascend）：
    python3 situ_glu_triton.py            # 正确性对拍 + 带宽达标率
    python3 situ_glu_triton.py --bench    # 只跑性能

为什么用 1D 展平 + 手算 gate/up 偏移，而不是 2D 分块：
    这是纯 elementwise 算子，访存模式决定一切。1D 展平让每个 program 的访存完全连续，
    BLOCK 用 2 的幂便于编译器向量化；2D 分块在这里只会引入额外的索引计算。
"""

import argparse
import os
import sys

import torch

try:
    import torch_npu  # noqa: F401  昇腾侧必须 import，才会挂上 torch.npu
except Exception:
    torch_npu = None

# ---- 设备适配（昇腾 / CUDA 共用一份代码）----
if hasattr(torch, 'npu') and torch.npu.is_available():
    DEV = 'npu'
    def sync(): torch.npu.synchronize()
    def new_event(): return torch.npu.Event(enable_timing=True)
    def dev_name(): return torch.npu.get_device_name(0)
    def dev_available(): return True
    DEFAULT_PEAK = 1271.6      # 本卡实测（torch copy 口径）
else:
    DEV = 'cuda'
    def sync(): torch.cuda.synchronize()
    def new_event(): return torch.cuda.Event(enable_timing=True)
    def dev_name(): return torch.cuda.get_device_name(0)
    def dev_available(): return torch.cuda.is_available()
    DEFAULT_PEAK = 1792.0            # A100 量级

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:  # 没有 triton 时仍可 import 本模块读取数学定义
    HAVE_TRITON = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if HAVE_TRITON:

    @triton.jit
    def situ_glu_kernel(x_ptr, y_ptr, n_out, d, beta, linear_beta,
                        BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_out

        # 输出第 offs 个元素对应输入里的 gate/up 两个位置（同一个 row，相差 d）
        row = offs // d
        col = offs % d
        gate_off = row * 2 * d + col
        up_off = gate_off + d

        gate = tl.load(x_ptr + gate_off, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(x_ptr + up_off, mask=mask, other=0.0).to(tl.float32)

        # situ(gate) = beta * tanh(gate/beta) * sigmoid(gate)
        situ = beta * tl.math.tanh(gate / beta) * tl.sigmoid(gate)
        # up' = linear_beta * tanh(up / linear_beta)
        up_t = linear_beta * tl.math.tanh(up / linear_beta)

        tl.store(y_ptr + offs, (situ * up_t).to(y_ptr.dtype.element_ty), mask=mask)


def situ_glu(x, beta=4.0, linear_beta=25.0, block=1024):
    """Triton 版 SiTU-GLU：x [..., 2d] -> [..., d]。"""
    assert HAVE_TRITON, "需要安装 triton（昇腾侧用 triton-ascend）"
    x = x.contiguous()
    d = x.shape[-1] // 2
    y = torch.empty(x.shape[:-1] + (d,), device=x.device, dtype=x.dtype)
    n_out = y.numel()
    grid = (triton.cdiv(n_out, block),)
    situ_glu_kernel[grid](x, y, n_out, d, float(beta), float(linear_beta), BLOCK=block)
    return y


def _ref(x, beta=4.0, linear_beta=25.0):
    from reference.situ import situ_and_mul
    return situ_and_mul(x.cpu(), beta=beta, linear_beta=linear_beta).to(x.dtype).to(x.device)


def _timeit(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    sync()
    s = new_event()
    e = new_event()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    sync()
    return s.elapsed_time(e) / iters


def check_and_bench(shape, beta, linear_beta, peak_bw_gbps, iters, block):
    x = torch.randn(shape, device=DEV, dtype=torch.bfloat16) * 6.0
    ref = _ref(x, beta, linear_beta)
    out = situ_glu(x, beta, linear_beta, block)

    rel = ((out.float() - ref.float()).abs() / (ref.float().abs() + 1e-3)).max().item()

    d = shape[-1] // 2
    rows = 1
    for s in shape[:-1]:
        rows *= s
    nbytes = rows * (2 * d + 2 * d + d) * 2  # 读 gate/up + 写 out，bf16

    ms = _timeit(lambda: situ_glu(x, beta, linear_beta, block), iters)
    bw = nbytes / (ms * 1e-3) / 1e9
    util = bw / peak_bw_gbps * 100 if peak_bw_gbps else float("nan")

    print(f"  {str(shape):>22}  max_rel={rel:.2e} {'OK' if rel < 1e-2 else 'FAIL'}  "
          f"{ms:7.4f} ms  {bw:7.1f} GB/s  达标率 {util:5.1f}%")
    return rel, util


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peak-bw", type=float, default=float(os.environ.get("PEAK_BW_GBPS", 0)))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    if not HAVE_TRITON:
        print("没装 triton，只打印数学定义：")
        print("  situ(gate) = beta * tanh(gate/beta) * sigmoid(gate)")
        print("  up'        = linear_beta * tanh(up/linear_beta)")
        print("  out        = situ(gate) * up'")
        return 0

    if not dev_available():
        print("没有可用的加速器（npu / cuda）")
        return 0

    if args.peak_bw <= 0:
        print("!! 未给 --peak-bw，达标率不显示（用 GPU 带宽实测值填入）")

    print(f"device: {dev_name()} (DEV={DEV})  BLOCK={args.block}")
    cases = [
        ((1, 1, 22016), 4.0, 25.0, DEFAULT_PEAK),   # d=11008，A100 带宽量级
        ((2, 7, 22016), 4.0, 25.0, DEFAULT_PEAK),
        ((8, 512, 22016), 4.0, 25.0, DEFAULT_PEAK),
        ((2, 7, 256), 4.0, 25.0, DEFAULT_PEAK),
    ]
    for shape, b, lb, default_bw in cases:
        check_and_bench(shape, b, lb, args.peak_bw or default_bw, args.iters, args.block)
    return 0


if __name__ == "__main__":
    sys.exit(main())
