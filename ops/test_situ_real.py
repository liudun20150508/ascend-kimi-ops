"""真实 SituAndMul (aclnnSituAndMul) vs 参考实现 (reference/situ.py) 对拍。

在 NPU 上跑自定义 AscendC 算子，与纯 torch golden 交叉对比数值误差。
不依赖 cann_ops_transformer 完整 import，用 torch.utils.cpp_extension JIT 编译包装。

用法：
    python test_situ_real.py
"""

import os
import sys

# 不写死绝对路径：本文件所在目录就是包根，跨机器（A2/A3）不用改代码
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch
import torch_npu
from reference.situ import situ_and_mul as situ_golden

# ★ 算子的 JIT 包装 + 路径解析统一在 aclnn_ops.py（唯一实现）——
#   接入层 integration/torch_ops_kimi3.py 与本 harness 都从那里调，本文件退回「只当测试」。
import aclnn_ops

# 兼容保留：这几个常量原先在本文件里定义（可能有外部引用），口径以 aclnn_ops 为准
CANN_PATH = aclnn_ops.CANN_PATH
VENDOR = aclnn_ops.vendor_dir("situ_and_mul")
WRAPPER_SRC = aclnn_ops.wrapper_src("situ_and_mul")
PKG_PATH = aclnn_ops.PKG_PATH


def run_real(x, beta, linear_beta):
    return aclnn_ops.situ_and_mul(x, beta, linear_beta)


def check(shape, beta=4.0, linear_beta=25.0, seed=0):
    torch.manual_seed(seed)
    # 输入 [..., 2d]，最后两半分别是 gate / up；值域放大一点以覆盖 tanh/sigmoid 的饱和区
    x = (torch.randn(shape, dtype=torch.bfloat16) * 6.0).npu()
    ref = situ_golden(x.cpu(), beta=beta, linear_beta=linear_beta)
    out = run_real(x, beta, linear_beta)
    out_cpu = out.cpu()

    ref = ref.to(out_cpu.dtype)
    err = (out_cpu.float() - ref.float()).abs()
    rel = (err / (ref.float().abs() + 1e-3)).max().item()
    max_abs = err.max().item()

    assert out_cpu.shape == ref.shape, f"shape mismatch: {out_cpu.shape} vs {ref.shape}"
    assert out_cpu.dtype == ref.dtype, f"dtype mismatch: {out_cpu.dtype} vs {ref.dtype}"

    print(f"[{str(shape):>24}] d={shape[-1]//2:>6}  "
          f"max_abs={max_abs:.3e}  max_rel={rel:.3e}  "
          f"{'OK' if rel < 1e-2 else 'FAIL'}")
    return rel < 1e-2


if __name__ == "__main__":
    print(f"device: {torch.npu.get_device_name(0)}")
    cases = [
        ((2, 3, 128), 4.0, 25.0),        # d=64
        ((4, 5, 256), 4.0, 25.0),        # d=128
        ((2, 7, 4096), 4.0, 25.0),       # d=2048
        ((2, 7, 22016), 4.0, 25.0),      # d=11008（KIMI3 真实 intermediate_size）
        ((3, 5, 256), 2.0, 8.0),         # 不同 beta/linear_beta
    ]
    all_ok = True
    for shape, b, lb in cases:
        all_ok &= check(shape, beta=b, linear_beta=lb)
    print("\n" + ("ALL PASS" if all_ok else "SOME FAIL"))
    sys.exit(0 if all_ok else 1)
