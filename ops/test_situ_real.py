"""真实 SituAndMul (aclnnSituAndMul) vs 参考实现 (reference/situ.py) 对拍。

在 NPU 上跑自定义 AscendC 算子，与纯 torch golden 交叉对比数值误差。
不依赖 cann_ops_transformer 完整 import，用 torch.utils.cpp_extension JIT 编译包装。

用法：
    python test_situ_real.py
"""

import os
import sys

sys.path.insert(0, "/workspace/kimi3_ops")

import torch
import torch_npu
from torch.utils.cpp_extension import load

from reference.situ import situ_and_mul as situ_golden

VENDOR = "/usr/local/Ascend/cann-9.0.0/opp/vendors/customize"

# 自定义算子的环境（等价于 source set_env.bash）
os.environ["ASCEND_CUSTOM_OPP_PATH"] = VENDOR + ":" + os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
os.environ["LD_LIBRARY_PATH"] = VENDOR + "/op_api/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

_OP = None


def _build():
    global _OP
    if _OP is not None:
        return _OP
    cann_path = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")
    tn_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
    src = "/workspace/kimi3_ops/situ_and_mul_wrapper.cpp"
    pkg_path = "/tmp/ops-transformer/torch_extension/cann_ops_transformer"
    abi = 1 if torch.compiled_with_cxx11_abi() else 0
    _OP = load(
        name="situ_and_mul_op",
        sources=[src],
        extra_include_paths=[
            os.path.join(tn_path, "include"),
            os.path.join(tn_path, "include/third_party/hccl/inc"),
            os.path.join(tn_path, "include/third_party/acl/inc"),
            os.path.join(cann_path, "include"),
            os.path.join(pkg_path, "common/inc"),
            os.path.join(VENDOR, "op_api/include"),
        ],
        extra_cflags=[
            "-fPIC", "-O2", "-w", f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        ],
        extra_ldflags=[
            "-L" + os.path.join(cann_path, "lib64"), "-lascendcl",
            "-L" + os.path.join(tn_path, "lib"), "-ltorch_npu",
            "-L" + os.path.join(VENDOR, "op_api/lib"), "-lcust_opapi",
        ],
        verbose=False,
    )
    return _OP


def run_real(x, beta, linear_beta):
    return _build().situ_and_mul(x, float(beta), float(linear_beta))


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
