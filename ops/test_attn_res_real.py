"""真实 AttnRes (aclnnAttnRes) vs 参考实现 (reference/attn_res.py) 对拍。"""

import os
import sys

sys.path.insert(0, "/workspace/kimi3_ops")

import torch
import torch_npu
from torch.utils.cpp_extension import load

from reference.attn_res import attn_res as attn_res_golden

VENDOR = "/usr/local/Ascend/cann-9.0.0/opp/vendors/customize"
os.environ["ASCEND_CUSTOM_OPP_PATH"] = VENDOR + ":" + os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
os.environ["LD_LIBRARY_PATH"] = VENDOR + "/op_api/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

_OP = None


def _build():
    global _OP
    if _OP is not None:
        return _OP
    cann = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")
    tn = os.path.dirname(os.path.abspath(torch_npu.__file__))
    src = "/workspace/kimi3_ops/attn_res_wrapper.cpp"
    pkg = "/tmp/ops-transformer/torch_extension/cann_ops_transformer"
    abi = 1 if torch.compiled_with_cxx11_abi() else 0
    _OP = load(
        name="attn_res_op",
        sources=[src],
        extra_include_paths=[
            os.path.join(tn, "include"),
            os.path.join(tn, "include/third_party/hccl/inc"),
            os.path.join(tn, "include/third_party/acl/inc"),
            os.path.join(cann, "include"),
            os.path.join(pkg, "common/inc"),
            os.path.join(VENDOR, "op_api/include"),
        ],
        extra_cflags=["-fPIC", "-O2", "-w", f"-D_GLIBCXX_USE_CXX11_ABI={abi}"],
        extra_ldflags=[
            "-L" + os.path.join(cann, "lib64"), "-lascendcl",
            "-L" + os.path.join(tn, "lib"), "-ltorch_npu",
            "-L" + os.path.join(VENDOR, "op_api/lib"), "-lcust_opapi",
        ],
        verbose=False,
    )
    return _OP


def run_real(br, ps, sw, eps):
    return _build().attn_res(br, ps, sw, float(eps))


def check(T, B, H, seed=0, eps=1e-6):
    torch.manual_seed(seed)
    br = (torch.randn(T, B, H, dtype=torch.bfloat16) * 2.0).npu()
    ps = (torch.randn(T, H, dtype=torch.bfloat16) * 2.0).npu()
    sw = (torch.randn(H, dtype=torch.bfloat16) * 0.5).npu()

    ref = attn_res_golden(br.cpu(), ps.cpu(), sw.cpu(), eps=eps)
    out = run_real(br, ps, sw, eps).cpu()

    assert out.shape == ref.shape, f"shape mismatch: {out.shape} vs {ref.shape}"
    assert out.dtype == ref.dtype, f"dtype mismatch: {out.dtype} vs {ref.dtype}"

    err = (out.float() - ref.float()).abs()
    rel = (err / (ref.float().abs() + 1e-3)).max().item()
    print(f"[T={T:>4} B={B:>2} H={H:>5}]  max_abs={err.max().item():.3e}  max_rel={rel:.3e}  "
          f"{'OK' if rel < 1e-2 else 'FAIL'}")
    return rel < 1e-2


if __name__ == "__main__":
    print(f"device: {torch.npu.get_device_name(0)}")
    cases = [
        (4, 2, 512),
        (8, 4, 4096),      # KIMI3 真实 hidden_size
        (16, 8, 4096),
        (32, 4, 1024),
        (2, 1, 512),       # 最小 B
    ]
    all_ok = True
    for T, B, H in cases:
        all_ok &= check(T, B, H)
    print("\n" + ("ALL PASS" if all_ok else "SOME FAIL"))
    sys.exit(0 if all_ok else 1)
