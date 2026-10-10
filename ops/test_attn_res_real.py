"""真实 AttnRes (aclnnAttnRes) vs 参考实现 (reference/attn_res.py) 对拍。"""

import os
import sys

# 不写死绝对路径（与 test_situ_real.py 同一套约定，换机器只设环境变量）
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch
import torch_npu
from reference.attn_res import attn_res as attn_res_golden

# ★ 算子的 JIT 包装 + 路径解析统一在 aclnn_ops.py（唯一实现）
import aclnn_ops

# 兼容保留：这几个常量原先在本文件里定义（可能有外部引用），口径以 aclnn_ops 为准
CANN_PATH = aclnn_ops.CANN_PATH
VENDOR = aclnn_ops.vendor_dir("attn_res")
WRAPPER_SRC = aclnn_ops.wrapper_src("attn_res")
PKG_PATH = aclnn_ops.PKG_PATH


def run_real(br, ps, sw, eps):
    return aclnn_ops.attn_res(br, ps, sw, eps)


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
