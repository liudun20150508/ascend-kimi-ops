"""融合算子 SituGluFused（aclnnSituGluFused）的 torch 侧 harness + 对拍。

与 `test_situ_real.py` 同构（JIT 编译 wrapper），区别在两处：
  ① 融合算子有两个输入（x[M,H] + weight[2d,H]），且 **weight 必须先做交错重排**；
  ② 重排需要的 `subN` **只有 host tiling 算出来才知道** ⇒ 只能"先空跑一次 → 读
     `/tmp/situ_glu_fused_tiling.txt` → 按真实 subN 重排 → 再正式跑"（verify_situ_glu_fused_aclnn.cpp 同款）。

用法：
    /usr/local/python3.11.15/bin/python3 test_situ_glu_fused_real.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch
import torch_npu
from reference.situ import situ_and_mul as situ_golden

# ★ 算子的 JIT 包装 + 路径解析统一在 aclnn_ops.py（唯一实现）
import aclnn_ops

TILING_DUMP = aclnn_ops.TILING_DUMP      # host tiling 落盘（由 op_host 写）

# 兼容保留：这几个常量原先在本文件里定义（可能有外部引用），口径以 aclnn_ops 为准
CANN_PATH = aclnn_ops.CANN_PATH
VENDOR = aclnn_ops.vendor_dir("situ_glu_fused")
WRAPPER_SRC = aclnn_ops.wrapper_src("situ_glu_fused")
PKG_PATH = aclnn_ops.PKG_PATH


def run_real(x, w_permuted, beta, linear_beta):
    """x:[M,H] bf16, w_permuted:[2d,H] bf16（**已按 subN 交错重排**）→ y:[M,d]"""
    return aclnn_ops.situ_glu_fused(x, w_permuted, beta, linear_beta)


# --------------------------------------------------------------------------
# 权重重排 + 两段式便捷入口 —— 实现已移到 aclnn_ops（算子特有的前置约束，属运行时）
#   这里保留同名转发，既有调用方（README、integration 自检）继续可用。
# --------------------------------------------------------------------------
interleave = aclnn_ops.interleave
read_subn = aclnn_ops.read_subn


def run_fused(x, gate, up, beta=4.0, linear_beta=25.0, subn=None):
    """便捷入口：给**原始布局**的 gate/up（各 [d,H]），自动完成重排（两段式）。

    ⚠️ 重排需要 subN，而 subN 只有跑过一次才知道 ⇒ 先空跑让 host 落盘 tiling →
       读 subN → 重排 → 正式跑。实现见 aclnn_ops.situ_glu_fused_auto()。
    """
    return aclnn_ops.situ_glu_fused_auto(x, gate, up, beta, linear_beta, subn)


def check(M=8, H=64, d=128, beta=4.0, linear_beta=25.0, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(M, H, dtype=torch.bfloat16) * 2.0).npu()
    gate = (torch.randn(d, H, dtype=torch.bfloat16) * 0.1).npu()
    up = (torch.randn(d, H, dtype=torch.bfloat16) * 0.1).npu()

    y, subn = run_fused(x, gate, up, beta, linear_beta)

    # golden：原始布局的等价实现（cat 后一次 matmul + situ）
    w_orig = torch.cat([gate, up], dim=0)                    # [2d, H]
    gate_up = (x.float() @ w_orig.float().T).to(torch.bfloat16)
    ref = situ_golden(gate_up.cpu(), beta=beta, linear_beta=linear_beta).to(y.dtype)

    out_cpu = y.cpu()
    assert out_cpu.shape == ref.shape, f"shape mismatch: {out_cpu.shape} vs {ref.shape}"
    err = (out_cpu.float() - ref.float()).abs()
    tol = 1e-2 + 1e-2 * ref.float().abs()
    nbad = int((err > tol).sum().item())
    rel = (err / (ref.float().abs() + 1e-3)).max().item()
    print(f"[M={M} H={H} d={d}] subN={subn}  max_abs={err.max().item():.3e}  "
          f"max_rel={rel:.3e}  超容差={nbad}/{ref.numel()}  {'OK' if nbad == 0 else 'FAIL'}")
    return nbad == 0


if __name__ == "__main__":
    print(f"device: {torch.npu.get_device_name(0)}")
    cases = [(8, 64, 128), (16, 128, 256), (32, 256, 512), (64, 512, 1024)]
    all_ok = True
    for M, H, d in cases:
        all_ok &= check(M, H, d)
    print("\n" + ("ALL PASS" if all_ok else "SOME FAIL"))
    sys.exit(0 if all_ok else 1)
