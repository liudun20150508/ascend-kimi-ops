"""真实华为 chunk_kda_fwd（aclnnChunkKdaFwd）vs 参考实现对比。

在 NPU 上跑真实算子，与 /workspace/kimi3_ops/reference 的 chunk_kda_reference
（分块参考）交叉对比 attn_out 和 final_state 的数值误差。

运行前需设置（见 run.sh 或本文件顶部注释）：
    export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/custom_transformer
    export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH

不依赖 cann_ops_transformer 包的完整 import（该包在本机缺 torchair），
用 torch.utils.cpp_extension 直接 JIT 编译 chunk_kda_fwd.cpp。
"""

import os
import sys

sys.path.insert(0, "/workspace/kimi3_ops")

import math

import torch
import torch_npu
from torch.utils.cpp_extension import load

from reference.kda_chunked import chunk_kda_reference

_OP_MODULE = None


def _build_op():
    global _OP_MODULE
    if _OP_MODULE is not None:
        return _OP_MODULE
    cann_path = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")
    tn_path = os.path.dirname(os.path.abspath(torch_npu.__file__))
    pkg_path = "/tmp/ops-transformer/torch_extension/cann_ops_transformer"
    src = os.path.join(pkg_path, "ops/csrc/chunk_kda_fwd.cpp")
    abi = 1 if torch.compiled_with_cxx11_abi() else 0
    _OP_MODULE = load(
        name="chunk_kda_fwd",
        sources=[src],
        extra_include_paths=[
            os.path.join(tn_path, "include"),
            os.path.join(tn_path, "include/third_party/hccl/inc"),
            os.path.join(tn_path, "include/third_party/acl/inc"),
            os.path.join(cann_path, "include"),
            os.path.join(pkg_path, "common/inc"),
        ],
        extra_cflags=[
            "-fstack-protector-all", "-Wl,-z,relro,-z,now,-z,noexecstack",
            "-fPIC", "-pie", "-s", "-fvisibility=hidden", "-D_FORTIFY_SOURCE=2",
            "-O2", "-w", f"-D_GLIBCXX_USE_CXX11_ABI={abi}",
        ],
        extra_ldflags=[
            "-L" + os.path.join(cann_path, "lib64"), "-lascendcl",
            "-L" + os.path.join(tn_path, "lib"), "-ltorch_npu",
        ],
        verbose=False,
    )
    return _OP_MODULE


def run_real(q, k, v, g, beta, a_log, dt_bias, scale, chunk_size=64):
    """跑真实算子，返回 (attn_out, final_state)。BSND 布局。"""
    module = _build_op()
    outs = module.chunk_kda_fwd(
        q, k, v, g, beta,
        a_log, dt_bias, None,
        float(scale), int(chunk_size), True, -5.0, True, False, "BSND", True,
    )
    return outs[0], outs[1]  # attn_out, final_state


def make_inputs(B, T, H, HV, K, V, seed=0, device="npu"):
    """慢衰减输入（对齐 reference/tests/test_kda_chunked.py 的 fp32 用例约定）。

    gate 用 g = randn*0.5 - 3.0 + dt_bias=-5 使 gamma≈0（慢衰减），避免 E=exp(cumsum)
    在单 chunk T 下下溢导致 (I+L) 求逆奇异。这是 chunk delta-rule 的固有数值问题，
    参考实现用朴素 torch.linalg.inv 无法在快衰减下保持有限。
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    shape = (B, T, H, K)
    q = (torch.randn(shape, generator=gen) * 0.05).to(torch.bfloat16)
    k = (torch.randn(shape, generator=gen) * 0.05).to(torch.bfloat16)
    v = (torch.randn((B, T, HV, V), generator=gen) * 0.05).to(torch.bfloat16)
    g = (torch.randn((B, T, HV, K), generator=gen) * 0.5 - 3.0).float()
    beta = torch.sigmoid(torch.randn((B, T, HV), generator=gen)).to(torch.bfloat16)
    a_log = torch.zeros(HV, dtype=torch.float32)
    dt_bias = torch.full((HV * K,), -5.0, dtype=torch.float32)
    return q, k, v, g, beta, a_log, dt_bias


def rel_err(actual, expected):
    a = actual.float()
    e = expected.float()
    return ((a - e).norm() / (e.norm() + 1e-30)).item()


def max_abs_err(actual, expected):
    return (actual.float() - expected.float()).abs().max().item()


def compare_case(B, T, H, HV, K, V, C, tag):
    q, k, v, g, beta, a_log, dt_bias = make_inputs(B, T, H, HV, K, V)
    scale = K ** -0.5

    # 真实算子（NPU）
    q_n = q.npu(); k_n = k.npu(); v_n = v.npu()
    g_n = g.npu(); beta_n = beta.npu(); a_log_n = a_log.npu(); dt_bias_n = dt_bias.npu()
    o_real, S_real = run_real(q_n, k_n, v_n, g_n, beta_n, a_log_n, dt_bias_n, scale, C)
    torch.npu.synchronize()

    # 参考实现（CPU fp32 计算，chunk 公式与华为 kernel 数学一致）
    o_ref, S_ref = chunk_kda_reference(
        q, k, v, g, beta, a_log, dt_bias,
        chunk_size=C, lower_bound=-5.0, scale=scale, output_final_state=True)

    o_real_c = o_real.cpu()
    S_real_c = S_real.cpu()

    print(f"[{tag}] B={B} T={T} H={H} HV={HV} K={K} V={V} C={C}")
    print(f"  attn_out     shape real={tuple(o_real_c.shape)} ref={tuple(o_ref.shape)} "
          f"dtype real={o_real_c.dtype} ref={o_ref.dtype}")
    print(f"  final_state  shape real={tuple(S_real_c.shape)} ref={tuple(S_ref.shape)} "
          f"dtype real={S_real_c.dtype} ref={S_ref.dtype}")
    print(f"  attn_out     rel_err={rel_err(o_real_c, o_ref):.3e}  max_abs={max_abs_err(o_real_c, o_ref):.3e}")
    print(f"  final_state  rel_err={rel_err(S_real_c, S_ref):.3e}  max_abs={max_abs_err(S_real_c, S_ref):.3e}")
    print()
    return rel_err(o_real_c, o_ref), rel_err(S_real_c, S_ref)


def main():
    torch.npu.set_device(0)
    cases = [
        (1, 64, 2, 2, 16, 16, 64, "单chunk 小K"),
        (1, 128, 2, 2, 16, 16, 64, "多chunk 小K"),
        (1, 64, 2, 4, 32, 32, 64, "GVA H2→HV4 K32"),
        (1, 64, 96, 96, 128, 128, 64, "KIMI3 真实 shape K128"),
    ]
    for case in cases:
        try:
            compare_case(*case)
        except Exception as e:  # noqa: BLE001
            print(f"[{case[-1]}] 失败: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
