#!/usr/bin/env python3
"""L2 验证：把 SiTU-GLU 接进**真实 K3 模型层**（官方 `KimiMLP`），同权重对比。

它回答的是 L1（算子级 microbench）答不了的两个问题：
  ① 算子被**真实模型层**调用时，结果是否还正确？（同权重 + 同输入 ⇒ 差异只能来自算子）
  ② 在**整层 MLP** 的尺度上，换掉 act_fn 到底快多少？（Amdahl：算子级 4.45x 会被 GEMM 稀释）

用法
----
    # 推荐：用持久化 venv（见文件尾「环境准备」）
    /workspace/venv_l2/bin/python kimi3_ops/scripts/l2_mlp_check.py            # 默认真实尺寸
    /workspace/venv_l2/bin/python kimi3_ops/scripts/l2_mlp_check.py --tiny     # 秒级自测
    /workspace/venv_l2/bin/python kimi3_ops/scripts/l2_mlp_check.py 4096 11008 512   # 自定义 H D M

为什么不跑整个 K3（而只跑 MLP 层）
----------------------------------
K3 的前向要 KDA（依赖 `fla`），而 `fla` 的 import 会拉起 triton driver；本机
`triton-ascend 3.2.0` 与 CANN 9.0.0 **不配套**（`npu_utils.cpp` 编译失败：
`RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE` 不在本机 driver 头里）。
但 **SiTU-GLU 的挂载点就在 `KimiMLP`**（`reference/modeling_kimi_linear.py:273`），
验证它完全不需要 KDA ⇒ 给 `fla.*` 打 stub，仍然用**官方类**，保留"真实模型层"的说服力。

环境准备（一次性；装进 /workspace 以免容器重建丢失）
---------------------------------------------------
    /usr/local/python3.11.15/bin/python3.11 -m venv --system-site-packages /workspace/venv_l2
    /workspace/venv_l2/bin/pip install -i https://mirrors.aliyun.com/pypi/simple/ \\
        'transformers>=4.56,<5' einops fla-core
    # 注意 transformers 必须是 4.x：5.x 移除了 reference 代码依赖的 OutputRecorder
    # 本机 pypi 直连约 20 KB/s，阿里云源约 835 KB/s —— 差异巨大，别用默认源
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
KIMI3_ROOT = os.path.dirname(HERE)
sys.path.insert(0, KIMI3_ROOT)
sys.path.insert(0, os.path.join(KIMI3_ROOT, 'integration'))

# ------------------------------------------------- 环境（aclnn_common.h 的来源）
# JIT 编译 wrapper 需要 aclnn_common.h（ACLNN_CMD 宏），它来自华为的 cann_ops_transformer 工具包。
# 注意：/tmp 那份会随容器重建丢失，/workspace 那份是持久副本 ⇒ 优先用 /workspace。
for _cand in ('/workspace/ops-transformer/torch_extension/cann_ops_transformer',
              '/tmp/ops-transformer/torch_extension/cann_ops_transformer'):
    if os.path.isdir(_cand):
        os.environ.setdefault('OPS_TRANSFORMER_PKG', _cand)
        break
os.environ.setdefault('ASCEND_HOME_PATH', '/usr/local/Ascend/cann-9.0.0')
os.environ.setdefault('KIMI3_VENDOR', os.path.join(os.environ['ASCEND_HOME_PATH'], 'opp/vendors/customize'))
os.environ.setdefault('ASCEND_CUSTOM_OPP_PATH', os.environ['KIMI3_VENDOR'])

# ---------------------------------------------------------------- fla stub
# `modeling_kimi_linear.py` 顶层 `from fla.* import ...` 会触发 triton driver 初始化，
# 本机 triton-ascend 与 CANN 不配套 ⇒ 编不过。KimiMLP 不用 fla（只有 KDA 类才用），
# 所以把 fla 相关模块塞成 stub 就能安全 import。
from unittest import mock  # noqa: E402

for _m in ['fla', 'fla.modules', 'fla.ops', 'fla.ops.kda', 'fla.ops.utils',
           'fla.ops.utils.index', 'fla.utils']:
    sys.modules[_m] = mock.MagicMock()

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

from reference.configuration_kimi_k3 import KimiLinearConfig  # noqa: E402
from reference.modeling_kimi_linear import KimiMLP  # noqa: E402
from reference.situ import situ_and_mul as situ_ref  # noqa: E402
import torch_ops_kimi3  # noqa: E402,F401  import 即注册 torch.ops.kimi3_ops.*
from torch import nn  # noqa: E402
from vllm_situ_glu import SituAndMulModule  # noqa: E402  接入层现成的"替换 act_fn"模块

PRESETS = {
    'tiny': (256, 512, 8),
    'real': (1024, 11008, 128),   # d=11008 是 K3 真实 intermediate_size
}


class RefSituGlu(nn.Module):
    """把 reference（纯 torch 实现）包成 Module，用于同口径时延对比。"""

    def __init__(self, beta, linear_beta):
        super().__init__()
        self.beta, self.linear_beta = beta, linear_beta

    def forward(self, t):
        return situ_ref(t, self.beta, self.linear_beta)


def bench(fn, iters=30, warmup=5):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - t0) / iters * 1e6   # μs


def main():
    ap = argparse.ArgumentParser(description='L2：SiTU-GLU 接进官方 KimiMLP 的同权重对比')
    ap.add_argument('H', nargs='?', type=int, help='hidden_size')
    ap.add_argument('D', nargs='?', type=int, help='intermediate_size')
    ap.add_argument('M', nargs='?', type=int, help='token 数（prefill 长度）')
    ap.add_argument('--tiny', action='store_true', help='用小尺寸快速自测')
    ap.add_argument('--iters', type=int, default=30, help='时延测的迭代次数')
    args = ap.parse_args()

    if args.tiny:
        H, D, M = PRESETS['tiny']
    elif args.H and args.D and args.M:
        H, D, M = args.H, args.D, args.M
    else:
        H, D, M = PRESETS['real']

    if not torch.npu.is_available():
        print('✗ NPU 不可用（本脚本验证的是 NPU kernel，必须上真机）')
        return 1

    torch.manual_seed(0)
    cfg = KimiLinearConfig(
        hidden_size=H, intermediate_size=D, num_hidden_layers=2,
        num_attention_heads=4, vocab_size=1024, hidden_act='situ',
        # ⚠️ K3 的真实取值；不给的话 _get_situ_activation_params 会回落成 beta=1.0 / linear_beta=None
        activation_situ_beta=4.0, activation_situ_linear_beta=25.0,
    )
    mlp = KimiMLP(cfg).to(torch.bfloat16).npu().eval()
    x = torch.randn(M, H, dtype=torch.bfloat16).npu()

    beta = float(getattr(mlp.act_fn, 'beta', 4.0))
    lb = float(getattr(mlp.act_fn, 'linear_beta', 25.0))
    nparam = sum(p.numel() for p in mlp.parameters())

    print('=' * 70)
    print(f'官方 KimiMLP：hidden={H} intermediate={D} 参数量={nparam:,}')
    print(f'  act_fn（原生）= {type(mlp.act_fn).__name__}(beta={beta}, linear_beta={lb})')
    print(f'  输入 x = {tuple(x.shape)} {x.dtype} {x.device}')
    print('=' * 70)

    # ① 原生 act_fn
    with torch.no_grad():
        y_native = mlp(x)
    torch.npu.synchronize()

    # ② 换成我的 AscendC 算子 —— 同一个 mlp 实例、同一份权重、同一个 x
    mlp.act_fn = SituAndMulModule(beta=beta, linear_beta=lb)
    with torch.no_grad():
        y_mine = mlp(x)
    torch.npu.synchronize()

    # ③ 数值对比（判据带绝对项，避免 |ref|≈0 处误判）
    a = y_native.float().cpu()
    b = y_mine.float().cpu()
    diff = (a - b).abs()
    tol = 1e-2 + 1e-2 * a.abs()
    nbad = int((diff > tol).sum().item())
    print('\n【数值】同权重、同输入，官方 KimiMLP 的输出差异')
    print(f'  max|diff|   = {diff.max().item():.3e}')
    print(f'  超容差      = {nbad} / {a.numel()}   ->  '
          f'{"✅ 与原生一致" if nbad == 0 else "❌ 不一致"}')
    print(f'  |y| 量级    = max {a.abs().max().item():.3f} / mean {a.abs().mean().item():.3f}')

    # ④ 时延对比（整层 MLP）
    mlp.act_fn = SituAndMulModule(beta=beta, linear_beta=lb)
    t_mine = bench(lambda: mlp(x), args.iters)
    mlp.act_fn = RefSituGlu(beta, lb)
    t_ref = bench(lambda: mlp(x), args.iters)

    print(f'\n【时延】整层 MLP（3 个 Linear + act_fn，M={M}）')
    print(f'  act_fn = 我的算子（AscendC）      : {t_mine:8.1f} μs')
    print(f'  act_fn = reference（纯 torch）    : {t_ref:8.1f} μs')
    print(f'  差值                              : {t_ref - t_mine:+8.1f} μs')
    if t_mine > 0:
        print(f'  整层加速                          : {t_ref / t_mine:.2f}x')
    print('\n注：这里测的是**整层 MLP**（含 3 个 Linear），act_fn 的差异会被 GEMM 稀释——')
    print('    这正是 Amdahl 的体现：算子级 4.45x 到层级别会缩到 x.x 倍。')
    return 0 if nbad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
