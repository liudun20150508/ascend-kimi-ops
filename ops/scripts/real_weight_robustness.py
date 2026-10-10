#!/usr/bin/env python3
"""真权重分布鲁棒性：**在 MXFP4 反量化 / outlier / 真实尺寸下，收益倍数会不会退化**。

为什么必须做这一步
------------------
本项目此前的验证用的是**随机高斯输入**。而 K3 的真实链路是：
`MXFP4 权重 → 软件反量化 → (动态 MXFP8 激活) → 进 SiTU-GLU`。
如果对方**真的加载 K3 权重实测**，第一个会被问的就是：

    「你这套数字是在真实权重分布下成立，还是只在随机数据上好看？」

本脚本把这件事先做了：造出**类真实分布**的输入，重新量两件事 ——
① **正确性**（判据 `max_rel < 1e-2`）② **收益倍数**（vs 等价 torch 组合）。

四类输入（同一 shape，只换分布）
------------------------------
  A `gauss`     : 纯高斯（此前所有数字的口径，作为对照基准）
  B `mxfp4`     : 经过 **MXFP4(E2M1 + 每 32 元素共享 E8M0 指数) 往返**的值 —— 模拟"权重经 4bit 反量化"
  C `outlier`   : 高斯 + 部分通道 ×50 —— 模拟 activation 的 outlier channel（K3 激活是动态 MXFP8 QAT）
  D `mxfp4out`  : B + C 叠加（最接近真实：量化台阶 + outlier 同时存在）

口径（跑之前就写好）
------------------
- 正确性：`max_rel < 1e-2`（与 `test_situ_real.py` 同判据）
- 收益倍数：**分母固定为「等价 torch 组合」**（6 个 elementwise），这是唯一正确的口径
  （华为没有 SiTU 变体，拿华为算子当分母是假命题）
- shape 用 **K3 真实 `intermediate_size = 11008`**（gate_up 末维 = 22016）

用法
----
    /usr/local/python3.11.15/bin/python3 scripts/real_weight_robustness.py
    /usr/local/python3.11.15/bin/python3 scripts/real_weight_robustness.py --rows 4096
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

for _c in ('/workspace/ops-transformer/torch_extension/cann_ops_transformer',
           '/tmp/ops-transformer/torch_extension/cann_ops_transformer'):
    if os.path.isdir(_c):
        os.environ.setdefault('OPS_TRANSFORMER_PKG', _c)
        break
os.environ.setdefault('ASCEND_HOME_PATH', '/usr/local/Ascend/cann-9.0.0')
os.environ.setdefault('KIMI3_VENDOR', os.environ['ASCEND_HOME_PATH'] + '/opp/vendors/customize')
os.environ.setdefault('ASCEND_CUSTOM_OPP_PATH', os.environ['KIMI3_VENDOR'])

import torch          # noqa: E402
import torch_npu      # noqa: E402,F401
import aclnn_ops      # noqa: E402
from reference.situ import situ_and_mul as golden  # noqa: E402

# MXFP4 的 E2M1 格点（与 mxfp4_dequant_check.py 同一套定义）
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
TOL = 1e-2


def mxfp4_roundtrip(x: torch.Tensor, group: int = 32) -> torch.Tensor:
    """向量化 MXFP4 往返：每 group 个元素共享一个 E8M0 指数，元素量化到 E2M1 格点。

    这就是「K3 权重在 A3 上必须做的软件反量化」的**逆过程**（量化→反量化），
    用它造出的分布带着真实的 **4-bit 量化台阶**。
    """
    orig_shape = x.shape
    xf = x.float().reshape(-1, group)
    bmax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-30)
    e = torch.floor(torch.log2(bmax)) - 2          # E2M1 的最大格点 6 → emax=2
    scale = torch.pow(torch.tensor(2.0, device=x.device), e)
    q = xf / scale
    idx = (q.abs().unsqueeze(-1) - E2M1.to(x.device)).abs().argmin(dim=-1)
    return (E2M1.to(x.device)[idx] * torch.sign(q) * scale).reshape(orig_shape)


def make_input(dist: str, shape, device, scale=6.0):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16, device='cpu') * scale
    if dist in ('mxfp4', 'mxfp4out'):
        x = mxfp4_roundtrip(x).to(torch.bfloat16)
    if dist in ('outlier', 'mxfp4out'):
        c = shape[-1]
        # 每 128 个通道里挑 1 个当 outlier（模拟 activation 的少数极值通道）
        idx = torch.arange(0, c, 128)
        x[..., idx] = x[..., idx] * 50.0
        x = x.to(torch.bfloat16)
    return x.to(device)


def timeit(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def torch_ref(x, beta=4.0, lb=25.0):
    """等价 torch 组合（6 个 elementwise）—— 收益倍数的**唯一正确分母**。"""
    gate, up = x.chunk(2, dim=-1)
    situ = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    return situ * (lb * torch.tanh(up / lb))


def my_op(x, beta=4.0, lb=25.0):
    return aclnn_ops.situ_and_mul(x, beta, lb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, nargs='+', default=[128, 512, 4096])
    ap.add_argument('--d', type=int, default=11008, help='K3 真实 intermediate_size')
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'verify_logs'))
    args = ap.parse_args()

    if not torch.npu.is_available():
        print('✗ NPU 不可用'); return 1
    print(f'device: {torch.npu.get_device_name(0)}   d={args.d}（K3 真实 intermediate_size）')
    print('分布：A=gauss（对照） B=mxfp4（4bit 反量化台阶） C=outlier（少数通道×50） D=mxfp4+outlier\n')

    dists = ['gauss', 'mxfp4', 'outlier', 'mxfp4out']
    rows_out = []
    for M in args.rows:
        shape = (M, 2 * args.d)
        print(f'--- shape={shape}  ({M} token) ---')
        for dist in dists:
            x = make_input(dist, shape, 'npu:0')
            ref = golden(x.cpu(), beta=4.0, linear_beta=25.0).to(torch.bfloat16).float()
            out = my_op(x).cpu().float()
            rel = ((out - ref).abs() / (ref.abs() + 1e-3)).max().item()
            t_my = timeit(lambda: my_op(x), args.iters)
            t_rf = timeit(lambda: torch_ref(x), args.iters)
            speedup = t_rf / t_my
            ok = rel < TOL
            print(f'  [{dist:8s}] max_rel={rel:.3e} {"OK " if ok else "FAIL"}   '
                  f'我的 {t_my:.4f} ms  torch 组合 {t_rf:.4f} ms  ⇒ 加速比 {speedup:.2f}x')
            rows_out.append({'rows': M, 'd': args.d, 'dist': dist, 'max_rel': rel, 'ok': ok,
                             'my_ms': t_my, 'torch_ms': t_rf, 'speedup': speedup})
        print()

    ok_all = all(r['ok'] for r in rows_out)
    sp = [r['speedup'] for r in rows_out]
    print(f'=== 小结 ===')
    print(f'  正确性：{sum(r["ok"] for r in rows_out)}/{len(rows_out)} 通过（判据 max_rel < {TOL:g}）'
          f'  ⇒ {"✅ 真实分布下不退化" if ok_all else "❌ 有分布超容差"}')
    print(f'  收益倍数（vs 等价 torch 组合）：{min(sp):.2f}x ~ {max(sp):.2f}x')
    print('  ★ 口径提醒：分母是「等价 torch 组合」，不是华为算子（华为无 SiTU 变体，见 docs/17 口径 A）')

    os.makedirs(args.outdir, exist_ok=True)
    p = os.path.join(args.outdir, 'real_weight_robustness.json')
    json.dump(rows_out, open(p, 'w'), indent=1)
    print(f'\n原始数字 → {p}')
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
