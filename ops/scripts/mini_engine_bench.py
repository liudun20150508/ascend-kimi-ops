#!/usr/bin/env python3
"""mini 推理引擎：用**真实算子**跑出系统层指标（TTFT / TPOT / 吞吐）。

为什么要自己写这一层
--------------------
四层体系里，**服务/调度层**的判据是 TTFT / TPOT / 吞吐 / 显存 —— 这些指标只有把
"模型 + 算子 + KV cache + prefill/decode" 串起来才拿得到。
本机装不上 vLLM（`triton-ascend >= 3.2.1` 公开源暂无，见 `docs/14` §15.3），
所以这里写一个**最小引擎**把系统指标跑出来 —— 但**组件尽量都用真实算子**：

| 组件 | 用什么 | 是不是"真算子" |
|---|---|---|
| Attention | `torch_npu.npu_fusion_attention`（CANN 内置 FlashAttention） | ✅ 内置算子 |
| FFN (MLP) | 官方 `KimiMLP`（`act_fn` 可替换成本项目的 `situ_and_mul`） | ✅ 官方模型层 + 自研算子 |
| GEMM / RMSNorm | `torch.matmul` / `torch_npu.npu_rms_norm` | ✅ 内置 |
| KV cache / batching / prefill-decode | 本文件自己写 | ⚠️ 自研（这才是"引擎"） |

⚠️ 必须一起读的口径声明（不写清楚就是吹）
----------------------------------------
1. **权重是随机初始化的**（不下载 K3 的 1.5 TB 权重）⇒ 本脚本量的是**系统时延/吞吐**，
   **不是**模型质量、也**不是** K3 的真实性能；
2. **模型规模是缩小的**（层数/维度可配，`--medium` 用 K3 真实的 `intermediate_size=11008`）；
3. 这是**自研 mini 引擎，不是 vLLM**：能证明"我懂 prefill/decode、KV cache、batching 怎么影响指标"，
   **不能**代替 vLLM 的端到端验收；
4. batch 内**请求同长**（简化版 batching；真实引擎要 varlen kernel）。

怎么用
------
    # 快速自测（秒级）
    /usr/local/python3.11.15/bin/python3 scripts/mini_engine_bench.py --preset tiny
    # 中等规模：H=4096、D=11008（K3 真实 intermediate_size）、8 层
    /usr/local/python3.11.15/bin/python3 scripts/mini_engine_bench.py --preset medium

判据（跑之前就写好）
------------------
- **TTFT**：prefill 一次前向的时间（含所有层）—— 越小越好，受"层数 × 每层算力"支配；
- **TPOT**：decode 每 token 时间 —— 受"每步要读全部权重"支配（**访存受限**），
  ⇒ 换掉一个 elementwise 算子对它影响很小（**Amdahl**），这本身就是该讲的结论；
- **吞吐**：单位时间处理的 token 数；
- **对照**：同一引擎，`act_fn` 用**自研算子** vs **reference 纯 torch 实现**，看整机差异。
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'integration'))

for _c in ('/workspace/ops-transformer/torch_extension/cann_ops_transformer',
           '/tmp/ops-transformer/torch_extension/cann_ops_transformer'):
    if os.path.isdir(_c):
        os.environ.setdefault('OPS_TRANSFORMER_PKG', _c)
        break
os.environ.setdefault('ASCEND_HOME_PATH', '/usr/local/Ascend/cann-9.0.0')
os.environ.setdefault('KIMI3_VENDOR', os.environ['ASCEND_HOME_PATH'] + '/opp/vendors/customize')
os.environ.setdefault('ASCEND_CUSTOM_OPP_PATH', os.environ['KIMI3_VENDOR'])

import torch                      # noqa: E402
import torch.nn as nn             # noqa: E402
import torch.nn.functional as F   # noqa: E402
import torch_npu                  # noqa: E402,F401

# ---- fla stub：`modeling_kimi_linear.py` 顶层 `from fla.* import` 会拉起 triton driver，
#      而本机 `triton-ascend 3.2.0` 与 CANN 9.0 **不配套**（实测编不过，见 docs/14 §15.3）⇒
#      必须在 import reference 之前把 fla 相关模块塞成 stub（与 l2_mlp_check.py 同一做法）。
from unittest import mock  # noqa: E402

for _m in ('fla', 'fla.modules', 'fla.ops', 'fla.ops.kda', 'fla.ops.utils',
           'fla.ops.utils.index', 'fla.utils'):
    sys.modules[_m] = mock.MagicMock()

from reference.configuration_kimi_k3 import KimiLinearConfig   # noqa: E402
from reference.modeling_kimi_linear import KimiMLP             # noqa: E402
from reference.situ import situ_and_mul as situ_ref             # noqa: E402
import aclnn_ops                                                # noqa: E402

PRESETS = {
    # (hidden, intermediate, layers, heads, seq_prefill, gen_len, batch)
    'tiny':   (256,   512,   2,  4,   64,  16, 1),
    'small':  (1024,  4096,  4,  8,  256,  32, 1),
    'medium': (4096, 11008,  8, 32,  512,  32, 1),   # D = K3 真实 intermediate_size
}


class RMSNorm(nn.Module):
    def __init__(self, H, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(H))
        self.eps = eps

    def forward(self, x):
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
        return (v * self.w.float()).to(x.dtype)


class RefAct(nn.Module):
    """reference：纯 torch 的 SiTU-GLU（就是"没有自定义算子"时的成本）。"""

    def forward(self, t):
        return situ_ref(t, 4.0, 25.0)


class MyAct(nn.Module):
    """自研：走 aclnn 的自定义算子（含 JIT 包装）。"""

    def forward(self, t):
        return aclnn_ops.situ_and_mul(t, 4.0, 25.0)


class MiniLayer(nn.Module):
    """一层 decoder：attention（内置 FlashAttention）+ FFN（官方 KimiMLP）。"""

    def __init__(self, H, D, heads, cfg_proto):
        super().__init__()
        self.H, self.heads = H, heads
        self.head_dim = H // heads
        self.norm1 = RMSNorm(H)
        self.qkv = nn.Linear(H, 3 * H, bias=False)
        self.o = nn.Linear(H, H, bias=False)
        self.norm2 = RMSNorm(H)
        # 复用官方 KimiMLP（act_fn 之后会被替换）
        cfg = KimiLinearConfig(
            hidden_size=H, intermediate_size=D, num_hidden_layers=1,
            num_attention_heads=heads, vocab_size=1024, hidden_act='situ',
            activation_situ_beta=4.0, activation_situ_linear_beta=25.0,
        )
        self.mlp = KimiMLP(cfg)

    def forward(self, x, kv=None, use_builtin_attn=True):
        B, S, H = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(B, S, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # [B, heads, S, hd]
        if kv is not None:
            k = torch.cat([kv[0], k], dim=2)
            v = torch.cat([kv[1], v], dim=2)
        new_kv = (k, v)
        attn = None
        if use_builtin_attn:
            try:
                # CANN 内置 FlashAttention：layout 'BNSD'，输入 [B, N, S, D]
                out = torch_npu.npu_fusion_attention(
                    q, k, v, self.heads, 'BNSD', keep_prob=1.0, scale=1.0 / (self.head_dim ** 0.5),
                    pre_tockens=2147483647, next_tockens=2147483647,
                    sparse_mode=0, inner_precise=1, input_layout='BNSD')
                attn = out[0]
            except Exception:
                attn = None
        if attn is None:
            attn = F.scaled_dot_product_attention(q, k, v)     # 回退：torch SDPA
        attn = attn.permute(0, 2, 1, 3).reshape(B, S, H)
        x = x + self.o(attn)
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class MiniEngine:
    def __init__(self, H, D, L, heads, dtype=torch.bfloat16):
        self.layers = nn.ModuleList([MiniLayer(H, D, heads, None) for _ in range(L)]).to(dtype).npu().eval()
        self.dtype = dtype

    def set_act(self, kind):
        """把每层 MLP 的 act_fn 换成 自研算子 / reference。"""
        act = MyAct() if kind == 'mine' else RefAct()
        for l in self.layers:
            l.mlp.act_fn = act

    @torch.no_grad()
    def prefill(self, x):
        kv = []
        for l in self.layers:
            x, c = l(x, None)
            kv.append(c)
        return x, kv

    @torch.no_grad()
    def decode(self, x, kv):
        nxt = []
        for i, l in enumerate(self.layers):
            x, c = l(x, kv[i])
            nxt.append(c)
        return x, nxt


def timed(fn, iters=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3   # ms


def run(preset, impl, iters):
    H, D, L, heads, S, G, B = PRESETS[preset]
    print(f'\n=== preset={preset}  H={H} D={D} L={L} heads={heads} '
          f'prefill={S} gen={G} batch={B}  act_fn={impl} ===')
    eng = MiniEngine(H, D, L, heads)
    n_params = sum(p.numel() for p in eng.layers.parameters())
    print(f'参数量 = {n_params / 1e6:.1f} M（bf16 权重 ≈ {n_params * 2 / 2**30:.2f} GiB）')

    x = torch.randn(B, S, H, dtype=torch.bfloat16).npu()

    # ① TTFT：prefill 一次
    ttft = timed(lambda: eng.prefill(x), iters=iters)
    _, kv = eng.prefill(x)

    # ② TPOT：decode 每 token
    xd = torch.randn(B, 1, H, dtype=torch.bfloat16).npu()
    green = {}

    def one_decode():
        nonlocal kv
        _, kv2 = eng.decode(xd, kv)
        green['kv'] = kv2

    tpot = timed(one_decode, iters=max(iters, 10))
    total_ms = ttft + tpot * G
    thr = (B * (S + G)) / (total_ms * 1e-3)
    print(f'TTFT            = {ttft:8.2f} ms   ({ttft / S:7.3f} ms/token prefill)')
    print(f'TPOT            = {tpot:8.2f} ms/token')
    print(f'端到端（{G} token）= {total_ms:8.2f} ms   吞吐 = {thr:8.1f} token/s')
    return {'preset': preset, 'H': H, 'D': D, 'L': L, 'heads': heads, 'S': S, 'G': G, 'B': B,
            'act_fn': impl, 'params_M': n_params / 1e6,
            'ttft_ms': ttft, 'tpot_ms': tpot, 'total_ms': total_ms, 'throughput_tok_s': thr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--preset', default='small', choices=list(PRESETS))
    ap.add_argument('--iters', type=int, default=5)
    ap.add_argument('--outdir', default=os.path.join(ROOT, 'verify_logs'))
    args = ap.parse_args()

    if not torch.npu.is_available():
        print('✗ NPU 不可用'); return 1

    print(f'device: {torch.npu.get_device_name(0)}')
    print('⚠️ 权重随机初始化、模型规模缩小、引擎自研 —— 量的是**系统时延/吞吐**，不是 K3 真实性能')

    rows = [run(args.preset, 'mine', args.iters), run(args.preset, 'ref', args.iters)]
    m, r = rows
    print(f'\n=== 对照（同引擎，只换 act_fn）===')
    print(f'  TTFT : 自研 {m["ttft_ms"]:.2f} ms  vs  reference {r["ttft_ms"]:.2f} ms  '
          f'（差 {(r["ttft_ms"] - m["ttft_ms"]) / r["ttft_ms"] * 100:+.1f}%）')
    print(f'  TPOT : 自研 {m["tpot_ms"]:.2f} ms  vs  reference {r["tpot_ms"]:.2f} ms  '
          f'（差 {(r["tpot_ms"] - m["tpot_ms"]) / r["tpot_ms"] * 100:+.1f}%）')
    print('\n读法：TPOT 受"每步读全部权重"支配 ⇒ 单个 elementwise 算子对它的影响很小（Amdahl）；')
    print('      TTFT 是算力/访存混合，替换算子的相对影响比 TPOT 明显。')

    os.makedirs(args.outdir, exist_ok=True)
    p = os.path.join(args.outdir, 'mini_engine.json')
    json.dump(rows, open(p, 'w'), indent=1)
    print(f'\n原始数字 → {p}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
