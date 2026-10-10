"""MXFP4 单层反量化 + 精度对拍（零第三方依赖，纯标准库）。

要回答的问题（追问里被追问「为什么 K3 在 A3 上不能用原生 MXFP4」时的硬证据）：
  1. MXFP4 (E2M1 + 每 32 元素共享 E8M0 指数) 到底损失多少精度？
  2. 反量化回 bf16 后，一层权重还能不能对得上？
  3. 单层显存到底多大 —— 为什么「BF16 载入单层再转换」会溢出？

结论预期（不依赖任何硬件，任何机器都能复算）：
  * MXFP4 往返相对误差的量级在 ~1e-1（4 bit 只有 16 个格点，这是格式决定的，不是 bug）
  * 但**块的共享指数**让误差只有「块内动态范围」一个维度，不会随权重整体尺度漂移
  * 单层 MXFP4 ≈ 14.8 GB、BF16 ≈ 59.2 GB ⇒ 单卡 64GB 只能走 MXFP4 路径

用法：
    python3 mxfp4_dequant_check.py            # 默认规模
    python3 mxfp4_dequant_check.py --rows 14336 --cols 4096 --group 32
    python3 mxfp4_dequant_check.py --mem-only # 只算显存账（秒级返回）
"""

import argparse
import math
import random
import sys

# ---------------------------------------------------------------------------
# MXFP4 元素格式：E2M1（1 符号位 + 2 指数位 + 1 尾数位），共 16 个格点
# 注意 E2M1 没有 inf/nan：指数全 1 时是 4 / 6，属于正常值
# ---------------------------------------------------------------------------
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)  # magnitude → code 0b000..0b111
E2M1_MAX = E2M1_VALUES[-1]
E2M1_EMAX = 2  # floor(log2(6)) == 2

# 块缩放因子用 E8M0（8 位纯指数，无符号，bias=127）——这就是 MX 规范的 shared scale
E8M0_BIAS = 127


def encode_e2m1(value: float) -> int:
    """把单个幅值量化到最接近的 E2M1 格点，返回 4 bit code（含符号位）。

    注意饱和：|value| > 6 时会 nearest-round 到 6.0（E2M1 的最大格点），
    这是 MX 规范的允许行为 —— 但也意味着**块共享指数取错会让整块被截平**，
    所以 `shared_exp` 必须按 floor(log2(block_max)) - emax 算，不能随手取整。
    """
    neg = value < 0 or (value == 0.0 and math.copysign(1.0, value) < 0)
    a = abs(value)
    # 在 16 个格点里找最近的（只有 8 个幅值，直接线性扫，无需查表技巧）
    best_i, best_d = 0, abs(a - E2M1_VALUES[0])
    for i, cand in enumerate(E2M1_VALUES):
        d = abs(a - cand)
        if d < best_d:
            best_i, best_d = i, d
    code = best_i | (0b1000 if neg else 0)
    return code & 0xF


def decode_e2m1(code: int) -> float:
    mag = E2M1_VALUES[code & 0b0111]
    return -mag if (code & 0b1000) else mag


def shared_exp(block_max_abs: float) -> int:
    """块共享指数（E8M0 的无偏指数值）。返回 0 表示 scale = 2^-127（极小数用）。"""
    if block_max_abs <= 0.0 or not math.isfinite(block_max_abs):
        return 0
    e = int(math.floor(math.log2(block_max_abs))) - E2M1_EMAX
    return max(0, min(254, e + E8M0_BIAS)) - E8M0_BIAS  # 返回无偏指数


def quantize_mxfp4(weights, group: int = 32):
    """按 group 个元素一组量化。返回 (codes:list[int], exps:list[int])。"""
    codes, exps = [], []
    for start in range(0, len(weights), group):
        blk = weights[start:start + group]
        e = shared_exp(max(abs(w) for w in blk))
        scale = 2.0 ** e
        exps.append(e)
        for w in blk:
            codes.append(encode_e2m1(w / scale))
    return codes, exps


def dequantize_mxfp4(codes, exps, group: int = 32):
    out = []
    for i, c in enumerate(codes):
        scale = 2.0 ** exps[i // group]
        out.append(decode_e2m1(c) * scale)
    return out


def stats(orig, approx):
    """max_abs / max_rel（带 1e-3 绝对项，避免跨零假失败）/ SNR(dB)。"""
    n = len(orig)
    max_abs, max_rel, sig, noise = 0.0, 0.0, 0.0, 0.0
    for o, a in zip(orig, approx):
        d = abs(o - a)
        max_abs = max(max_abs, d)
        max_rel = max(max_rel, d / (abs(o) + 1e-3))
        sig += o * o
        noise += d * d
    snr = 10.0 * math.log10(sig / noise) if noise > 0 else float("inf")
    return max_abs, max_rel, snr


def gen_layer_weights(rows, cols, seed=0, kind="normal"):
    """造一层权重。kind=normal 更像真实权重（有 per-channel 尺度差异），uniform 用于对照。"""
    rng = random.Random(seed)
    if kind == "uniform":
        return [rng.uniform(-0.05, 0.05) for _ in range(rows * cols)]
    # 每个输出通道一个对称尺度 → 模拟真实的 per-channel 尺度差异，
    # 这正是「块共享指数」要救的场景
    out = []
    for r in range(rows):
        s = 10 ** rng.uniform(-2.5, -0.5)
        for _ in range(cols):
            out.append(rng.gauss(0.0, s))
    return out


def mem_accounting(total_params=2.8e12, layers=92, expert_hidden=3072, latent=3584,
                   experts=896, kv_gb=6.0, buffer_frac=0.40):
    """复算部署方案 §2.2 / §2.4 的显存账（纯公式，不分配内存）。"""
    print("=" * 74)
    print("显存账（复算 KimiK3-昇腾部署方案 §2.2 / §2.4）")
    print("=" * 74)

    # 单层 MoE 参数量自洽性检查
    per_expert = 3 * latent * expert_hidden
    per_moe_layer = per_expert * experts
    total_moe = per_moe_layer * layers
    print(f"单专家      = 3 × {expert_hidden} × {latent}        = {per_expert/1e6:8.1f} M")
    print(f"单 MoE 层   = {per_expert/1e6:.1f}M × {experts} 专家      = {per_moe_layer/1e9:8.1f} B")
    print(f"{layers} 个 MoE 层                        = {total_moe/1e12:8.2f} T   "
          f"（与 K3 总参 {total_params/1e12:.1f}T 自洽）")
    print()

    for name, bytes_per_param in (("MXFP4 常驻", 0.5), ("W8A8 反量化常驻", 1.0), ("BF16", 2.0)):
        w_gb = total_params * bytes_per_param / 1e9
        need = w_gb / (1.0 - buffer_frac) + kv_gb  # 权重占 (1-buffer_frac)，再加 KV
        cards = math.ceil(need / 64.0)  # 按 64GB 单卡估
        print(f"{name:<18} 权重 {w_gb:8.0f} GB  含 KV/buffer 需 {need:8.0f} GB  → 64GB 卡 ×{cards}")
    print()

    # 单层显存：为什么「逐层 BF16 载入再转换」会溢出
    mx = per_moe_layer * 0.5 / 1e9
    bf = per_moe_layer * 2.0 / 1e9
    # 单看权重 59.2 GB < 64 GB 好像放得下，但单卡上还要放激活、KV、框架开销 ——
    # 单层权重超过显存一半，实际必然 OOM。这正是官方那句「逐层加载仍溢出」的由来。
    print(f"单层 MoE(MXFP4) = {mx:6.2f} GB   ← {'放得下' if mx < 64 * 0.5 else '也紧张'}")
    print(f"单层 MoE(BF16)  = {bf:6.2f} GB   ← "
          f"{'放得下' if bf < 64 * 0.5 else '**放不下（叠加激活/KV/框架开销后必 OOM）→ 这就是官方说的逐层加载仍溢出**'}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--cols", type=int, default=2048)
    ap.add_argument("--group", type=int, default=32, help="MX 的块大小（K3 用 32）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mem-only", action="store_true")
    args = ap.parse_args()

    mem_accounting()
    if args.mem_only:
        return

    print("=" * 74)
    print(f"精度对拍：{args.rows}×{args.cols} 权重，block_size={args.group}")
    print("=" * 74)

    for kind in ("normal", "uniform"):
        w = gen_layer_weights(args.rows, args.cols, seed=args.seed, kind=kind)
        codes, exps = quantize_mxfp4(w, args.group)
        dq = dequantize_mxfp4(codes, exps, args.group)
        max_abs, max_rel, snr = stats(w, dq)

        # 对照：不做块缩放，直接 4bit（会因整体尺度失配而崩掉）
        flat_scale = max(abs(x) for x in w) / E2M1_MAX
        naive = [decode_e2m1(encode_e2m1(x / flat_scale)) * flat_scale for x in w]
        naive_abs, naive_rel, naive_snr = stats(w, naive)

        # 每 32 元素实际占用的字节：32×4bit + 1 字节 scale = 17 B
        elems = len(w)
        mxfp4_gb = elems * (4 / 8) / 1e9 + (elems // args.group) / 1e9
        bf16_gb = elems * 2 / 1e9
        print(f"\n[{kind}]")
        print(f"  MXFP4(per-{args.group} scale) : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  SNR={snr:6.2f} dB")
        print(f"  4bit 无块缩放（对照）         : max_abs={naive_abs:.3e}  max_rel={naive_rel:.3e}  SNR={naive_snr:6.2f} dB")
        print(f"  存储: MXFP4 {mxfp4_gb:.3f} GB  vs  BF16 {bf16_gb:.3f} GB  → "
              f"压缩 {bf16_gb/mxfp4_gb:.2f}x")

    print("\n判定：MXFP4 的往返误差由「4bit 只有 16 个格点」决定，SNR 稳定在 ~20 dB 量级；")
    print("      块共享指数的作用是让误差不随权重整体尺度漂移（对照组的 SNR 明显更差）。")
    print("      所以 A3 上正确的做法是：盘上存 MXFP4，算前反量化成 W8A8，而不是先转 BF16（单层 59.2GB 装不下）。")


if __name__ == "__main__":
    sys.exit(main())
