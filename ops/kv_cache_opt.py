"""KV cache 优化计算器（零依赖）—— 把每种优化手段的收益算成一个数字。

    python3 kv_cache_opt.py
    python3 kv_cache_opt.py --layers 92 --s 32768 --kv-budget-gb 10
    python3 kv_cache_opt.py --sweep          # 扫序列长，看 KV 什么时候爆显存

要回答的四件事：
  1. MHA / GQA / MQA / MLA 的 KV 到底差多少倍？（同一份配置下）
  2. 量化（bf16→fp8→int4）能省多少？
  3. 「省 KV」为什么等于「提吞吐」？（KV 决定能塞多大 batch，batch 决定吞吐）
  4. 长上下文下哪个手段是唯一出路？

⚠️ 参数出处（别混）：
   · layers=92、MoE/专家数：来自 `kimi3_deploy/KimiK3-昇腾部署方案.md`（K3 真实规格）
   · hidden=4096 / 32 heads / head_dim=128：来自 `reference/configuration_kimi_k3.py` 的默认值
   · **kv_lora_rank=512、qk_rope_head_dim=64 是 DeepSeek-V2 风格的示意值** ——
     K3 config 里这两个字段是 Optional(None)，真实值请从模型目录的 config.json 取，
     然后用 --kv-lora / --rope-dim 覆盖。下面的绝对值会随它变，但**倍数关系不变**。
"""

import argparse

# 每种数值格式的实际存储字节数（MX 格式每 32 元素多 1 字节 scale）
DTYPE_BYTES = {
    "bf16": 2.0, "fp16": 2.0,
    "fp8": 1.0, "int8": 1.0,
    "int4": 0.5, "mxfp4": 0.5 + 1.0 / 32,
}


def kv_bytes_per_token(kind, layers, heads, head_dim, kv_heads, kv_lora, rope_dim, dtype):
    """返回「每个 token 的 KV cache 字节数」（batch=1、不算序列维）。"""
    b = DTYPE_BYTES[dtype]
    if kind == "mha":                      # 每个 Q 头各一套 K、V
        per_layer = 2 * heads * head_dim
    elif kind == "gqa":                    # 分组共享：kv_heads 组
        per_layer = 2 * kv_heads * head_dim
    elif kind == "mqa":                    # 全部 Q 头共用 1 组
        per_layer = 2 * head_dim
    elif kind == "mla":                    # 低秩压缩：只缓存 latent + 一份 decoupled rope key
        per_layer = kv_lora + rope_dim
    else:
        raise ValueError(kind)
    return layers * per_layer * b


def human(nbytes):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if nbytes < 1024 or u == "TB":
            return f"{nbytes:.1f} {u}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def exp_structures(args):
    print("=" * 84)
    print(f"实验一：四种注意力结构的 KV cache（layers={args.layers}  heads={args.heads} "
          f"head_dim={args.head_dim}  dtype={args.dtype}）")
    print("=" * 84)
    rows = [
        ("MHA  (每头独立)", "mha", args.heads),
        (f"GQA  (kv_heads={args.gqa_kv_heads})", "gqa", args.gqa_kv_heads),
        ("MQA  (全共享)", "mqa", 1),
        (f"MLA  (lora={args.kv_lora}+rope={args.rope_dim})", "mla", 1),
    ]
    per_tok = {}
    print(f"  {'结构':<30}{'每 token':>12}{'×S=32K':>14}{'相对 MHA':>12}")
    base = None
    for name, kind, kvh in rows:
        v = kv_bytes_per_token(kind, args.layers, args.heads, args.head_dim,
                               kvh, args.kv_lora, args.rope_dim, args.dtype)
        per_tok[kind] = v
        if base is None:
            base = v
        print(f"  {name:<30}{human(v):>12}{human(v * 32768):>14}{v / base:>11.2f}x")
    print()
    print(f"  → MLA 相对 MHA 省 {base / per_tok['mla']:.1f}x，相对 GQA{args.gqa_kv_heads} "
          f"省 {per_tok['gqa'] / per_tok['mla']:.1f}x")
    print("    注意 MLA 只缓存**压缩后的 latent**（K/V 用时再投影出来），所以是 1 份而不是 2 份 ——")
    print("    这就是它能同时做到「MHA 的表达力 + 比 MQA 还小的 cache」的原因。")
    return per_tok


def exp_dtype(args, per_tok):
    print()
    print("=" * 84)
    print("实验二：量化位宽的影响（以 MLA 结构为例，S=32768）")
    print("=" * 84)
    print(f"  {'格式':<10}{'每 token':>12}{'×S=32K':>14}{'相对 bf16':>12}")
    ref = None
    for dt in ("bf16", "fp8", "int8", "int4", "mxfp4"):
        v = kv_bytes_per_token("mla", args.layers, args.heads, args.head_dim,
                               1, args.kv_lora, args.rope_dim, dt)
        if ref is None:
            ref = v
        print(f"  {dt:<10}{human(v):>12}{human(v * 32768):>14}{v / ref:>11.2f}x")
    print()
    print("  → 量化是最直接的手段：位宽砍半，KV 就砍半，且不需要改模型结构。")
    print("    但代价是精度：KV 是**激活**不是权重，分布随序列漂移，长上下文累积误差明显。")
    print("    工程做法：K 按 channel 量化、V 按 token 量化（KIVI 的观察），关键层留高精度。")


def exp_batch(args, per_tok):
    print()
    print("=" * 84)
    print(f"实验三：KV 预算 → 能塞多少并发（KV 预算 {args.kv_budget_gb} GB，"
          f"S={args.s}，MLA 结构）")
    print("=" * 84)
    budget = args.kv_budget_gb * 1024 ** 3
    per_seq = per_tok["mla"] * args.s
    max_seqs = budget / per_seq
    print(f"  单请求 KV 占用       = {human(per_seq)}")
    print(f"  KV 预算内可容纳请求数 = {max_seqs:.2f}")
    print()
    print("  → 这就是「省 KV = 提吞吐」的因果链：")
    print("    KV 小 → 同显存能塞下更多并发 → 一次权重读取服务更多 token → decode 吞吐上升。")
    print("    decode 是访存受限的，权重读取量是固定的，所以**并发数几乎线性决定吞吐**。")
    print(f"    反过来说：如果 max_seqs < 1，这个 (S={args.s}) 的配置**根本跑不起来**。")
    print()
    print("  补救顺序（问答可以背）：① 换结构(MLA/GQA) ② 量化 ③ 加卡切 TP ④ 降 S/滑窗")


def exp_sweep(args, per_tok):
    print()
    print("=" * 84)
    print(f"实验四：长序列扫描（MLA，batch=1，看 KV 何时超过单卡可用显存 {args.card_gb} GB）")
    print("=" * 84)
    print(f"  {'序列长':>12}{'MLA':>14}{'GQA' + str(args.gqa_kv_heads):>16}{'MHA':>14}{'MLA是否放得下':>16}")
    avail = args.card_gb * 1024 ** 3 * 0.5      # 只给 KV 一半显存（另一半留给权重/激活）
    for s in (4096, 32768, 131072, 1048576):
        mla = per_tok["mla"] * s
        gqa = per_tok["gqa"] * s
        mha = per_tok["mha"] * s
        ok = "✅" if mla <= avail else "❌ 装不下"
        print(f"  {s:>12}{human(mla):>14}{human(gqa):>16}{human(mha):>14}{ok:>16}")
    print()
    print("  → 1M 上下文是**结构问题不是调优问题**：")
    print("    MHA/GQA 在 128K 以上就已经把单卡显存吃穿了，只有 MLA（+量化）才可能。")
    print("    这也是为什么「长上下文」的模型清一色在用 MLA / 线性注意力 / 稀疏注意力。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=92, help="K3 真实 MoE 层数")
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--gqa-kv-heads", type=int, default=8)
    ap.add_argument("--kv-lora", type=int, default=512, help="MLA 的 KV 压缩秩（示意值）")
    ap.add_argument("--rope-dim", type=int, default=64, help="MLA 的 decoupled rope 维度（示意值）")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--s", type=int, default=32768)
    ap.add_argument("--kv-budget-gb", type=float, default=10.0)
    ap.add_argument("--card-gb", type=float, default=64.0)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    per_tok = exp_structures(args)
    exp_dtype(args, per_tok)
    exp_batch(args, per_tok)
    if args.sweep:
        exp_sweep(args, per_tok)
    else:
        print("\n（加 --sweep 看长序列下 KV 什么时候爆显存）")

    print()
    print("=" * 84)
    print("KV cache 优化的五个层次（问答按这个顺序讲）")
    print("=" * 84)
    print("  ① 结构层：GQA/MQA 减头数、MLA 低秩压缩、跨层共享、滑窗/稀疏 —— 省得最多，但要改模型")
    print("  ② 数值层：fp8/int8/int4 量化 —— 最直接；坑是 K/V 要用不同粒度（K 按 channel、V 按 token）")
    print("  ③ 显存层：PagedAttention 消碎片、Prefix Caching 复用前缀、KV 换出到 CPU/SSD")
    print("  ④ 计算层：FlashDecoding 并行化 decode、KV 复用（beam search / 多轮对话）")
    print("  ⑤ 系统层：PD 分离时把 KV 传输与计算重叠（layerwise push）、前缀亲和路由")
    print()
    print("  判据（别只说方向）：KV 占用 GB / 每 token KV 读取字节 / 可容纳并发数 / TPOT / 长文精度")


if __name__ == "__main__":
    main()
