"""SiTU-GLU 验证 harness：shape → 随机数据 → 参考实现 的端到端示范。

验证两个层面：
1. 参考实现本身与 modeling 源码 `SituAndMul` 逐位一致（用随机数据跑两边对比）。
2. 输出 shape/dtype 符合预期，数值在合理范围（tanh/sigmoid 有界性 sanity check）。

后续开发 AscendC 算子时，把 `candidate` 换成你的 kernel 输出，
对比 `reference` 即可，容差按 fp32→bf16 设定 rtol=1e-2, atol=1e-2。
"""

import sys
sys.path.insert(0, "/workspace/kimi3_ops")

import torch

from reference.situ import situ_and_mul


def _reference_from_source(x: torch.Tensor, beta: float, linear_beta: float | None):
    """与 modeling_kimi_linear.py 中 SituAndMul.forward 等价的就地实现。"""
    d = x.shape[-1] // 2
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)


def main():
    torch.manual_seed(0)

    # KIMI3 真实 shape：MLP 的 gate_up 拼接后是 2 * intermediate_size
    hidden = 7168
    intermediate = 33792
    batch, seq = 1, 8

    # bf16 输入（KIMI3 dtype = bfloat16）
    x = torch.randn(batch, seq, 2 * intermediate, dtype=torch.bfloat16)

    beta, linear_beta = 4.0, 25.0
    ref = situ_and_mul(x, beta=beta, linear_beta=linear_beta)
    ref_src = _reference_from_source(x, beta=beta, linear_beta=linear_beta)

    # 1. 参考实现 vs modeling 源码逐位一致
    assert ref.shape == (batch, seq, intermediate), f"shape 不对: {ref.shape}"
    assert ref.dtype == torch.bfloat16, f"dtype 不对: {ref.dtype}"
    torch.testing.assert_close(ref, ref_src, rtol=0, atol=0,
                                msg="参考实现与 modeling 源码不一致")

    # 2. 数值 sanity check：SiTU 有界，gate/up 经 tanh 压缩后 |out| 有上界
    out = ref.float()
    bound = beta * 1.0 * linear_beta * 1.0  # tanh,sigmoid ∈ (0,1) 的最大量级
    print(f"shape        = {tuple(ref.shape)}")
    print(f"dtype        = {ref.dtype}")
    print(f"|out| max    = {out.abs().max().item():.6f}")
    print(f"理论量级上界  = {bound:.1f}")
    assert out.abs().max().item() <= bound + 1e-3

    # 3. 反例验证：SiLU（常见误实现）应产生显著差异，证明 harness 能抓错
    def silu_wrong(x, beta, linear_beta):
        d = x.shape[-1] // 2
        g = x[..., :d].float()
        u = x[..., d:].float()
        return (g * torch.sigmoid(g) * u).to(x.dtype)

    wrong = silu_wrong(x, beta, linear_beta)
    diff = (wrong.float() - ref.float()).abs().max().item()
    print(f"SiLU 误实现 vs 正确参考 最大误差 = {diff:.6f}")
    assert diff > 1e-3, "harness 未能区分错误实现，检查容差设定"

    print("\n✅ SiTU 验证链路跑通：shape→随机数据→参考对比 方法论成立")


if __name__ == "__main__":
    main()
