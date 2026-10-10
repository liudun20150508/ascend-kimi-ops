# KIMI3 算子开发工作区

只靠 shape + 随机数据 + 参考实现来验证算子正确性，不需要下载模型权重。

## 部署前提（包外依赖，需目标机器预装）

本仓库交付的是**算子源码 + 编译成品 + 参考实现 + 测试 + 文档**，不含 CANN/PyTorch 环境（这是昇腾 A3 机器的标准软件栈，需自行预装）：

| 依赖 | 版本 | 说明 |
|---|---|---|
| CANN | 9.0.0 | 含 AscendC 编译器 `ccec`/`bisheng`、算子脚手架 `msopgen` |
| torch / torch_npu | 2.10.0+cpu / 2.10.0 | |
| ops-transformer | — | 仅需 `torch_extension/cann_ops_transformer/common/inc/aclnn_common.h`（提供 `ACLNN_CMD` 宏） |

**运行前提**：目标设备为昇腾 A3（SOC `ascend910_93`），已按上述版本装好 CANN + torch_npu，并对齐 `aclnn_common.h` 路径。复现步骤见报告 §8.2 与 `docs/01`。

## 目录结构

```
kimi3_ops/
├── reference/                  # 参考实现（正确性金标准）
│   ├── situ.py                 #   SiTU-GLU
│   ├── kda.py                  #   KDA 递推
│   ├── kda_chunked.py          #   KDA 分块
│   ├── attn_res.py             #   AttnRes 残差栈
│   ├── modeling_kimi_linear.py #   MoonshotAI 官方建模源码
│   ├── modeling_kimi_k3.py
│   └── configuration_kimi_k3.py
├── kernels/                    # AscendC 算子工程（msopgen 生成 + 手写 kernel）
│   ├── SituAndMul/             #   ✅ SiTU-GLU（已开发 + 验证）
│   └── AttnRes/                #   ✅ AttnRes 残差栈（已开发 + 验证）
├── situ_and_mul_wrapper.cpp    # SiTU-GLU torch 包装（ACLNN_CMD）
├── attn_res_wrapper.cpp        # AttnRes torch 包装
├── test_situ_real.py           # SiTU-GLU 对拍 harness
├── test_attn_res_real.py       # AttnRes 对拍 harness
├── situ_op.json / attn_res_op.json  # msopgen 算子定义
└── tests/                      # 参考实现自测（纯 torch）
    ├── test_situ.py / test_kda.py / test_kda_chunked.py
```

## 验证方法论（无权重版）

1. 从 config 拿 shape/dtype
2. 固定 seed 生成随机输入
3. 参考实现跑出 golden 输出
4. 待开发算子跑候选输出
5. 对比：shape/dtype/layout 一致 + 数值误差在容差内（fp32→bf16 用 rtol=atol=1e-2）

## 缺口清单（KIMI3 vs 本机框架）与最终状态

| 算子 | 本机框架覆盖 | 最终状态 |
|---|---|---|
| MLA attention | ATB `MultiLatentAttention` 已覆盖 | ✅ 华为已做 |
| MoE dispatch/combine | `npu_ops_transformer` 已覆盖 | ✅ 华为已做 |
| sigmoid 路由 + topk | ATB `FusedAddTopkDiv` 已有 | ✅ 华为已做 |
| KDA | `aclnnChunkKdaFwd`（缺 Python wrapper） | ✅ 华为已做，已补 wrapper + 验证（vs 参考 ~4e-3） |
| **SiTU-GLU** | ❌ 只有 SiLU | ✅ **自研 AscendC kernel + 验证**（d=11008 时 max_rel≈7.5e-3） |
| **AttnRes 残差栈** | ❌ 无 | ✅ **自研 AscendC kernel + 验证**（H=4096 时 max_rel≈7.7e-3） |
| **LatentMoE 投影** | ❌ 形状不符 | 🔧 **无需新 kernel**：图级组合（down/up Linear + MoE + RMSNorm，expert 内用 SiTU-GLU） |

**结论**：华为真正缺的算子只有 2 个（SiTU-GLU、AttnRes），均已自研并验证通过；LatentMoE 是模型图接线问题（在 vllm/sglang 适配层插 down/up 投影），不属算子开发。

## 已确认的 KDA 数学（关键，别写错）

KDA = Gated Delta Rule：
```
S_t = S_{t-1} * exp(g_t) + β_t · k_t · (v_t - k_tᵀ·S_{t-1})ᵀ
o_t = q_tᵀ · S_t
```
三个 in-kernel 预处理（modeling 源码都启用了）：
1. q/k L2 norm（head_dim 维，eps=1e-6）
2. gate: `g = -5.0 * sigmoid(exp(A_log) * (g_raw + dt_bias))`（safe_gate, lower_bound=-5.0）
3. beta: `beta = sigmoid(beta_raw)`

**注意**：q/k L2 norm 是数值稳定性的硬要求 —— harness 实测缺了它误差会爆炸到 2e8。

## 参考来源

- modeling 源码: moonshotai/Kimi-K3 的 modeling_kimi_linear.py
- KDA 算法: fla-org/flash-linear-attention 的 chunk_kda + naive.py
- 性能/拆 kernel 参考: MoonshotAI/FlashKDA（CUDA/CUTLASS，重写为 AscendC 时参考）
