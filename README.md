# 昇腾 Ascend C 算子：Kimi K3 部署适配

本项目为 **Kimi K3 模型**（MoonshotAI，2.8T 参数 MoE）在**华为昇腾**国产芯片上的本地化部署而开发：补齐昇腾官方算子库缺失的算子，并给出完整部署方案。目标平台：昇腾 A3（SOC `ascend910_93`，CANN 9.0.0）。

> 交付的是**芯片级算子**（AscendC kernel → CANN aclnn 接口 → NPU 二进制），不是 PyTorch 自定义算子——能被 PyTorch / vLLM / SGLang 调用，也能被裸 C++（`aclrt`+`aclnn`）直接调用。

## 背景与目的

Kimi K3（MoonshotAI）是一个 2.8T 参数的超大规模 MoE 模型。把它本地化部署到国产昇腾芯片时，遇到两类问题：

**① 华为昇腾算子库缺 K3 需要的两个算子：**

| 算子 | 作用 | 官方现状 |
|---|---|---|
| SiTU-GLU（`SituAndMul`） | K3 的 MLP 激活函数 | 只有 SiLU，无 SiTU 变体 |
| AttnRes（残差栈） | K3 线性注意力的残差状态栈 | 无对应算子 |

**② vLLM 接入有一个算子图级问题：** LatentMoE 的 down/up 投影层在官方算子库中形状不符、无法直接接线。这不是「缺算子」，而是要在 vLLM/SGLang 适配层做**图级组合**（down/up Linear + MoE + RMSNorm），不需要新写 kernel。

本仓库的目的就是解决这两类问题：

- 用 Ascend C **自研补齐 SiTU-GLU、AttnRes 两个缺失算子**，并真机对拍验证；
- 落地 **LatentMoE 图级接线**（`integration/latent_moe.py`，与官方 `KimiSparseMoeBlock` 同权重超容差 **0 / 384**）；
- 提供 K3 在昇腾上的**完整部署方案**（显存/算力测算、并行拓扑、PD 分离、量化路径）。

此外，KDA 算子华为已有 `aclnnChunkKdaFwd`，本仓库补了缺失的 Python wrapper 并完成验证。

## 核心成果

| 算子 | 类型 | 作用 | 验证结果 |
|---|---|---|---|
| `SituAndMul`（SiTU-GLU） | 自研 AscendC kernel | K3 MLP 激活函数 | `max_rel ≈ 7.5e-3`，带宽达标率 **81.5%**，加速比 **4.45x~7.68x** |
| `AttnRes`（残差栈） | 自研 AscendC kernel | K3 线性注意力残差栈 | `max_rel ≈ 7.7e-3` |
| `SituGluFused`（融合） | 自研 AscendC kernel | GEMM + SiTU epilogue | 4 组 shape 数值全 PASS（含真实尺寸 `M=128 H=1024 d=11008`） |
| KDA | 补 Python wrapper | 线性注意力（Gated Delta Rule） | vs 参考实现 `~4e-3` |
| LatentMoE 投影 | 图级组合（非新 kernel） | down/up 投影接线 | 与官方同权重超容差 **0 / 384** |

> 数值容差判据：bf16 输出 `max_rel < 1e-2`。验证采用「无权重」方式（shape + 随机数据 + 参考实现对拍），不需要下载模型权重。
> 三条算子（SituAndMul / AttnRes / SituGluFused）已注册为 `torch.ops.kimi3_ops.*`（含 fake/meta + `torch.compile` 图捕获 + CPU 兜底）。

## 算子详解

### SiTU-GLU（`SituAndMul`）

K3 的 MLP 激活函数。SiTU 与常见的 SiLU 不同——它是 sigmoid-tanh 的乘积形式，且 up 分支做 tanh 压缩：

```
gate, up = x[..., :d], x[..., d:]                 # d = intermediate_size
situ(gate) = beta * tanh(gate / beta) * sigmoid(gate)
up'        = linear_beta * tanh(up / linear_beta)
out        = situ(gate) * up'
```

KIMI3 配置：`beta = 4.0`、`linear_beta = 25.0`。参考实现在 fp32 内部计算、输出转回 bf16，这是对拍容差的依据。

### AttnRes（残差栈）

本质是一个「mini-attention」：`B+1` 个残差向量做 RMS-norm + 学到的 score 点积 + softmax + 加权求和：

```
v      = [block_residual; prefix_sum]            # (T, B+1, H)
k      = RMSNorm(v)                              # H 维归一化
scores = sum_h k * score_weight                  # 学到的常量点积
out    = softmax(scores) · v                     # 在 B+1 维加权求和
```

### SituGluFused（融合算子）

dense MLP 的 `GEMM + SiTU epilogue` 融合：核心难点是把 gate/up 权重**重排进同一个 tile**，让 Matmul 一次算完再在 AIV 上做 SiTU 后处理。数值已真机对拍 PASS；性能仍在优化（GEMM tiling 待提升，见「当前状态与局限」）。

### KDA（Gated Delta Rule）

线性注意力的状态递推：

```
S_t = S_{t-1} * exp(g_t) + β_t · k_t · (v_t - k_tᵀ·S_{t-1})ᵀ
o_t = q_tᵀ · S_t
```

三个 in-kernel 预处理（modeling 源码都启用）：① q/k L2 norm（head_dim 维）；② gate：`g = -5.0 * sigmoid(exp(A_log) * (g_raw + dt_bias))`（safe_gate）；③ beta：`sigmoid(beta_raw)`。

## 性能数据（A3 真机，两条独立路线互证）

| 口径 | 数字 |
|---|---|
| C++ 直调（`scripts/bench_situ_aclnn.cpp`） | 0.4003 ms / 985 GB/s / **带宽达标率 81.5%** |
| 经 torch_npu（`bench_situ_real.py`，大 shape） | 0.4568 ms / 987 GB/s / 77.6% |
| **加速比 vs 等价 torch 组合** | **4.45x（大 shape）~ 7.68x（小块）** |
| 峰值带宽 | 1272.9 GB/s（torch copy）/ 1208.7 GB/s（C++ copy） |

## 部署要点

K3 是典型 **memory-bound** 部署，关键结论（详见 [`deploy/`](deploy/) 与 [`docs/技术笔记.md`](docs/技术笔记.md)）：

- **最小可行规模**：64 卡 = 4 节点 Atlas 800 A3（`DP4 / TP16 / EP64`）；生产 PD 分离 16 节点（8P+8D）
- **显存墙**：2.8T 参数常驻，MXFP4 约 1.5TB，加 KV/buffer 约 40 卡 → 工程取整 64 卡；算力只要 1~2 卡等效
- **MXFP4 非原生**：A2/A3 硬件原生仅支持 INT8/FP8，MXFP4 须软件反量化（W8A8）
- **PD 分离**：KV cache 通过 RDMA 单边传输，Prefill/Decode 各自独立扩缩容
- **跨 rank 交换**：896 专家 ÷ EP64 = 14 < 每 token 激活 16，跨 rank 交换必然发生

## 目录结构

```
ascend-kimi-ops/
├── ops/                       # Ascend C 算子实现（源码 + 编译成品 + 参考实现 + 测试 + 文档）
│   ├── kernels/               # AscendC 算子工程
│   │   ├── SituAndMul/        #   SiTU-GLU（自研）
│   │   ├── AttnRes/           #   AttnRes 残差栈（自研）
│   │   ├── SituGluFused/      #   融合算子（GEMM + SiTU epilogue）
│   │   └── Kimi3Ops/          #   三算子合成一个包（vendor 一次只装一套）
│   ├── reference/             # 参考实现（正确性金标准）
│   ├── integration/           # torch.ops 注册 + vLLM/SGLang 接线 + LatentMoE
│   ├── scripts/               # 上机探针 / 一键验收 / 性能与正确性基准 / C++ 直调
│   ├── triton_kernels/        # Triton 实现（三实现对比的第三极）
│   ├── tests/                 # 纯 torch 自测
│   ├── docs/                  # 知识库 01~20（环境/流程/性能/tiling/融合/实测记录/收益总账）
│   ├── dist/                  # 编译成品（.run 离线安装包）
│   ├── *_wrapper.cpp / *_op.json / aclnn_ops.py
│   └── 设计验证测试报告.md / .pdf
├── deploy/                    # Kimi K3 昇腾部署方案
└── docs/                      # 技术笔记
```

## 验证方法

无权重验证：从 config 取 shape/dtype → 固定 seed 生成随机输入 → 参考实现跑 golden 输出 → 自研算子跑候选输出 → 对比 shape/dtype/layout 与数值容差（fp32→bf16 用 `rtol=atol=1e-2`）。

验证机制（判据先行 / 独立第二实现互证 / 负对照 / 证据分级 / 失败取证）详见 [`ops/README.md`](ops/README.md) 与 [`docs/技术笔记.md`](docs/技术笔记.md)。

## 构建与测试

> 需要昇腾 A3 环境（CANN 9.0.0 + torch_npu），完整依赖与前提见 [`ops/README.md`](ops/README.md)。

```bash
# 编译算子（需先 source CANN 环境）
bash ops/kernels/SituAndMul/build.sh    # SiTU-GLU
bash ops/kernels/AttnRes/build.sh       # AttnRes

# 对拍验证（NPU 上，与参考实现交叉对比）
python ops/test_situ_real.py            # SiTU-GLU
python ops/test_attn_res_real.py        # AttnRes

# 参考实现自测（纯 torch，无需 NPU）
python -m pytest ops/tests/
```

## 当前状态与局限

| 项 | 状态 |
|---|---|
| 融合算子（GEMM+SiTU） | 数值验证通过；性能仍在优化（GEMM tiling 待提升到峰值带宽） |
| 端到端 TTFT/TPOT | 待跑（依赖 `triton-ascend >= 3.2.1`，公开源尚未发布） |
| 与华为算子的直接对比 | 华为无 SiTU-GLU / AttnRes，无直接对标对象；对照口径为等价 torch 组合 |

## License

[MIT](LICENSE)
