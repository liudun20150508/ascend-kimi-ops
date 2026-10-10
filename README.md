# 昇腾 Ascend C 算子：Kimi K3 部署适配

本项目为 **Kimi K3 模型**（MoonshotAI，2.8T 参数 MoE）在**华为昇腾**国产芯片上的本地化部署而开发：补齐昇腾官方算子库缺失的算子，并给出完整部署方案。目标平台：昇腾 A3（SOC `ascend910_93`，CANN 9.0.0）。

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
- 给出 **LatentMoE 图级接线**的判断与参考（见 [`ops/README.md`](ops/README.md)）；
- 提供 K3 在昇腾上的**完整部署方案**（显存/算力测算、并行拓扑、PD 分离、量化路径）。

此外，KDA 算子华为已有 `aclnnChunkKdaFwd`，本仓库补了缺失的 Python wrapper 并完成验证。

## 内容

- **`ops/`** — Ascend C 算子实现（源码 + 编译成品 + 参考实现 + 测试 + 文档）
  - `SituAndMul`（SiTU-GLU）：自研 kernel，`d=11008` 时 `max_rel ≈ 7.5e-3`，带宽达标率约 81%
  - `AttnRes`（残差栈）：自研 kernel，`H=4096` 时 `max_rel ≈ 7.7e-3`
- **`deploy/`** — Kimi K3 昇腾部署方案
- **`docs/`** — 开发过程中的技术笔记与问题汇总

## 目录结构

```
ascend-kimi-ops/
├── ops/                       # Ascend C 算子实现
│   ├── kernels/
│   │   ├── SituAndMul/        #   SiTU-GLU（自研）
│   │   └── AttnRes/           #   AttnRes 残差栈（自研）
│   ├── reference/             # 参考实现（正确性金标准）
│   ├── tests/                 # 纯 torch 自测
│   ├── docs/                  # 算子开发文档
│   ├── dist/                  # 编译成品（.run 离线安装包）
│   ├── *_wrapper.cpp / *_op.json
│   ├── test_*_real.py         # 对拍 harness
│   └── 设计验证测试报告.md / .pdf
├── deploy/                    # Kimi K3 昇腾部署方案
└── docs/                      # 技术笔记
```

## 验证方法

无权重验证：从 config 取 shape/dtype → 固定 seed 生成随机输入 → 参考实现跑 golden 输出 → 自研算子跑候选输出 → 对比 shape/dtype/layout 与数值容差（fp32→bf16 用 `rtol=atol=1e-2`）。

详细说明见 [`ops/README.md`](ops/README.md)。

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

## License

[MIT](LICENSE)
