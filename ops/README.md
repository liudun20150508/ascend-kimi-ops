# KIMI3 算子开发工作区

只靠 shape + 随机数据 + 参考实现来验证算子正确性，不需要下载模型权重。

## ★ 这个项目交付的是什么（定位，先看这段）

**是「芯片算子」，不是「PyTorch 自定义算子」。** 四层归属（每层都有实测证据）：

| 层 | 产物 | 在哪 | 证据（可当场复现） |
|---|---|---|---|
| ① **device 侧 kernel** | **AscendC**（类 CUDA C）kernel → 编成 NPU 二进制 | `kernels/*/op_kernel/*.cpp` → `.run` → `opp/vendors/customize/` | `__aicore__` / `TQue` / `DataCopy` / `InitBuffer` / `GetBlockIdx` **24 处** |
| ② **CANN 层** | **aclnn C 接口**（算子的对外 API），**可进 GE 图** | `vendor/op_api/include/aclnn_{situ_and_mul,attn_res,situ_glu_fused}.h` | `vendor/.../npu_supported_ops.json` 里登记了三个算子 |
| ③ **PyTorch 层** | `torch.ops.kimi3_ops.*`（`torch.library.custom_op`，含 fake/meta + CPU 兜底） | `integration/torch_ops_kimi3.py` | `torch.ops.kimi3_ops.situ_and_mul(x,4.0,25.0)` → 输出 `device = npu:0` |
| ④ **模型/图层** | 模型文件里替换 `act_fn` / LatentMoE 接线 | `integration/vllm_situ_glu.py`、`latent_moe.py` | L2 官方 `KimiMLP` 同权重：超容差 **0 / 131,072** |

**并且 ①~③ 可以完全绕开 PyTorch**：`scripts/bench_situ_aclnn.cpp` 直接用 `aclrt` + aclnn 调
（这正是"独立第二实现互证"里的那条路线）。

> 一句话：**AscendC kernel 是本体，CANN 是它的家，PyTorch 只是它的一个调用方。**
>
> 为什么这个区分重要：常见的"写算子"是 `torch.autograd.Function` 包一层、或写 **CUDA 扩展**（产物是 Python/C++ 扩展）；
> 而这里的产物是 **CANN 算子**（aclnn 接口 + NPU 二进制 + 图模式声明），能被 PyTorch / vLLM / SGLang 调用，
> **也能被裸 C++ 调用** —— 前者是在框架内使用，后者是**芯片级算子开发**。

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
│   ├── SituAndMul/             #   ✅ SiTU-GLU（已开发 + 验证 + 性能优化）
│   ├── AttnRes/                #   ✅ AttnRes 残差栈（已开发 + 验证）
│   ├── SituGluFused/           #   ✅ 融合算子（GEMM + SiTU epilogue，真机对拍 PASS）
│   └── Kimi3Ops/               #   ★ **三个算子合成一个包**（vendor 一次只装得下一套）
├── situ_and_mul_wrapper.cpp    # SiTU-GLU torch 包装（ACLNN_CMD）
├── situ_glu_fused_wrapper.cpp  # **融合算子** torch 包装（aclnnSituGluFused）
├── test_situ_glu_fused_real.py # 融合算子 torch 侧 harness + 对拍（含自动两段式权重重排）
├── attn_res_wrapper.cpp        # AttnRes torch 包装
├── aclnn_ops.py                # ★ 三个算子的 aclnn 运行时（JIT 包装**唯一实现**；接入层与 harness 都从这里调）
├── test_situ_real.py           # SiTU-GLU 对拍 harness
├── test_attn_res_real.py       # AttnRes 对拍 harness
├── bench_situ_real.py          # SiTU-GLU 性能基准：带宽达标率（问答要的「快多少」）
├── bench_gemm_real.py          # GEMM 算力达标率（MFU）：算力受限算子的对标口径
├── mxfp4_dequant_check.py      # MXFP4 单层反量化 + 精度对拍 + 显存账（零依赖可跑）
├── triton_kernels/                     # Triton 实现（三实现对比的第三极）
│   └── situ_glu_triton.py
├── scripts/
│   ├── probe_ascend_env.sh     # 上机第一件事：固化设备数/单卡显存/CANN 版本
│   ├── verify_fused_rebuild.sh # 融合算子一键验证：编译 → 安装 → 对拍 → 失败自动取证
│   ├── bench_swiglu_vs_situ.py # ★「相对华为」那一格：内置 SwiGlu vs 自研 SiTU-GLU，同卡同 shape 同判据比带宽达标率
│   └── check_fused_index.py    # 融合算子「重排 + 取数公式」纯 CPU 自检（30 秒，含反向自证）
├── situ_op.json / attn_res_op.json  # msopgen 算子定义
├── docs/                       # 知识库（编号 01~18，见目录）
│   ├── 01-环境与SOC.md
│   ├── 02-AscendC算子开发流程与坑.md
│   ├── 03-KDA真实算子对比.md
│   ├── 04-性能分析与带宽达标率.md      # 理论下限 / 80% 判定线 / msprof 拆解 / 排查树
│   ├── 05-GEMM-tiling推导与达标率.md   # 「tiling 怎么定的」的完整推导
│   ├── 06-Triton-三实现对比.md
│   ├── 07~13 …                 # 融合设计 / 框架落地 / 双机规划（见目录）
│   ├── 14-上机实测记录.md       # ★ 全部真机数字的原始出处（含 §16 融合算子重建）
│   ├── 17-收益总账与相对华为口径.md # ★ 收益三层账 + 「相对华为」的正确口径 + 话术
│   └── 18-相对华为口径与K3算子落位.md # ★ 两问的答卷（那格 ❌ 该不该做 + K3 部署里算子怎么落位）
└── tests/                      # 参考实现自测（纯 torch）
    ├── test_situ.py / test_kda.py / test_kda_chunked.py
```

## 性能数据（问答「快多少」的答案在哪）

| 问题 | 数字从哪来 | 判定标准 |
|---|---|---|
| SiTU-GLU 快多少 / 达标吗 | `bench_situ_real.py`（`--bench-bw-only` 先测峰值带宽） | 访存受限算子，**带宽达标率 ≥80%** 才算到位 |
| AttnRes 呢 | 同思路，瓶颈是两趟读 + 归约，见 `docs/04 §7` | 按「每 token 访存字节 ÷ 带宽」算下限 |
| GEMM/Matmul 呢 | `bench_gemm_real.py`（`--bench-peak-only` 先测峰值算力） | 算力受限算子，**MFU ≥70%**；`M=1` 的 GEMV 要换带宽口径 |
| tiling 怎么定的 | `docs/05` 的三级容量推导（L0C → L0A/L0B → L1） | 推导出来的，不是试出来的 |
| 收益是多少 / 相对华为怎么说 | `docs/17` + `benefit_total.py` | 口径 A（分母 = 等价 torch 组合）、口径 B（省的是 HBM 往返，只在 prefill 放大） |

> 这两个数字（带宽达标率 / MFU）就是「我不只会写算子，还能量化自己的工作」的证据。
> 只报加速比不报达标率，回答不了「还有多少空间」这个追问。

**★ 实测（A3 真机 2026-09-19，两条独立路线互证）**：

| 口径 | 数字 |
|---|---|
| C++ 直调（`scripts/bench_situ_aclnn.cpp`） | 0.4003 ms / 985 GB/s / **达标率 81.5%** |
| 经 torch_npu（`bench_situ_real.py`，大 shape 8×512×22016） | 0.4568 ms / 987 GB/s / **达标率 77.6%** |
| **加速比 vs 等价 torch 组合** | **4.45x（大 shape）~ 7.68x（小块）** |
| 峰值带宽 | 1272.9 GB/s（torch copy）/ 1208.7 GB/s（C++ copy） |

```bash
# torch_npu 路线的跑法（缺 triton 也没关系：脚本会自己判断）
/usr/local/python3.11.15/bin/python3 scripts/run_torch_harness.py test_situ_real.py
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
| **LatentMoE 投影** | ❌ 形状不符 | 🔧 **无需新 kernel**：图级组合（down/up Linear + MoE + RMSNorm，expert 内用 SiTU-GLU）；**接线层已落地并验证** `integration/latent_moe.py`（✅ 自检两段全过：不变量探针 11 项 + 与官方 `KimiSparseMoeBlock` 同权重**超容差 0 / 384**；语义照官方 `reference/modeling_kimi_linear.py:776-838` 逐行对齐） |

**结论**：华为真正缺的算子只有 2 个（SiTU-GLU、AttnRes），均已自研并验证通过；LatentMoE 是模型图接线问题（在 vllm/sglang 适配层插 down/up 投影），不属算子开发。

**验证路线**：✅ C++ 直调（aclrt+aclnn）与 ✅ **torch_npu（2026-09-19 打通）** 两条都跑通。
融合算子（GEMM + SiTU epilogue）：旧版编译/安装/加载全通但运行时挂住 → **已按官方 MIX 样例重写 kernel/host
并上机跑通（`docs/14` §16.4：4 组 shape 数值全 PASS，含真实尺寸 `M=128 H=1024 d=11008`）**。
⛔ 重复执行报 `507057`（空 kernel 也复现 ⇒ 环境/框架层）；⏸ 阶段 2（C 取 UB）三种组合未通、已回退
⇒ 交付版是**通路验证版，无 HBM 收益**（`docs/14` §16.6）。

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
