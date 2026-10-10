# 把自研算子接进推理框架（vLLM / SGLang）

> 目标：让 KIMI3 的 SiTU-GLU / AttnRes 在框架里被真正用上，而不是停在"仓库里有个 kernel"。
> 本目录是**接入层**（框架适配层 + 图模型层的接线代码），算子本体在 `../kernels/`。

## 1. 三层接入（混在一起谈就会乱）

```
① 算子层        AscendC kernel + aclnn          →  ../kernels/SituAndMul、AttnRes
   ↓                                                
② 框架适配层    让框架能调到它                    →  本目录 torch_ops_kimi3.py（torch.ops 注册）
   ↓                                                vllm_situ_glu.py / sglang_situ_glu.py（模型层接线）
③ 图/模型层     算子怎么连、要不要融               →  MergedSituGluMlp（融合版接口，kernel 已就绪）
                                                    latent_moe.py（LatentMoE 接线，缺口第 ③ 项）
```

**关键认知**：SiTU-GLU **不是** vLLM/SGLang 的既有算子（两边都只有 `SiluAndMul`），所以它**没有
`CustomOp.forward_npu` 这种挂载点**，接入点必然落在 **KIMI3 的模型实现文件**里。这不是"绕过机制"，
而是"模型私有算子"的正确位置。

## 2. 本机已经验证过的（不需要 vLLM / 不需要权重）

```bash
/usr/local/python3.11.15/bin/python3 integration/torch_ops_kimi3.py    # 注册 + 图捕获
/usr/local/python3.11.15/bin/python3 integration/vllm_situ_glu.py      # vLLM 侧替换后数值等价
/usr/local/python3.11.15/bin/python3 integration/sglang_situ_glu.py    # SGLang 侧同上
```

| 验证项 | 结果 | 为什么重要 |
|---|---|---|
| `torch.ops.kimi3_ops.situ_and_mul(x, β, lb)` 直接调用 | ✅ 落 NPU（`npu:0`），与 golden 0 超容差 | 框架要的是"torch op"，不是"某个函数" |
| **fake/meta 实现** | ✅ `(2,7,22016) → (2,7,11008)` | 没有它就**不能图捕获**，动态 shape/显存预估也断 |
| **`torch.compile` 图捕获** | ✅ `aot_eager` 后端跑通，与 eager 差 0.0 | 对应 ACL Graph / `--compilation-config` 路线 |
| 替换后与 vLLM native（同精度）一致 | ✅ **0 / 45,088,768 元素超容差** | 证明"换掉激活不改结果" |
| 替换后 vs 真实 bf16 基线 | 0.3% 元素超容差 | 基线逐算子 bf16（~6 次舍入）比我们的 fp32 内核粗糙 ⇒ 换上是**精度更稳** |
| CPU 参考路径 | ✅ | 相当于 `forward_native`，框架永远有兜底 |
| **融合算子 `situ_glu_fused`**（2026-09-19 新加） | ✅ 4 组 shape 数值 ALL PASS（超容差 0）；`torch.ops` 路径、fake/meta、`torch.compile`、CPU 兜底**全通** | 图级融合的入口：能直接替换 `MergedSituGluMlp.forward` 里那两行 |
| **★ L2：接进真实模型层**（2026-09-19 新加，`../scripts/l2_mlp_check.py`） | ✅ **官方 `KimiMLP` 同权重对比**：`d=11008 / M=128` 下**超容差 0 / 131,072**（max\|diff\| 1.953e-03）；整层 MLP **1.75x**（我的算子 177.8 μs vs reference 纯 torch 311.6 μs） | 补上 L1 答不了的两问：算子被**真实模型层**调用后是否仍正确、在**整层**尺度上快多少（Amdahl） |
| **LatentMoE 接线层 `latent_moe.py`**（2026-09-19 新加） | ✅ **段 1 + 段 2 全部跑通**：段 1 不变量探针 **11 项全 OK**；段 2 与官方 `KimiSparseMoeBlock` 灌同权重 —— **超容差 0 / 384 元素**（接线时序 gate→down→moe→norm→up 与官方一致） | 缺口清单第 ③ 项「LatentMoE 投影」的落地：它是**图接线**不是新 kernel（`docs/12` §4） |

> 判据一律用 **`±(atol + rtol·|ref|)`**，不用纯相对误差：`|ref|≈0` 处会把 bf16 的正常绝对误差
> 放大成假失败（`docs/14` §10 有实测记录）。

> ★ **NPU 路径的来源**：三个算子的 NPU 实现都走 `../aclnn_ops.py`（JIT 包装的**唯一实现**），
> 不再转发到 `test_*_real.py` —— 测试脚本退回「只当测试」。权重重排与两段式（先空跑读 subN）
> 也在那里：`aclnn_ops.interleave()` / `aclnn_ops.situ_glu_fused_auto()`。
> 安装状态可自查：`python3 aclnn_ops.py`（只查文件系统，不需要 NPU）。

## 3. ⛔ 端到端仍不可行，但卡点只剩**一个包**（2026-09-19 更正）

### 3.1 原判断（**已被推翻**，保留备查）

```
$ pip download vllm-ascend==0.9.2rc1 --no-deps && unzip -p *.whl '*/METADATA' | grep Requires-Dist
Requires-Dist: torch-npu==2.5.1.post1.dev20250619      ← 钉死 torch_npu 2.5.1
Requires-Dist: torchvision<0.21.0
Requires-Dist: numpy<2.0.0
```

⇒ 当时结论：「vllm-ascend 0.9.x 属 CANN 8.x + torch_npu 2.5.1 时代，装上会与 CANN 9.0 打架」。

**错在只看了 0.9.x 一个版本**——它不是最新版。

### 3.2 实际情况（用 pypi JSON API 实测的版本矩阵）

| vllm-ascend | 要求 torch-npu | 要求 triton-ascend | 本机能否满足 |
|---|---|---|---|
| 0.19.1rc1 | 2.9.0 | **3.2.0 ✓**（pypi 有） | ✗ 要降 torch_npu |
| **0.20.2rc1 / 0.21.0rc1 / 0.22.1rc1** | **2.10.0 ✓** | 3.2.1 ✗ | **只差 triton 一个小版本** |
| 0.23.0（正式版） | 2.10.0.post4 | 3.2.2 ✗ | 差两处 |

`pip install --dry-run vllm-ascend==0.22.1rc1` 实测输出：

```
Requirement already satisfied: torch==2.10.0      (2.10.0+cpu)   ← 本机满足
Requirement already satisfied: torch-npu==2.10.0  (2.10.0)       ← 本机满足
ERROR: Could not find a version that satisfies the requirement triton-ascend==3.2.1
       (from versions: 3.2.0rc2, 3.2.0rc3, 3.2.0rc4, 3.2.0)
```

⇒ **torch / torch_npu 是同代的**（0.20.2~0.22.1 要的正是本机的 2.10.0）。真正的卡点是
**`triton-ascend` 版本缺口**：pypi 与华为镜像（`repo.huaweicloud.com/repository/pypi/simple`）
**最高都只有 3.2.0**，而 vllm-ascend 要 3.2.1/3.2.2。

### 3.3 ⚠️ 别拿 3.2.0 去顶 —— 它与 CANN 9.0 **不配套**（实测）

装了 `triton-ascend==3.2.0` 后连 `import triton` 都过不去（fla 的 import 会拉起 triton driver）：

```
error: 'RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE' is not a member of 'rtLimitType_t'
      .../triton/backends/ascend/npu_utils.cpp:321
```

⇒ 它要的驱动 API 在本机 CANN 9.0.0 的 driver 头里**不存在**。所以 vllm-ascend 要 3.2.1+
**不是保守，而是 3.2.0 确实不适配 CANN 9.0**。「用 3.2.0 顶一下」这条路**不成立**。

**结论（更正后）**：端到端仍不可行，但原因**不是**「版本代际不匹配」，而是
「**`triton-ascend >= 3.2.1` 在公开源上拿不到**」。等它发布、或从华为官方渠道获取即可。

> 附带实测：`vllm` 本体**从 0.27.1 起才有 aarch64 wheel**
> （`vllm-0.27.1-cp38-abi3-manylinux_2_28_aarch64.whl`），本机架构对得上；
> 但 vllm-ascend 与 vllm 的版本配套关系**尚未核对**。

> 另注：**SGLang 同理**（其昇腾后端同样受 `triton-ascend` 约束）—— 这条原判断仍然成立。

## 4. 环境就绪后的执行清单（照做即可）

```bash
# ① 确认版本配套（三者必须一起选：CANN ↔ torch_npu ↔ vllm-ascend/sglang）
python3 -c "import torch_npu; print(torch_npu.__version__)"
# ② 装框架（版本以官方配套表为准，别自己猜；内网只有阿里云源可达）
pip install vllm-ascend==<与本机 torch_npu 配套的版本>

# ③ 打本目录的补丁（两处：注册 + 模型里换 act_fn）
#    模型文件位置：pip show -f vllm-ascend / vllm 之后按 docs/12 §2 的方法定位
# ④ 冒烟：用极小模型（如 tiny 配置）先跑通 forward，再上真权重
# ⑤ 量收益：对照未替换版本，看 TTFT / TPOT / 每层耗时（这是问答要的端到端数字）
```

**没有权重时能做到哪一步**（本机现状）：

| 项 | 可做 | 说明 |
|---|---|---|
| 算子正确性 / 性能 | ✅ | 已做（`../docs/14` §9/§10/§14.3） |
| 算子被 torch 当一等公民（含图捕获） | ✅ | 本目录 `torch_ops_kimi3.py` |
| 单层 MLP 级别的"替换前 vs 替换后"数值与时延 | ✅ | `vllm_situ_glu.py` 自检里可扩 |
| 端到端 TTFT / TPOT / 吞吐 | ⛔ | 需权重 + 框架（本机两者都缺） |

## 5. 文件清单

| 文件 | 作用 |
|---|---|
| `torch_ops_kimi3.py` | **核心**：`torch.library.custom_op` 注册 `kimi3_ops::situ_and_mul` / `::attn_res` / `::situ_glu_fused`（**三个**，各自含 fake/NPU/CPU 三份实现），带自检 |
| `vllm_situ_glu.py` | vLLM 侧：`SituAndMulModule`（替换 MLP 的 `act_fn`）、`MergedSituGluMlp`（融合版接口）、端到端清单，带自检 |
| `sglang_situ_glu.py` | SGLang 侧：同样结构的层 + vLLM↔SGLang 概念映射表，带自检 |
| `latent_moe.py` | **缺口清单第 ③ 项**：LatentMoE 的图接线层（`LatentMoeProjection` + `latent_moe_forward`），照官方 `reference/modeling_kimi_linear.py` 逐行对齐；⚠️ 自检未在本机运行过 |
| `../situ_glu_fused_wrapper.cpp` + `../test_situ_glu_fused_real.py` | **融合算子**（GEMM+SiTU epilogue）的 torch 包装与 harness：`torch.ops.kimi3_ops.situ_glu_fused` 已注册；harness 含**两段式权重重排**（先空跑读 subN → 再按 subN 重排）|
