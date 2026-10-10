# Kimi K3 昇腾（Ascend）本地化部署方案

> 文档性质：技术方案 + 资源测算 + 成本估算（用于立项/报价评审）
> 目标平台：华为昇腾 Atlas 800 A3 / Atlas 800 A2
> 版本：v1.0 · 编制日期：待填
> 状态：**待评审**（标注「待确认」处需业务方补充）

---

## 0. 结论摘要（一页看懂）

| 项目 | 结论 |
|---|---|
| **部署方式** | 放弃「现有存量设备直接跑」的预期；K3 必须重新规划集群 |
| **最小可行规模** | **4 台 Atlas 800 A3 服务器**（共 64 个 64GB NPU 实例），拓扑 `DP4 / TP16 / EP64` |
| **生产规模（PD 分离）** | **16 台 Atlas 800 A3**（8 Prefill + 8 Decode），每侧 `DP8 / TP16 / PP1` |
| **支持 1M 上下文** | 需 **≥8 节点** |
| **A2 备选路径** | 8 节点 × 8 卡 = 64 卡，`DP8 / TP8 / EP64`，**仅支持 256K 上下文且需纯文本模式** |
| **硬件采购估算** | 最小可行约 **600~1000 万元**；生产级约 **2400~4000 万元**（估算，**须以厂商报价为准**） |
| **实施周期** | 最小可行 **4~6 周**；含 PD 分离与压测验收 **8~10 周** |
| **最大技术风险** | ① 权重为 MXFP4，A3 **非硬件原生支持**，须软件反量化 ② 896 专家负载均衡 ③ 大规模 EP 通信 |
| **硬性门槛** | 昇腾官方口径：**至少 64 卡**才可实现高效推理；8 卡服务器（显存与 HCCS 带宽不足）无法承载 |

---

## 1. 项目背景与目标

### 1.1 模型概况

Kimi K3 是 MoonshotAI 发布的混合架构超大规模 MoE 模型，核心规格：

| 项目 | 数值 |
|---|---|
| 总参数 | **2.8T（约 2.78 万亿）** |
| 激活参数 | **104B** |
| 网络层数 | **93 层**（1 dense + 92 MoE） |
| 注意力构成 | **69 层 KDA（线性注意力）+ 24 层 Gated MLA** |
| MoE 专家 | **896 路由专家**，每 token 激活 **16 个 + 2 个共享专家**（约 56× 稀疏） |
| Latent MoE 维度 | 3584 |
| 单专家 MoE 隐藏维度 | 3072 |
| 隐藏维度 / 头数 | 7168 / 96 |
| 词表 | 160K |
| 上下文长度 | 1,048,576（约 100 万 token） |
| 视觉编码器 | MoonViT-V2（27 层，401M） |
| 原生量化 | **MXFP4 权重**（group_size=32）+ 动态 MXFP8 激活（QAT） |
| 权重体积 | 约 **1.5TB**，96 个 safetensors 分片 |
| 许可 | 自定义 Kimi K3 License（MaaS 商用有营收阈值条款，**须法务确认**） |

### 1.2 部署三大难点

1. **显存墙**：2.8T 参数必须全部常驻，单卡 64GB 仅放权重就需 23 卡（MXFP4）至 44 卡（W8A8）。
2. **专家负载失衡**：896 专家、每 token 激活 16 个，跨 rank 交换不可避免，需大 EP + 通算融合。
3. **KDA 计算密度低**：线性注意力算力密度低，长序列 Prefill 是时延瓶颈。

### 1.3 方案目标（待确认）

- 目标并发 / QPS：**待确认**
- 上下文长度分布（平均 / P99）：**待确认**
- 是否需要多模态（图像）输入：**待确认**
- SLA（TTFT / TPOT）：**待确认**

---

## 2. 资源需求测算

> 所有结论可用本节公式复算。**测算的核心结论：K3 的瓶颈是显存容量，不是算力。**

### 2.1 显存账（决定卡数）

```
权重显存 = 2.8T 参数 × 每参数字节数
  ├─ 保持 MXFP4 在显存（0.5 B/参数）  → 约 1.4~1.5 TB
  ├─ 反量化常驻 W8A8（1 B/参数）      → 约 2.8 TB
  └─ FP16/BF16（2 B/参数）            → 约 5.6 TB（不经济，卡数约为 MXFP4 的 4 倍）

KV cache = 24 层 MLA × 序列长度 × 每 token 每层字节数
  ※ 其余 69 层为 KDA 线性注意力，状态与序列长度无关（这是 K3 能跑 1M 上下文的原因）
  ※ 官方口径：KV cache 压缩约 75%

激活与通信 buffer = 大 EP alltoall 缓冲区（不可忽略，占比可观）
```

**按 64GB / NPU 实例推算：**

| 精度路径 | 纯权重卡数 | 加 KV cache / buffer（按显存占用率 55~60%） | 结论 |
|---|---|---|---|
| **MXFP4 常驻**（推荐） | ~23 | **~40** | 工程取整为 **64 卡**——4 节点 × 16 NPU 是自然拓扑单元，并为长上下文 KV 预留余量 |
| 反量化 W8A8 常驻 | ~44 | ~76 | **卡数近乎翻倍，不推荐** |

**⇒ 关键结论：优先保持 MXFP4 常驻（见 §3.4 路线 A），避免权重显存翻倍导致集群规模失控。**

**⇒ 这与「昇腾官方要求至少 64 卡」的结论一致，互为印证。**

**关键变量：卡数由单卡显存决定**

| 单卡显存 | 最小可行卡数 |
|---|---|
| 64GB | **64** |
| 128GB | **32**（理论，需实测验证，官方基准仍为 64 卡） |

> **行动项**：在目标机执行 `npu-smi info` 确认单卡 HBM 容量与实例口径，将实测值固化进本方案。

### 2.2 算力账（不是瓶颈）

```
每 token 前向 FLOPs ≈ 2 × 激活参数(104B) = 约 208 GFLOP
1000 tokens/s 吞吐 → 约 208 TFLOPS 需求
按 30% MFU 计 → 约 700 TFLOPS 有效算力 ⇒ 仅约 1~2 卡的等效算力（此处为算力折算，**非设备数量**）
```

**⇒ 算力需求约 1~2 卡等效，显存需求 40+ 卡。K3 是典型的 memory-bound 部署，资源规划必须以显存为准，绝不能按算力估卡。**

> 注：以上为理论上限推算。实际达成率受大 EP 通信开销与小 GEMM 效率限制，需在 P3 阶段实测修正。

### 2.3 通信账（决定拓扑）

- 896 专家 / EP64 ⇒ **每 rank 约 14 个路由专家**，而每 token 需访问 16 个 ⇒ **跨 rank 交换是必然动作**。
- 依赖 **MC2 通算融合**（Token 重排 + 跨 rank 分发 + 局部专家计算 + 结果回收组织成设备侧流水线）。
- 依赖 A3 的 **784 GB/s D2D 互联带宽**（1:1 无收敛）。
- **普通 8 卡 HCCS 服务器无法承载此通信量**——这是「必须整机整节点规划」的根本原因。

---

## 3. 技术方案

### 3.1 硬件配置

**推荐（A3）**：4 台 Atlas 800 A3 超节点服务器

| 项目 | 配置 |
|---|---|
| NPU | 每台 16 个 64GB NPU 实例（单机片上内存 1024GB） |
| CPU | 鲲鹏 920 |
| 卡间互联 | D2D 双向 784 GB/s |
| 网络 | 400GE RoCE（多机全互联） |
| 集群规模 | 4 节点（可扩展至 384 卡超节点） |

**备选（A2）**：8 节点 × 8 卡 = 64 卡。**限制：仅 256K 上下文、须 `--language-model-only`（纯文本）**，故仅作为预算受限时的降级方案。

### 3.2 软件栈与版本

| 组件 | 版本 / 镜像 | 说明 |
|---|---|---|
| CANN | **9.0.0 / 9.0.1** | A3 推荐；部分 MXFP4/AscendC 算子需用 `bisheng` 自行编译 |
| 推理引擎 | **vLLM-Ascend**（≥ v0.23.0 首次支持 K3，v0.26.0rc 完整适配）或 **SGLang** | |
| 官方镜像 | `quay.io/ascend/vllm-ascend:kimi-k3-a3`（Ubuntu）/ `...-openeuler`；A2 用 `kimi-k3` 标签 | **必须用官方镜像**，避免量化配置类与上游版本失配 |
| SGLang 镜像 | `lmsysorg/sglang:cann9.0.0-a3-k3` | |

> ⚠️ **MindIE 不适配**：MindIE LLM 多机特性仅支持 TP 并行、且模型支持列表不含 K3，与 K3 所需的 EP64 大专家并行路径不兼容。**不建议在 MindIE 上规划本方案。**

### 3.3 并行策略

**最小可行（混合部署）**

```
DP4（跨 4 节点） / TP16（单节点内 16 NPU 全做 TP） / EP64（896 专家摊到 64-rank EP 组）
```

**生产形态（PD 分离）**

```
Prefill 侧：8 节点，DP8 / TP16 / PP1
Decode   侧：8 节点，DP8 / TP16 / PP1
KV 传输：memcache KV 池 + Mooncake Connector
```

**长上下文（1M）**：≥8 节点，`DP8`，`--max-model-len 1048576`。

### 3.4 量化方案（**本项目最大风险点，须优先验证**）

**事实陈述**：

> 昇腾 **A2 / A3（含 910C）硬件原生仅支持 INT8、FP8**；**MXFP4 原生支持属于 A5（950）/ Atlas 350**。
> 而 K3 原生权重是 **MXFP4**。

**因此 A3 上部署 K3 走「软件反量化」路径**，二选一：

| 路径 | 做法 | 优点 | 风险 |
|---|---|---|---|
| **A. CANN 社区 recipe** | 使用 `cann-recipes-infer` 的 AscendC MXFP4→W8A8 融合算子；`KT_MXFP4_DEPOOL=1` 时按需将专家权重从 DDR 搬入 HBM | 省 HBM、已在 A3 落地 | 需自行用 `bisheng` 编译 `libmxfp4fused.so`；缺该算子时长 Prefill 会失败 |
| **B. msModelSlim 离线转换** | 将 MXFP4 权重离线转为 W4A8 再部署（路由专家 w1/w2/w3 做 W4A8 动态量化；Router 保持高精度） | 链路更可控 | 权重体积增大 → 卡数需求上升 |

> ⚠️ msModelSlim 官方案例明确提示：**K3 即使逐层加载，单卡 64GB 显存仍会溢出，必须先完成专家并行适配（`ep_patches.py`）**。
> ⚠️ 常见坑：量化基底不一致（W8A8 与 GGUF）；MXFP4 对账需配套转换器与 `verify_mxfp4_layer.py` 校验 nibble 序。

**选型建议：优先路线 A。** 路线 B 若常驻 W8A8 权重（约 2.8TB），会直接导致卡数从 ~40 涨到 ~76（见 §2.1）。若走路线 B 的 W4A8 分支可缓解显存问题，但需重新量化 + 全量精度验证，项目周期会显著拉长。

### 3.5 服务化配置（A3 四节点参考）

```bash
# Node0（主节点）
vllm serve <model_path> \
  --tensor-parallel-size 16 \
  --data-parallel-size 4 \
  --data-parallel-size-local 1 \
  --enable-expert-parallel \
  --max-model-len 133120 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.85 \
  --quantization ascend \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true,"enable_flashcomm1":true}' \
  --tokenizer-mode kimi_k3 \
  --reasoning-parser kimi_k3 \
  --tool-call-parser kimi_k3

# Node1~3：追加 --headless，并将 --data-parallel-start-rank 分别设为 1 / 2 / 3
```

### 3.6 需启用的关键优化特性

| 特性 | 作用 |
|---|---|
| **KDA 融合算子** | Prefill 按 64 Token 分块批量化；Decode 用 Ascend C 融合 Kernel 一次完成 Q/K L2Norm、Gate/Beta、状态衰减、Delta Rule、外积更新与 Final State 回写 |
| **MC2 通算融合** | 支撑 896 专家大 EP 的通信-计算流水 |
| **Prefix Cache / Chunked Prefill / 异步调度** | 框架级吞吐优化 |
| **FlashComm1** | 多模态布局转换前移至第一层 Decoder 边界，减少重复 All-Gather |
| **DSpark 投机推理** | 压 decode 单步时延（SGLang 首发，vLLM-Ascend 亦支持） |

---

## 4. 性能预期

> ⚠️ **以下指标均需实测压测确认，本方案不提供承诺值。**

| 指标 | 预期 | 说明 |
|---|---|---|
| TTFT | 常规上下文可接受；**1M 上下文 Prefill 可能达数十秒~分钟级** | 长 Prefill 是本架构固有短板 |
| TPOT | DSpark 路径目标 **< 50ms** | 依赖投机推理 |
| 多模态 | **仅支持图像输入，不支持视频** | vLLM-Ascend FAQ 明确 |

---

## 5. 成本估算

> ⚠️ **以下为量级估算，非报价。实际成本须向华为及其渠道/租赁商询价。** 媒体与租赁商页面出现的成本数字属推算，不可作为依据。

| 项目 | 最小可行（4 台 A3） | 生产（16 台 A3） |
|---|---|---|
| 硬件采购 | 约 **600~1000 万元** | 约 **2400~4000 万元** |
| 机柜/电力/制冷 | 待测算（10U/台，功耗需按整机 TDP 核算） | 待测算 |
| 网络设备 | 400GE 交换机与光模块，待测算 | 待测算 |
| 权重存储 | 约 1.5TB（MXFP4）/ 2.8TB（W8A8），需配高速存储 | 同左 |
| 实施人力 | 2~3 人 × 4~6 周 | 3~4 人 × 8~10 周 |

**替代路径（建议同时评估）**：云上按需租用昇腾算力，避免一次性 CAPEX。适用于验证期与峰值弹性扩容。

---

## 6. 实施计划

| 阶段 | 周期 | 内容 | 交付物 |
|---|---|---|---|
| **P0 环境与选型确认** | W1 | 现场 `npu-smi` 盘点、CANN 9.0.0+ 与驱动、官方镜像部署、1.5TB 权重下载与校验 | 环境就绪报告 |
| **P1 单机跑通** | W2 | 单节点 TP16 拉起，短上下文推理，精度对拍 | 单机可用性报告 |
| **P2 多机组网** | W3 | 4 节点 RoCE 组网、EP64 拉起、专家负载均衡验证 | 4 节点可用性报告 |
| **P3 性能调优** | W4 | Prefix Cache、CUDA Graph、通算融合、batch 调参 | 性能基线报告 |
| **P4 PD 分离（可选）** | W5~W6 | 8P8D 部署、KV 池、Mooncake Connector | PD 分离方案 |
| **P5 压测与验收** | W7~W10 | 真实业务流量压测、SLA 验收 | 验收报告 |

---

## 7. 风险清单与应对

| # | 风险 | 等级 | 影响 | 应对 |
|---|---|---|---|---|
| R1 | **MXFP4 在 A3 非硬件原生**，须软件反量化 | **高** | 性能打折、显存需求上升、需自编译算子 | **P0 阶段单卡验证反量化链路**；优先选路线 A |
| R2 | 896 专家负载失衡 | 高 | 长尾 rank 拖慢整体 | 依赖 MC2 通算融合 + 官方 recipe 的均衡策略；压测验证 |
| R3 | 大规模 EP 通信开销 | 高 | 吞吐不达标 | 必须 A3 超节点（784GB/s D2D）；不可用普通 8 卡机 |
| R4 | 长 Prefill 时延 | 中 | 1M 上下文体验差 | Chunked Prefill + PD 分离 + 限制最大上下文 |
| R5 | 自编译 MXFP4 算子失败 | 中 | 首次长 Prefill 失败 | 准备 `KT_MXFP4_DEPOOL=0` 回退（代价：常驻整份 W8A8 池） |
| R6 | MindIE 路径不可用 | 中 | 架构选型返工 | 明确走 vLLM-Ascend / SGLang |
| R7 | **许可证合规** | **高** | 商用法律风险 | 法务确认 Kimi K3 License（MaaS 营收阈值、品牌展示条款）；企业内部使用豁免 |
| R8 | 供应链 / 到货周期 | 中 | 项目延期 | 提前锁定交期；并行评估云上租用 |
| R9 | 不支持视频输入 | 低 | 功能缺失 | 与业务方确认需求边界 |

---

## 8. 待确认项（业务方补充后方案方可定稿）

1. [ ] 目标并发数 / QPS / 日均 token 量
2. [ ] 上下文长度分布（均值、P99）
3. [ ] 是否需多模态（图像）能力
4. [ ] 是否必须 PD 分离
5. [ ] 现有昇腾设备清单（`npu-smi info` 输出）
6. [ ] 采购预算上限 / 是否接受云上租用
7. [ ] SLA 指标（TTFT、TPOT）
8. [ ] 是否有存量业务要求 MindIE 兼容

---

## 附录 A：现有单卡设备（1 × A3）的验证价值

> **范围说明**：本节针对编制方**现有的 1 张 A3 卡**，与本方案 §2 的采购规模（64 卡）**无关**——后者是客户侧硬件规划，不受现有设备约束。
> 建设期正式硬件到位前，用单卡排除最高风险项，降低后期返工成本。

### A.1 单卡容量约束（决定能做什么、不能做什么）

```
单专家参数量 = 3 × 3072 × 3584   ≈ 33 M        # latent 3584 × expert_hidden 3072
单 MoE 层     = 33M × 896 专家    ≈ 29.6 B
92 个 MoE 层  = 29.6B × 92        ≈ 2.72 T      # 与 K3 总参 2.8T 吻合，余量为 attention/embedding

单层显存占用：MXFP4 ≈ 14.8 GB   |   BF16 ≈ 59.2 GB
```

**⇒ 这精确解释了昇腾官方「K3 即使逐层加载，单卡 64GB 显存仍会溢出」的提示**：溢出发生在**以 BF16 载入单层（59.2GB）再做转换**的流程，叠加激活/KV 后必然超限；而非 MXFP4 原样读取路径。

**⇒ 单卡的验证边界：可以做「MXFP4 原样读取 + 反量化 + 对拍」（单层约 15GB）；不能做「BF16 载入 + 重新量化/校准」（单层约 60GB）。**

### A.2 单卡可完成的验证项

| 可验证项 | 可行性 | 方法 | 价值 |
|---|---|---|---|
| **MXFP4 单层反量化 + 精度对拍** | ✅ | 单层权重（约 15GB）加载 → 反量化 → 整层前向对拍 | **排除 R1（最大风险）** |
| K3 特有算子正确性 | ✅ | 已完成的 SiTU-GLU、AttnRes、KDA 对拍 | 直接作为交付物附录 |
| 单专家 FFN 精度 | ✅ | 单专家约 33M 参数（MXFP4 约 16.5MB），可完整加载 | 专家级数值正确性 |
| 单层峰值显存实测 | ✅ | MXFP4 单层加载实测 | 校验 §2.1 显存账推算 |

> 注：单专家体量仅 16.5MB，**性能样本过小**，不宜用于带宽/算力利用率测算；如需性能外推，应以**单层**为最小单元。

### A.3 单卡无法完成、须另行安排的项目

| 项目 | 原因 | 建议 |
|---|---|---|
| 重新量化 / 校准 | BF16 单层约 59.2GB，单卡必溢出 | 正式环境 + EP 适配（`ep_patches.py`）后做 |
| HCCL 通信微基准 | **需 ≥2 卡** | 借用 / 租用 2 卡 |
| MFU / TTFT / TPOT | 需完整集群 | 正式环境 P3 阶段压测 |
| EP64 专家负载均衡 | 需 EP64 拓扑 | 正式环境 P2 阶段验证 |
| 1M 上下文 Prefill 时延 | 需长序列 + 完整集群 | 正式环境 P5 阶段压测 |

> **硬件口径确认**：以上按单卡 64GB HBM 估算。请在目标机执行 `npu-smi info` 确认**实际设备数**与单卡显存，并将实测值固化进本方案。

---

## 附录 B：结论可复算的公式汇总

```
最小卡数 ≈ (权重显存 + KV cache + 通信/激活 buffer) / 单卡可用显存
         ≈ 2.8T × 每参数字节数 ÷ (单卡显存 × 可用率 0.55~0.60)

K3 每 token 前向 FLOPs ≈ 2 × 104e9 = 2.08e11
每秒算力需求(TFLOPS) ≈ 目标吞吐(tokens/s) × 2.08e11 ÷ 1e12
所需卡数(算力口径) ≈ 每秒算力需求 ÷ (单卡算力 × MFU)

每 rank 专家数 = 896 ÷ EP 度数           # EP64 → 14
跨 rank 交换必然发生 ⟺ 每 rank 专家数 < 每 token 激活数(=16)
```

---

## 附录 C：信息来源

| 内容 | 来源 |
|---|---|
| 模型规格、架构、量化格式 | [MoonshotAI/Kimi-K3](https://github.com/moonshotai/Kimi-K3)、[Kimi K3 技术报告](https://ar5iv.labs.arxiv.org/html/2607.24653) |
| 昇腾官方适配范围与部署拓扑 | [华为昇腾 0day 适配 K3](https://www.c114.net.cn/industry/105079.html)、[华为计算长文](https://www.163.com/dy/article/L2U2OKQI05568W0A.html) |
| vLLM-Ascend K3 部署教程与启动参数 | [vLLM-Ascend 官方文档](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/tutorials/models/Kimi-K3.html)、[PR #14125](https://github.com/vllm-project/vllm-ascend/pull/14125) |
| A2/A3 不原生支持 MXFP4 | [CANN 大模型量化基础](https://raw.gitcode.com/cann/cann-learning-hub/raw/master/tutorials/llm_inference/slides/04_llm_quantization_fundamentals.pdf)、[SGLang Ascend 量化矩阵](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu_quantization) |
| MXFP4 反量化 recipe（A3 已落地） | [cann-recipes-infer](https://gitcode.com/Twilight-Fanyi/cann-recipes-infer) |
| K3 W4A8 量化与 64GB 溢出提示 | [msModelSlim K3 案例](https://mindstudio-docs-master.readthedocs.io/zh-cn/latest/msmodelslim/docs/zh/best_practices/kimi_k3_adaptation_case/) |
| 至少 64 卡门槛 | [财联社/新华财经](https://www.cnfin.com/gs-lb/detail/20260728/4447047_1.html) |
| MindIE 支持范围 | [MindIE LLM 多机特性文档](https://www.hiascend.com/document/detail/zh/mindie/230/mindiellm/llmdev/mindie_llm0296.html) |
| Atlas 800 A3 硬件规格 | [华为企业业务官网](https://e.huawei.com/cn/products/computing/ascend/atlas-800t-a3) |
