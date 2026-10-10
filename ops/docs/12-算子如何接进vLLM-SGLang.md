# 我的算子怎么进 vLLM / SGLang（算子适配 + 图级优化）

> 回答三个问题：**「图级优化」到底指什么？要改 vLLM 源码吗？自定义算子怎么注册？**
>
> ⚠️ 路径/类名会随版本变化，这里给的是**机制**和**该去哪个文件找**，具体行号请以本机
> `pip show -f vllm` / `python -c "import vllm, os; print(os.path.dirname(vllm.__file__))"` 为准。

---

## 0. 先把三层分清楚（这三个问题混在一起问就会乱）

| 层 | 干什么 | 改动落在哪 | 我的项目对应 |
|---|---|---|---|
| **① 算子层** | 写 kernel + aclnn 接口 | CANN `opp/vendors/customize` | `kernels/SituAndMul/`、`situ_and_mul_wrapper.cpp` |
| **② 框架适配层** | 让框架在 NPU 上调用正确实现 | `vllm-ascend`（plugin，**不是** vLLM 主干） | 目前**没做** |
| **③ 图/模型层** | 算子怎么连、要不要融、mem 复用 | 模型实现文件 或 plugin | LatentMoE 接线属于这层 |

**一句话**：算子写完了 ≠ 推理跑起来了。中间还差 ②③ 两层，而这两层**才是"适配"这个词的真正含义**。

---

## 1. 「图级优化」到底指什么

不是单一技术，是**在算子之上改写计算图**的一类手段：

| 手段 | 做什么 | 例子 |
|---|---|---|
| **算子融合** | 多个算子合成一个，省中间张量 | `Linear + SiTU-GLU` 融合；`Conv + BN` 折叠 |
| 常量折叠 | 编译期算掉常量表达式 | 位置编码表预计算 |
| 死代码消除 | 清掉不参与输出的分支 | 导出图里的多余分支 |
| 内存复用 | 中间张量的 buffer 重叠 | MLIR 的 bufferization / vLLM 的 KV pool |
| **图捕获** | 把整张前向图捕获成一次提交 | `torch.compile` / **CUDA Graph**（昇腾对应 ACL Graph） |
| 通算重叠 | 通信与计算交错调度 | PD 分离的 layerwise push |

**在 vLLM 里，能落到实处的只有前五种**，而且是分散在不同地方的：

- **融合** → 手写 fused kernel（`vllm/model_executor/layers/` 下一堆 `*_and_*` 的文件就是这个）
- **图捕获** → `--compilation-config`（torch.compile）/ CUDA Graph；调试时用 `--enforce-eager` 关掉
- **前四种** → 大多由 `torch.compile` 的 inductor 后端做，或者干脆在模型定义里就写好

> 所以「图级优化」在工程上**不是一个你在 vLLM 里"打开"的开关**，而是：
> **该融合的地方你手写了融合 kernel，该捕获的地方你开了图模式。**

---

## 2. 要改 vLLM 源码吗？（**通常不改**）

vLLM 为了支持多家硬件，把"可替换的部分"做成了扩展点：

```
vLLM 主干（NVIDIA 视角写）
   ↑ 不碰
vllm-ascend（昇腾 plugin）        ← 昇腾的活都在这层
   ↑ 不碰
你的自定义算子 / 模型文件          ← 你的活在这层
```

具体三个扩展点（按重要性）：

### ① `CustomOp` —— 算子级替换（**最重要**）

vLLM 把每个"可能有后端差异"的算子做成一个类，自带多份实现：

```python
# vllm/model_executor/custom_op.py（示意，行号随版本变）
class CustomOp(nn.Module):
    def forward_native(self, *a, **k): ...   # PyTorch 参考实现（永远能用）
    def forward_cuda(self,   *a, **k): ...   # NVIDIA 实现
    def forward_npu(self,    *a, **k): ...   # ← vllm-ascend 加的就是这个
    def forward(self, *a, **k):              # 运行时按当前 platform dispatch
        return self._dispatch(...)
```

典型成员：`SiluAndMul`、`RMSNorm`、`RotaryEmbedding`、`VocabParallelEmbedding`…

**我的算子要进去，就是在这里加一个 `forward_npu`**（或在 vllm-ascend 里给已有 CustomOp 补 NPU 实现），
`forward_npu` 里调用我的 aclnn 接口。

> 这也解释了为什么"SiLU 有融合算子而 SiTU 没有"——`SiluAndMul` 是 vLLM 的 CustomOp 成员之一，
> 昇腾给它实现了 `forward_npu`；而 **SiTU 根本不是 vLLM 的既有算子**，所以连挂载点都没有，
> 必须新加一个（或写进模型定义）。

### ② Platform 抽象 —— 设备能力声明

```python
# vllm/platforms/__init__.py 决定 current_platform
# vllm-ascend 实现 NPUPlatform：声明显存查询、是否支持图模式、attention 后端等
```

**这层决定框架"敢不敢用某些特性"**，比如能不能上 CUDA Graph、走哪个 attention 后端。
一般**不用你改**，vllm-ascend 已经做了。

### ③ Model Registry —— 新模型挂载

```python
# vllm/model_executor/models/registry.py：架构名 → 模型类 的映射
"KimiK3ForCausalLM": ("kimi_k3", "KimiK3ForCausalLM"),
```

**如果 vLLM 官方不支持 KIMI3，就要加一个模型文件 + 注册项。** 这一步在 vllm-ascend 或你自己的
fork 里做，同样**不动 vLLM 主干**。

---

## 3. 我的 SiTU-GLU 接进去的完整链路（4 步）

```
① 编 kernel 并安装
   cd kimi3_ops/kernels/SituAndMul && bash build.sh
   ./build_out/custom_opp_*.run --install-path=$ASCEND_HOME_PATH/opp --quiet
   → 落到 opp/vendors/customize/，产生 aclnnSituAndMul

② 包装成 torch 可调用（已有）
   situ_and_mul_wrapper.cpp（ACLNN_CMD 宏）→ 或者编成常驻 .so，不必每次 JIT

③ 在 vllm-ascend 里挂载（**新增的一步**）
   方案 a：给 vLLM 的对应 CustomOp 补 forward_npu
   方案 b：在 KIMI3 模型实现里，把 self.act_fn 直接换成调用 aclnnSituAndMul
   ┌ 注意：vLLM 官方没有 SituAndMul 这个 CustomOp，所以更可能是 b
   └ 而 b 意味着"改模型文件"，不是"改 vLLM 主干"

④ 验证
   正确性：先用 test_situ_real.py 对拍（算子层）
   端到端：同 prompt 下 vLLM 输出与 HF 参考一致（框架层）
   性能：TTFT / TPOT / 每层耗时（对照未替换版本）
```

**关键认知**：`SiTU-GLU` 是 KIMI3 的**模型私有算子**，它天然不属于 vLLM 的通用算子集。
所以它的接入方式必然是 **③b（模型文件里接）**，而不是"注册成 vLLM 的通用算子"。
**这也是"补缺算子"和"通用算子"的区别在工程上的体现。**

---

## 4. LatentMoE 的「图接线」具体改哪

属于第 ③ 层（图/模型层），改的是**模型定义**，不是算子：

```python
# 现状（vLLM/SGLang 里的 KIMI3 模型文件）
hidden = moe(hidden)

# 接线后
latent = self.down_proj(hidden)          # Linear  H → moe_hidden    ← 加一个投影
latent = moe(latent)                     # MoE（已有算子）
latent = self.rms_norm(latent)           # 可选
hidden = self.up_proj(latent)            # Linear  moe_hidden → H    ← 加一个投影
```

**用的全是现成算子**（Linear=GEMM、MoE、RMSNorm），所以：
- **不需要写新 kernel** ← 这就是 README 里说"它是图接线不是算子开发"的意思
- 改动量 = 在模型文件里加两个 Linear 和一个 norm 的定义与调用
- 风险在**形状对不对**，不在算子

---

## 5. SGLang 那边（对应关系）

概念一一对应，搬过去就行：

| vLLM | SGLang |
|---|---|
| `CustomOp.forward_npu` | `sglang/srt/layers/` 下的层实现（同样按 platform 分支） |
| `platforms/` | `sglang/srt/hardware_backend/` |
| Model Registry | `sglang/srt/models/` + `model_config` |

**所以"我的算子怎么进 SGLang"这个问题，答案的结构和 vLLM 完全一样**——
这本身就能在追问里说明"我理解的是机制，不是某个框架的 API"。

---

## 6. 话术（30 秒）

> 「算子写完只是第一层。完整链路是三层：**算子层**（AscendC kernel + aclnn，装到 opp vendors）、
> **框架适配层**（vLLM 的 `CustomOp` 有机会就补 `forward_npu`，没有挂载点就在模型文件里直接调）、
> **图/模型层**（LatentMoE 那种接线、以及要不要把算子融进 GEMM）。
>
> **「图级优化」不是一个开关**，具体就三件事：该融合的手写融合 kernel、
> 该捕获的开图模式（CUDA Graph / ACL Graph）、该复用的交给 bufferization。
>
> 而且**不用改 vLLM 主干**——昇腾的活都在 vllm-ascend 这层 plugin 里，
> 我的算子挂在它下面。这也是为什么 SiTU-GLU 的接入点在模型文件里，
> 而不是"注册成 vLLM 通用算子"——**它是模型私有算子，本来就不属于通用算子集**。」
