# vLLM / SGLang / llama.cpp 在这台机器上怎么跑（含 PD 分离）

> 环境实测：**1 个 Docker 容器 / 2 张卡（davinci4、davinci5）/ 40 核配额 / 240 GiB 内存 /
> SOC ascend910_93 / CANN 9.0.0 / 驱动 25.5.0 / **无 RDMA 设备** / 无 torch、无 vLLM。**
>
> 所以本文分两段：**现在能做什么**（第 1 节）和 **补齐依赖后的完整路径**（第 2~4 节）。

---

## 1. 当前环境能跑到哪一步（先别白费力气）

| 实验 | 现在能做 | 卡在哪 |
|---|---|---|
| `npu-smi info` / 设备账 | ❌ | 容器无执行权限（本次会话） |
| torch_npu 基础可用性 | ❌ | **没装 torch / torch_npu**（`/usr/lib/python3/dist-packages` 只有 numpy 1.21.5） |
| vLLM-Ascend 单卡跑小模型 | ❌ | 上面 + 没装 vllm-ascend + 没模型权重 |
| 2 卡 TP（验证 HCCL） | ❌ | 上面全部（设备节点倒是齐的，见 `docs/07`） |
| **同机 PD 分离**（1 卡 P + 1 卡 D） | ⚠️ 理论上可以 | 上面全部；传输后端要选不依赖 RDMA 的 |
| **跨节点 PD 分离** | ❌ **物理不具备** | 只有 1 个节点，且 `/sys/class/infiniband` 不存在 |

> `pip` 在这台机器上是能用的（`pip 22.0.2` 在 dist-packages 里），所以补齐依赖是**可能的**，
> 前提是能访问 PyPI / 华为源。

---

## 2. 补齐依赖的路径（顺序别乱，每步都有验收点）

```bash
# 0) 先固化设备账，后面所有配置都要用到这个数
npu-smi info
echo "可见卡: $ASCEND_RT_VISIBLE_DEVICES"      # 空 = 全部可见

# 1) CANN 环境（已装，只要 source）
source /usr/local/Ascend/cann-9.0.0/set_env.sh

# 2) torch + torch_npu —— 版本必须和 CANN 9.0.0 配套！
#    别自己猜版本号：查 CANN 9.0.0 对应的 torch_npu 版本表（官方《Ascend Extension for PyTorch》）
pip install torch==<与torch_npu配套的版本>
pip install torch_npu==<对应CANN 9.0.0的版本>

# 验收：必须返回 True 和 2
python3 -c "import torch, torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"

# 3) vLLM-Ascend —— 同样要和 CANN/torch_npu 三者配套
pip install vllm-ascend==<对应版本>

# 4) 拿一个**小**模型做最小验证（别一上来下 7B/72B，先验证链路）
#    Qwen2.5-0.5B / TinyLlama-1.1B 这类，权重几百 MB
```

**为什么强调"三者配套"**：`vllm-ascend ↔ torch_npu ↔ CANN` 是强耦合的，
版本错一个就会出现"能 import 但算子找不到"这种最难查的错误。
查版本表的入口是 vLLM-Ascend 官方文档的 Installation / 版本对应章节。

---

## 3. 推理框架三条路线

### 3.1 vLLM-Ascend（服务化，PD 分离主力）

```bash
# ① 单卡最小验证（先确认能出 token，再谈优化）
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --trust-remote-code --max-model-len 4096 \
  --gpu-memory-utilization 0.9

# ② 2 卡 TP（这一步顺带把 HCCL 验了）
vllm serve <模型> --tensor-parallel-size 2

# ③ 同机 PD 分离：P 用卡 0、D 用卡 1（kv_port 别冲突、engine_id 必须唯一）
#    P 端
ASCEND_RT_VISIBLE_DEVICES=0 vllm serve <模型> --port 8004 \
  --tensor-parallel-size 1 \
  --kv-transfer-config '{"kv_connector":"MooncakeLayerwiseConnector",
    "kv_role":"kv_producer","kv_port":"30000","engine_id":"0",
    "kv_connector_module_path":"vllm_ascend.distributed.mooncake_layerwise_connector",
    "kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},
                                 "decode":{"dp_size":1,"tp_size":1}}}'
#    D 端：ASCEND_RT_VISIBLE_DEVICES=1，kv_role 改 kv_consumer，kv_port 改 30200，engine_id 改 2
#    proxy
python load_balance_proxy_layerwise_server_example.py --prefiller-hosts <P_IP> --decoder-hosts <D_IP>
```

> ⚠️ **同机 PD 分离的传输后端是关键**：`Mooncake*Connector` 的设计目标是 RDMA，
> 本容器没有 IB 设备。可行路线是先把 KV 传输退化成可用的后端
> （如 `AscendStoreConnector` 走 KV Pool / 或 Mooncake 的 TCP 模式），
> **能验证"分离的流程和参数"，但验证不了"网络传输性能"**。
> 这一点必须在追问里主动说清，否则被问"你测的 KV 传输多少 GB/s"会很难看。

### 3.2 SGLang（PD 分离另一条路）

```bash
# Prefill
python -m sglang.launch_server --model <模型> --port 30000 \
  --disaggregation-mode prefill --disaggregation-transfer-backend ascend \
  --disaggregation-bootstrap-port 8998
export ASCEND_MF_STORE_URL="tcp://<PREFILL_HOST_IP>:24669"

# Decode：--disaggregation-mode decode（同 backend / 同 bootstrap 端口）
# Router
sglang_router.launch_router --pd-disaggregation --prefill <P_URL> --decode <D_URL>
```

`--disaggregation-transfer-backend` 可选 `mooncake`（默认）/ `nixl` / **`ascend`** / `fake`。
走 `ascend` 需要 `pip install memfabric-hybrid`；走 mooncake 则设 `ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE=true`。
**调试阶段先用 `fake`**（不真传，只验证调度和建连）——这是个很实用的排查技巧。

### 3.3 llama.cpp（单机 / 边缘，定位完全不同）

```bash
# 编译要显式打开 CANN 后端（选项名以官方 CANN 后端文档为准）
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cmake -B build -DGGML_CANN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# 跑：关键是 -ngl 必须放满
./build/bin/llama-cli -m model.gguf -ngl 99 -c 4096 -p "你好"
```

---

## 4. llama.cpp 在国产 NPU 上的注意点（你问的重点）

| # | 注意点 | 为什么 |
|---|---|---|
| 1 | **`-ngl`（offload 层数）必须放满，否则静默变慢** | 放不满 = 一部分层跑在 CPU，性能断崖式下跌，而且**不报错**——这是最常见的"怎么这么慢"的原因 |
| 2 | **GGUF 的 k-quant 不是 NPU 的甜点** | Q4_K/Q5_K 这类量化是**为 CPU 的 dequant 成本设计的**；NPU 擅长 INT8/FP8/W8A8。拿 Q4_K_M 跑 NPU，反量化开销可能吃掉收益 |
| 3 | **KV cache 量化要先确认后端支持** | `--cache-type-k/-v` 在某些后端是静默回退或不支持，别默认它生效了 |
| 4 | **llama.cpp 没有 PD 分离，也没有真正的 continuous batching** | 它是 slot-based 并发、单机定位。**拿它对比 vLLM 的吞吐是不公平的**——说明时要说清这是两个场景 |
| 5 | **它的优势是"能跑起来"** | 单用户、低并发、边缘、CPU/异构、量化部署、数据不出内网——这正是你在 S3000 上做 MUSA 移植的场景 |
| 6 | **调优入口很窄** | 基本只有 `-ngl / -c / -b / -ub / -t`，没有 tiling、没有 prefix caching。所以"llama.cpp 调优"和"算子级调优"是两件事 |

### 怎么答（结合你的 MUSA 经历）

> 「llama.cpp 我做的是**算子层移植**（把 ggml 的核心算子用 MUSA 重写），
> 所以我知道它的性能瓶颈不在框架层，而在 **ggml 那几十个 kernel 的访存效率**上。
> 注意它是**单机/边缘定位**：没有 PD 分离、没有 continuous batching，
> 所以我不会拿它的吞吐跟 vLLM 比——那不是一个场景。
> 它的价值是"低并发、量化、数据不出内网"，这也是当时那个项目（深圳鹏锐）的需求。」

**这段回答的结构**：定位差异 → 我知道瓶颈在哪 → 主动划清可比性。
比"我 llama.cpp 很快"强十倍。

---

## 5. 通用坑清单（昇腾上跑推理框架）

| 坑 | 现象 | 处理 |
|---|---|---|
| 版本不配套 | import 成功但算子找不到 / 段错误 | vllm-ascend ↔ torch_npu ↔ CANN 三者查表对齐 |
| 选卡没生效 | 用了错误的卡 | `ASCEND_RT_VISIBLE_DEVICES`（**注意是 RT**，不是 CUDA 那个） |
| 多卡建链失败 | HCCL 初始化卡住/超时 | `HCCL_SOCKET_IFNAME` 指定网卡；`HCCL_RDMA_TIMEOUT` 调大 |
| 显存不够 | OOM | `--gpu-memory-utilization`、`--max-model-len` 先降下来 |
| 首次编译慢 | 跑起来卡几分钟 | vLLM 的 custom op JIT，属正常；别误判成死锁 |
| 图模式问题 | 结果不对或崩 | 先 `--enforce-eager` 关图模式，跑通再开 ACL Graph |
| **容器 CPU 配额** | 多线程起不来 | `nproc` 返回 640 但配额只有 **40 核**，线程数要按配额设（见本文开头环境实测） |

---

## 6. 给这台机器的"最小可行实验路径"

按这个顺序走，每步都有明确验收点，不要跳：

```bash
① npu-smi info                          → 验收：看到 2 个 device、型号、HBM 容量
② source /usr/local/Ascend/cann-9.0.0/set_env.sh && python3 -c "import torch, torch_npu; print(torch.npu.device_count())"
                                        → 验收：输出 2
③ vllm serve Qwen2.5-0.5B --max-model-len 4096
                                        → 验收：能 curl 出 token（先证明链路通）
④ vllm serve <7B> --tensor-parallel-size 2
                                        → 验收：跑通 = HCCL 通信链路实测通过（补上 docs/07 的缺口）
⑤ 同机 PD 分离（1P+1D，先 fake/降级传输后端）
                                        → 验收：分离流程与参数正确；**不产出网络性能数字**
```

**③④ 是这台机器真正能新增的证据**：③ 证明推理链路通，④ 证明 HCCL 两卡通信通
（这正好把 `kimi3_ops/docs/07` 里"单机 2 卡能验证什么"从"文档"变成"实测"）。
⑤ 只能验证流程，性能结论必须等真多节点。
