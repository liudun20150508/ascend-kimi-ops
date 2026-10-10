# HCCL：单机能不能模拟多节点？（结论 + 可执行方案）

> 背景：`遇到的问题与解决方案.md §3.6` 把「HCCL 集合通信实验」列为缺口，理由是「需 ≥2 卡」。
> 现在这台机器上有 2 张卡（容器内 `/dev/davinci4`、`davinci5`），所以先问清楚：
> **单机 2 卡能替代多节点实验到什么程度？**

---

## 0. 一句话结论

**能验证「通信对不对 + 参数写没写对」，不能验证任何「网络好不好」的结论。**

分界线在数据面：单机内 HCCL 的 data path 走 **HCCS/SDMA**（片间高速链路，量级几百 GB/s），
**根本不经过网卡**。所以单机跑出来的带宽、延迟、重叠收益，都不能外推到多机。

---

## 1. 本机实测条件（决定能做到哪一步）

| 条件 | 实测 | 影响 |
|---|---|---|
| HCCL 必需设备节点 | ✅ `/dev/davinci4`、`/dev/davinci5`、`davinci_manager`、`devmm_svm`、`hisi_hdc` | 四件套齐 → HCCL 有运行条件 |
| IPC 共享内存 | ✅ `/dev/shm` 存在 | rank 间建链需要 |
| HCCL 测试工具 | ✅ `/usr/local/Ascend/cann-9.0.0/tools/hccl_test`（含 `hostfile`、`opbase_test/`） | 路线二可用 |
| CANN | ✅ 9.0.0 | —— |
| python 栈 | ❌ 无 torch / torch_npu（只有 numpy 1.21.5） | 路线一需先装 |
| RDMA 设备 | ❌ `/sys/class/infiniband` 不存在，只有 `eth0` | **RoCE 数据面做不了** |
| MPI | ❓ 未确认 | 路线二需 `mpirun` |

---

## 2. 三条路线，各自的天花板

### 路线一：`torch.distributed` + HCCL（推荐，最像真实用法）

```bash
# 单机多卡：2 个进程各绑一张卡
export ASCEND_RT_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 kimi3_ops/scripts/hccl_2card_sim.py --size-mb 64

# 伪多节点：两个终端各起 1 进程、各绑 1 张卡，控制面走 socket
# 终端1
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 \
  ASCEND_RT_VISIBLE_DEVICES=0 python3 kimi3_ops/scripts/hccl_2card_sim.py --no-spawn
# 终端2
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 RANK=1 WORLD_SIZE=2 LOCAL_RANK=0 \
  ASCEND_RT_VISIBLE_DEVICES=1 python3 kimi3_ops/scripts/hccl_2card_sim.py --no-spawn
```

前置：`pip install torch==2.10.0 torch_npu==2.10.0`。
脚本会打印：world_size、建链参数、AllReduce 正确性、busbw、以及**异步分块 AllReduce + 并行计算**的重叠做法演示。

### 路线二：CANN 自带 `hccl_test`（不用装 torch，但要 MPI + 编译）

```bash
cd /usr/local/Ascend/cann-9.0.0/tools/hccl_test
make MPI_HOME=/path/to/mpi ASCEND_DIR=$ASCEND_HOME_PATH

# 单节点、2 卡
mpirun -n 2 ./bin/all_reduce_test -b 8K -e 64M -f 2 -d fp32 -o sum -p 2

# 多节点（两节点为例）：hostfile 写 每节点IP:进程数
mpirun -f hostfile -n 16 ./bin/all_reduce_test -b 8K -e 64M -f 2 -d fp32 -o sum -p 8
```

> 注意 `-p` 是**每节点**的 NPU 数。多机时必须先设 `HCCL_SOCKET_IFNAME` 指向 RoCE 网卡
> （`export HCCL_SOCKET_IFNAME==eth0` 或被 `^=` 排除法），否则 HCCL 选错网卡直接建链失败。

### 路线三：SoftRoCE（把 `eth0` 变成 RDMA 设备）

```bash
sudo modprobe rdma_rxe
sudo rdma link add rxe0 type rxe netdev eth0
ibv_devices                       # 应看到 rxe0
```

天花板：**只能验证 verbs 编程模型**（`../../rdma_host_demo/`），
带宽走软件栈，无意义。
现实障碍：需要容器有 `CAP_SYS_MODULE` + `NET_ADMIN`（`--privileged` 或对应 cap），
本容器没有（`/sys/class/infiniband` 为空就是证据）。**这条路线在当前环境走不通。**

---

## 3. 能 / 不能验证什么（这张表是回答对方的关键）

| 想验证的东西 | 单机 2 卡能吗 | 说明 |
|---|---|---|
| HCCL 集合通信**正确性**（AllReduce/AllGather/AllToAll/ReduceScatter） | ✅ | 与网络无关，纯语义 |
| rank 表 / socket 建链 / `HCCL_SOCKET_IFNAME` 写对没有 | ✅ | 控制面走 socket，伪多节点即可验证 |
| 通信-计算**重叠的做法**（异步分块 + 等待） | ✅ | 能确认「真的重叠了」 |
| 重叠的**收益有多大** | ❌ | 单机 HCCS 太快，网络影响被彻底掩盖 |
| AllReduce 的 busbw 数字 | ❌ | 是 HCCS 带宽，不是网卡带宽 |
| Ring vs Tree 的**选择逻辑** | 🟡 | 能看到 HCCL 选了什么，但选优结论要真机 |
| RoCE 数据面带宽/延迟 | ❌ | 无 IB 设备 |
| PFC / ECN / DCQCN 拥塞控制 | ❌ | 无网卡 |
| **跨节点 PD 分离的 KV 单边传输** | ❌ | 单节点 + 无网卡，缺的正是那一步 |
| EP64 / 大集群负载均衡 | ❌ | 只有 2 卡 |

---

## 4. 最容易骗自己的三个地方（问答被追问时的防线）

1. **拿单机 busbw 当网络带宽**：HCCS 是片间链路，和 RoCE 的 400Gbps（≈50 GB/s）不是一个量级，
   拿它去算「KV 传输占端到端比例」会得出完全相反的结论。
2. **拿单机重叠收益当多机收益**：单机通信太快，重叠几乎必然"看起来很划算"；
   多机里通信才是瓶颈，重叠能省多少要看 `通信时间 / 计算时间` 的比值。
3. **拿"伪多节点"当多机**：进程分开、走 socket 建链，但数据面仍在同一台机器内。
   它能证明"rank 表写对了"，不能证明"网络能扛住"。

> 一句话：**单机实验证明「我会配」，真机实验才能证明「我配得好不好」。**

---

## 5. 什么时候必须上真机（当前环境的硬缺口）

跨节点实验需要同时满足：

1. **≥2 个物理节点**（当前：1 个 Docker 容器，`/proc/1/cgroup` = `docker/2c68c0f7...`）
2. **RoCE 网卡可见**（当前：只有 `eth0`，`/sys/class/infiniband` 不存在）
3. **`/etc/hccn.conf` 有 NPU 与网卡绑定**（当前：文件不存在）

三条一条都不满足 → **PD 分离、HCCL 多机、RoCE 拥塞控制这三个实验在本文环境无法做**，
只能做单机 2 卡那一层。
