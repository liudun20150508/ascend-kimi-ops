"""单机 2 卡跑 HCCL 集合通信 —— 能替代「多节点实验」到什么程度。

定位（先看这句，再决定要不要跑）：
    能验证：HCCL 通信正确性、rank 表/建链流程、参数是否写对、通信-计算重叠的**做法**
    不能验证：RoCE 数据面的带宽/延迟、PFC/ECN/DCQCN、跨节点 PD 分离的 KV 单边传输
    原因：单机内 HCCL 数据面走 HCCS/SDMA（片间高速链路，量级几百 GB/s），
          根本不经过网卡。把单机数字外推到多机是最典型的自欺欺人。

用法（二选一）：

  A. 「单机多卡」—— 2 个进程各绑一张卡，最接近真实用法
     export ASCEND_RT_VISIBLE_DEVICES=0,1
     torchrun --nproc_per_node=2 hccl_2card_sim.py --size-mb 64

  B. 「伪多节点」—— 2 个进程当作 2 个"节点"，各自 1 张卡，走 socket 建链
     # 终端1
     MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 RANK=0 WORLD_SIZE=2 LOCAL_RANK=0 \
       ASCEND_RT_VISIBLE_DEVICES=0 python3 hccl_2card_sim.py --no-spawn
     # 终端2
     MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 RANK=1 WORLD_SIZE=2 LOCAL_RANK=0 \
       ASCEND_RT_VISIBLE_DEVICES=1 python3 hccl_2card_sim.py --no-spawn
     # 想更像多机：给 eth0 加别名 IP（需 root 或 --cap-add NET_ADMIN），
     # 两个进程分别用不同 MASTER_ADDR/HCCL_IF_IP，验证的是「控制面能连上」，
     # 数据面依然走 HCCS —— 这一点不要自欺。

前置依赖（本机目前缺）：
    torch + torch_npu 未安装（/usr/lib/python3/dist-packages 里只有 numpy）。
    装法：pip install torch==2.10.0 torch_npu==2.10.0（CPU 版 torch + NPU 插件）
    不想装 torch 就走 CANN 自带工具，见 docs/07 的「路线二：hccl_test」。
"""

import argparse
import os
import sys
import time

import torch
import torch_npu  # noqa: F401  （import 即注册 NPU 后端）
import torch.distributed as dist


def human(nbytes):
    for u in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024 or u == "GiB":
            return f"{nbytes:.1f} {u}"
        nbytes /= 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-mb", type=float, default=64.0, help="AllReduce 张量大小 MiB")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-spawn", action="store_true",
                    help="不由 torchrun 启动（伪多节点模式，读 RANK/WORLD_SIZE 环境变量）")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # 绑卡：容器里可见的设备是 davinci4/5，逻辑编号仍从 0 开始
    torch.npu.set_device(local_rank)
    dev_name = torch.npu.get_device_name(local_rank)

    # 建链：这一步就是"rank 表 + socket 建链"，多机时走 HCCL_SOCKET_IFNAME 指定的网卡
    dist.init_process_group(backend="hccl", rank=rank, world_size=world)
    if rank == 0:
        print(f"[env] world_size={world} backend=hccl "
              f"master={os.environ.get('MASTER_ADDR', '?')}:{os.environ.get('MASTER_PORT', '?')}")
        print(f"[env] HCCL_SOCKET_IFNAME={os.environ.get('HCCL_SOCKET_IFNAME', '<未设置>')}  "
              f"HCCL_IF_IP={os.environ.get('HCCL_IF_IP', '<未设置>')}")
    print(f"  rank {rank}/{world}: 绑定 {dev_name} (local_rank={local_rank})")

    # ---------------- 正确性：AllReduce(sum) ----------------
    n = int(args.size_mb * 1024 * 1024 / 4)  # fp32
    x = torch.full((n,), float(rank + 1), dtype=torch.float32).npu()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    expect = float(world * (world + 1) // 2)
    ok = bool(torch.allclose(x, torch.full_like(x, expect)))
    if rank == 0:
        print(f"\n[正确性] AllReduce sum over {world} ranks: "
              f"期望 {expect} → {'PASS ✅' if ok else 'FAIL ❌'}")

    # ---------------- 性能：算 busbw（算法带宽） ----------------
    for _ in range(args.warmup):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / args.iters * 1e3

    nbytes = n * 4
    # Ring AllReduce 的 busbw 口径：每卡实际搬运 = 2*(N-1)/N * 数据量
    busbw = (2.0 * (world - 1) / world) * nbytes / (ms * 1e-3) / 1e9
    if rank == 0:
        print(f"[性能] AllReduce {human(nbytes)} 用时 {ms:.3f} ms")
        print(f"       算法带宽(busbw) ≈ {busbw:.1f} GB/s   ← 单机 HCCS，**不是网卡数字**")

    # ---------------- 通信-计算重叠的做法演示 ----------------
    # 真机多机优化的核心手法就是「边算边传」：把 AllReduce 切成 chunk 挂到计算流后面。
    # 单机也能验证做法对不对（是否真的重叠），但收益数字不可外推。
    chunk = max(1, n // 4)
    y = torch.zeros_like(x)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    handles = []
    for i in range(0, n, chunk):
        handles.append(dist.all_reduce(x[i:i + chunk], async_op=True))
    for _ in range(8):  # 假装有一份等量的计算
        y = y * 1.0000001 + 0.0
    for h in handles:
        h.wait()
    torch.npu.synchronize()
    ms_overlap = (time.perf_counter() - t0) * 1e3
    if rank == 0:
        print(f"[重叠] 分 4 块异步 AllReduce + 并行计算，总耗时 {ms_overlap:.3f} ms")
        print("       ← 做法可验证；收益大小单机不可信（HCCS 太快，网络影响被掩盖）")

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("\n下一步（要拿真数字必须做的）：")
        print("  1) 两个物理节点，各自配好 HCCL_SOCKET_IFNAME 指向 RoCE 网卡")
        print("  2) hostfile 写两个节点 IP:每节点进程数，用 mpirun -f hostfile 跑")
        print("  3) 把这里的 busbw 换成 RoCE 的实测值，再算「通信占端到端比例」")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
