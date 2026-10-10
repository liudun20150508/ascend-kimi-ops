#!/usr/bin/env python3
"""单机 2-die HCCL 通信实测（补「并行/通信层」的证据）。

为什么需要它
------------
四层体系里，**并行/通信层**的判据是「通信量 ÷ 互联带宽」和「能不能避免跨 rank 交换」。
这两个判据都要**互联带宽的实测值**才有意义（K3 方案里我们用的 A3 D2D 784 GB/s 是厂商标称）。

⚠️ 边界（必须说清，别把结论说过头）
--------------------------------
- 本机是**一张物理卡的 2 个 die**（`npu-smi` 显示 NPU 2 下 Chip 0/1），
  **不是两张卡、更不是跨机** ⇒ 测到的是 **HCCS 片内互联**，不能代表跨节点 RoCE。
- 跨机带宽要 `ib_write_bw` / `hccn_tool -ping`（见 `docs/10` 的 L1 阶梯），本机做不了。
- 本机**没有 mpirun**，所以走 torch_npu 的 HCCL（`init_process_group('hccl')`）单机 2 进程。

判据（跑之前就写好）
------------------
1. 小包看**时延**（allreduce 8 KiB 应 < 100 μs 量级），大包看**总线带宽 busbw**；
2. 大包 busbw 应与该互联的标称带宽同量级（同量级 = 通路正常；差一个数量级 = 配置问题）；
3. 两边 rank 数值一致（allreduce 后逐元素相等）—— 通信正确性。

用法
----
    /usr/local/python3.11.15/bin/python3 scripts/hccl_2die_check.py
    /usr/local/python3.11.15/bin/python3 scripts/hccl_2die_check.py --iters 50
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch                      # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.multiprocessing as mp  # noqa: E402
import torch_npu                  # noqa: E402,F401

DEFAULT_SIZES = [8 * 1024, 64 * 1024, 1024 ** 2, 8 * 1024 ** 2, 64 * 1024 ** 2]


def worker(rank, world_size, sizes, iters, outdir):
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29517')
    torch.npu.set_device(rank)
    dist.init_process_group(backend='hccl', rank=rank, world_size=world_size)

    # 正确性前置：allreduce(sum) 的结果必须两端一致、且等于 world_size × 原值
    probe = torch.arange(8, dtype=torch.float32).npu() + 1.0
    dist.all_reduce(probe)
    torch.npu.synchronize()
    ok = torch.allclose(probe.cpu(), (torch.arange(8, dtype=torch.float32) + 1.0) * world_size, rtol=1e-5)
    if rank == 0:
        print(f"[正确性] allreduce(sum) 校验：{'✅ 通过' if ok else '❌ 不一致'}")

    res = []
    for nbytes in sizes:
        n = nbytes // 4  # fp32
        t = torch.ones(n, dtype=torch.float32).npu()
        for _ in range(3):
            dist.all_reduce(t)
        torch.npu.synchronize()
        st = torch.npu.Event(enable_timing=True)
        en = torch.npu.Event(enable_timing=True)
        st.record()
        for _ in range(iters):
            dist.all_reduce(t)
        en.record()
        torch.npu.synchronize()
        ms = st.elapsed_time(en) / iters
        if rank == 0:
            algbw = nbytes / (ms * 1e-3) / 1e9
            busbw = algbw * 2 * (world_size - 1) / world_size   # ring allreduce 的换算
            res.append({'bytes': nbytes, 'ms': ms, 'algbw_GBps': algbw, 'busbw_GBps': busbw})
            print(f"  allreduce {nbytes / 1024:9.1f} KiB  {ms:9.4f} ms  "
                  f"algbw {algbw:8.1f} GB/s  busbw {busbw:8.1f} GB/s")
    if rank == 0:
        os.makedirs(outdir, exist_ok=True)
        json.dump(res, open(os.path.join(outdir, 'hccl_2die.json'), 'w'), indent=1)
    dist.barrier()
    dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--world', type=int, default=2)
    ap.add_argument('--outdir', default=os.path.join(os.path.dirname(HERE), 'verify_logs'))
    args = ap.parse_args()

    n = torch.npu.device_count()
    if n < 2:
        print(f'[SKIP] 需要 ≥2 个 device，本机只有 {n} 个')
        return 0
    world = min(args.world, n)
    print(f'device: {torch.npu.get_device_name(0)} × {world}（★ 同一张卡的 {world} 个 die，不是跨卡/跨机）')
    print(f'allreduce 各尺寸，{args.iters} 次取平均：\n')
    mp.spawn(worker, args=(world, DEFAULT_SIZES, args.iters, args.outdir), nprocs=world, join=True)
    print('\n判据：小包看时延、大包看 busbw（应与互联标称同量级）；结果 JSON 落在 outdir')
    print('⚠️ 这是 HCCS 片内互联口径，**不能**用来代表跨节点 RoCE（那要 hccn_tool / ib_write_bw）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
