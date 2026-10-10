# GEMM：tiling 怎么定的 + 算力达标率（昇腾 A3）

> 解决的问题：`遇到的问题与解决方案.md §1.3` 判定「两个算子都是访存密集，问答必考的
> matmul/attention 无证据」，§1.2 深挖轮「tiling 怎么定的？瓶颈在哪？」答不了。
> 本文把 GEMM 的 tiling 决策**手推一遍**（每一步都能被追问），并给出真机达标率的测法。

---

## 1. 先判瓶颈：GEMM 是算力受限的（与 elementwise 相反）

```
FLOPs = 2·M·N·K          （乘 + 加各一次）
Bytes = 2·(M·K + K·N + M·N)   （读 A、读 B、写 C，bf16 = 2B）

AI = 2MNK / (2(MK + KN + MN))
```

当 M=N=K 时，`AI ≈ K/3`。K=4096 ⇒ **AI ≈ 1365 FLOP/B**，
远高于脊点（峰值算力/峰值带宽，量级数百）⇒ **算力受限**。

> 与 SiTU-GLU（AI≈3.7，访存受限）正好分居 roofline 两侧 ——
> 问答讲「两类算子怎么分别优化」时，这两个数字就是最干净的对照。

---

## 2. tiling 三层：容量约束决定参数，不是拍脑袋

昇腾 GEMM 的数据通路是三级搬运（每级都有硬容量上限）：

```
GM(HBM) ──MTE2──> L1(512KB) ──MTE1──> L0A/L0B(各 64KB) ──MAD──> L0C(128KB) ──> GM
                    ↑ 分块 stepM/stepN/stepKa/stepKb      ↑ BaseM/BaseN/BaseK     ↑ FIXP(向量)
```

| 层级 | 容量（910_93 量级，以官方文档为准） | 承载的参数 |
|---|---|---|
| L0A | 64 KB | `BaseM × BaseK × sizeof(dtype)` |
| L0B | 64 KB | `BaseK × BaseN × sizeof(dtype)` |
| L0C | 128 KB | `BaseM × BaseN × 4`（fp32 累加） |
| L1 | 512 KB | `(stepKa×stepM + stepKb×stepN) × sizeof(dtype)` |
| UB | 192 KB | FIXP 后处理（加 bias / 激活）的 buffer |

**推导（bf16）：**

```
L0A: BaseM × BaseK × 2B ≤ 64KB
L0B: BaseK × BaseN × 2B ≤ 64KB
L0C: BaseM × BaseN × 4B ≤ 128KB     ← 最紧的一条

取 BaseM = BaseN = 128  ⇒ L0C = 128×128×4 = 64 KB  ✅（还剩一半余量）
取 BaseK = 64           ⇒ L0A = 128×64×2  = 16 KB  ✅
                          L0B = 64×128×2  = 16 KB  ✅
```

⇒ **`BaseM=BaseN=128, BaseK=64` 就是这么定出来的**：先由最紧的 L0C 定 M/N，再由 L0A/L0B 定 K。
换成 fp16/int8 时，L0C 约束不变（仍是 fp32 累加），但 L0A/L0B 余量翻倍 ⇒ BaseK 可以加到 128。

**L1 分块与双缓冲：**

```
L1 需要同时放：A 的 stepKa 块 + B 的 stepKb 块
(stepKa×stepM + stepKb×stepN) × 2B ≤ 512KB

取 stepM=stepN=128, stepKa=stepKb=4（即 A/B 各预取 4 个 K 块）：
(4×128 + 4×128) × 2 = 2048 B... 这是每 K 元素的 L1 占用口径，实际按块算：
4 块 A(128×64×2=16KB) + 4 块 B(64×128×2=16KB) = 128 KB ✅ 远小于 512KB
⇒ 还可以开 depthA1/depthB1（L1 上的多缓冲）把 MTE2 与 MTE1 重叠起来
```

> **这就是「tiling 怎么定的」的完整答案**：不是试出来的，是在 L0C → L0A/L0B → L1
> 三条容量约束下**逐级反推**出来的；最后再用 depth 留出的余量做流水。

---

## 3. 流水：四级 + 双缓冲，目标是让 MTE2 / MTE1 / MAD 重叠

```
迭代时间 ≈ max(MTE2 时间, MTE1 时间, MAD 时间)   ← 流水做对了
迭代时间 ≈ MTE2 + MTE1 + MAD                     ← 流水没做对（等于串行）
```

对照：SiTU-GLU 改造前的 `TQue depth=1` 就是后者。GEMM 上对应 `depthA1/depthB1 = 1`。
判定方法与 `docs/04` 一样：`msprof` 看 MTE2/MTE1/MAC/VEC 的时间是否相加。

---

## 4. 达标率（MFU）：GEMM 的「快多少」怎么量化

```
实测 FLOPS = 2·M·N·K / 实测时间
MFU(算力达标率) = 实测 FLOPS / 峰值 bf16 算力
理论下限时间 T_min = 2·M·N·K / 峰值算力
```

| MFU | 判定 | 典型原因 |
|---|---|---|
| ≥ 70% | 显式 GEMM 到位 | —— |
| 40 ~ 70% | 正常范围，看 shape | M/N/K 某维太小、K 方向分块紧、尾块浪费 |
| < 40% | 有问题 | tiling 不合理、流水没重叠、L2 命中差、尾部（fixpipe）没重叠 |

**shape 依赖必须说清楚**（高频追问）：同一个 kernel 在 K 小、M×N 大时 MFU 会掉；
decode 阶段的 GEMM 是 `M=1` 的 GEMV（**必然访存受限**，要换评判口径，不能拿 MFU 说事）。

---

## 5. 怎么测（`bench_gemm_real.py`）

```bash
python3 bench_gemm_real.py                                   # 默认几组 shape
python3 bench_gemm_real.py --bench-peak-only                 # 先测本机 bf16 GEMM 峰值算力
python3 bench_gemm_real.py --peak-tflops 400                 # 用实测峰值算 MFU
python3 bench_gemm_real.py --m 4096 --n 4096 --k 4096
```

脚本报三件事：`torch.matmul`（底层就是 `aclnnMatmul`）延时、实测 TFLOPS、MFU；
并按 `M=N=K` 扫描 K，画出「MFU 随 K 上升」的趋势 —— 用来说明「大 K 才吃满算力」。

> 「与 `aclnnMatmul` 对比」的准确说法：`torch.matmul` 在 NPU 上就是走 `aclnnMatmul`，
> 所以它是**基线**而不是对手。要证明自己的 kernel 有优势，得比的是
> 「融合后的算子」（如 `SiTU-GLU` 把 GEMM 的 epilogue 吃进去）——那才是昇腾生态里真正缺的活。

---

## 6. 常见不达标原因（真机排查顺序）

1. **M 或 N 太小**（< 128）：Cube 一个 Base 块都填不满，先做算子融合把多个小 GEMM 拼大。
2. **K 太小**：L1 的 K 方向分块 `stepKa` 打不满，MTE1 与 MAD 无法重叠 → 改 `depthKa` 或换 split-K。
3. **尾块**：`M % 128 != 0` 时最后一块算力被浪费；pad 到对齐或在 tiling 里单独处理尾块。
4. **L2 命中差**：大 K 时 B 矩阵被反复读；用 `M 方向分块 + 保持 A 驻留 L1` 提高复用率。
5. **FIXP 没重叠**：`bias + 激活` 的后处理如果是独立 kernel，就把带宽又赔回去了 —— 正解是 epilogue 融合。

---

## 7. 与「promotion 到其它芯片」的关系（叙事用）

tiling 的三个约束（L0C → L0A/L0B → L1）在别的加速器上是同一件事，只是名字不同：

| 昇腾 | NVIDIA | 国产 GPU（概念对应） |
|---|---|---|
| L1 | shared memory | 片上 shared buffer |
| L0A/L0B | register file（fragment A/B） | 计算单元本地存储 |
| L0C | accumulator（FP32） | 累加器 |
| `BaseM/N/K` | `CUTLASS` 的 `TileShape` | 同样的分块语义 |

> ⇒ 「tiling 怎么定的」的答案，换硬件只需要换容量常数，**推导方式一模一样**。
> 这就是 §1.4「换硬件只换 ISA，不换方法」最硬的证据。

---

> ⚠️ 本文的容量常数（L0A/L0B 64KB、L0C 128KB、L1 512KB、UB 192KB）为 910_93 量级的
> 工程常用值，**上机前请以官方文档/`platform_ascendc` 查询结果为准**（host 端可用
> `PlatformAscendC::GetCoreMemSize(CoreMemType::L1/L0A/L0B/L0C/UB)` 打印真实容量）。
