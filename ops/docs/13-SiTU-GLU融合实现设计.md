# SiTU-GLU 融合实现设计（GEMM + epilogue）

> 目标算子：`gate_up = x @ [W_gate; W_up]^T`（`[M, 2d]`）→ `out = situ(gate) · tanh_scale(up)`（`[M, d]`）
> 一次 GEMM + epilogue 完成，`gate_up` 这个中间张量**不落 HBM**。

---

## 0. 先说一个坑：**直接接 epilogue 是走不通的**

我一开始的想法是"GEMM 出 `[M, 2d]`，epilogue 顺手算 SiTU 就行"。**这个想法有个死结**：

```
Gate 在输出的前 d 列，Up 在后 d 列。
要算 out = situ(gate) × tanh(up)，必须**同时**拿到 gate 和 up 的对应列。

但 epilogue 是按 N 方向的 tile 顺序处理的：
  tile 0..42 全是 gate（BaseN=256, d=11008 → 43 块）
  tile 43..85 才是 up
→ 处理 tile 0 时算出的 situ(gate) 要一直存到 tile 43 才用得上，
  而那是 [M=128, d=11008] fp32 = 5.6 MB —— **UB 是 192 KB，差 30 倍**。
```

**所以不能按原始布局直接融。**

---

## 1. 解法：重排权重，让 gate / up **配对进同一个 tile**

把权重按 `subN` 为单位交错排列（**离线预处理一次**）：

```
原始  W [2d, H]：  [gate_0 .. gate_{d-1}] [up_0 .. up_{d-1}]

重排后 W' :  [gate_0, up_0, gate_1, up_1, ...]     ← 每个 "对" 是 subN 行
              ↑ subN 行  ↑ subN 行
```

即：

```
W'[2j·subN + i]     = W[j·subN + i]           # gate 片的第 j 块
W'[(2j+1)·subN + i] = W[d + j·subN + i]       # up   片的第 j 块
```

取 `BaseN = 2 × subN`，**第 j 个输出 tile 里恰好同时有 gate_j 和 up_j** —— 死结解开了。

epilogue 在 tile 内：

```
片 0（前 subN 列 = gate_j）: situ(gate)  → 暂存 UB
片 1（后 subN 列 = up_j）  : tanh_scale(up) → 与上一步的 situ 相乘
                                            → 写 out[:, j·subN : (j+1)·subN]
```

输出列顺序**天然正确**（第 j 个 tile 就对应 out 的第 j 片），不需要再重排回来。

### UB 占用（这个方案的代价）

```
同时要暂存：situ(gate_j) + tanh(up_j) = 2 × subN × BaseM × 4B
BaseM=128, subN=64  →  2 × 64 × 128 × 4 = 64 KB   ✅ 远小于 UB 192KB
```

### 前提条件

| 条件 | KIMI3 是否满足 |
|---|---|
| `d % subN == 0` | ✅ `11008 = 64 × 172`（也 = 128×86, 256×43） |
| 权重可离线重排 | ✅ 一次预处理，推理期不变 |
| `2d % BaseN == 0` | ✅ `22016 = 256 × 86` |

---

## 2. kernel 结构

```cpp
#include "kernel_operator.h"
#include "lib/matmul_intf.h"          // Matmul 高阶 API
#include "situ_glu_fused_tiling.h"

using namespace AscendC;
using AT = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;   // x  [M,H]
using BT = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;   // W' [2d,H]（重排后）
using CT = MatmulType<TPosition::GM, CubeFormat::ND, float>;        // 中间累加 fp32
using BiasT = MatmulType<TPosition::GM, CubeFormat::ND, float>;

class KernelSituGluFused {
public:
    Matmul<AT, BT, CT, BiasT> mm;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, SituGluFusedTilingData* t) {
        // ---- Matmul 侧：标准初始化，tiling 由 host 用 MatmulApiTiling 算好 ----
        mm.Init(&t->mmTiling);
        mm.SetTensorA(xGm);            // A = x
        mm.SetTensorB(wGm);            // B = W'（重排后的权重）

        // ---- epilogue 侧：UB 缓冲（fp32）----
        //   gBuf: situ(gate_j) 的结果
        //   uBuf: tanh_scale(up_j) 的结果
        //   各 subN × BaseM 个 fp32
        pipe.InitBuffer(gQueue, 1, t->subN * t->baseM * sizeof(float));
        pipe.InitBuffer(uQueue, 1, t->subN * t->baseM * sizeof(float));
        pipe.InitBuffer(tmpQueue, 1, t->subN * t->baseM * sizeof(float));
        pipe.InitBuffer(outQueue, 1, t->subN * t->baseM * sizeof(bfloat16_t));
    }

    __aicore__ inline void Process() {
        LocalTensor<float> cLocal;      // L0C 上的 fp32 结果 tile

        while (mm.Iterate()) {                       // ← 逐 tile 迭代
            mm.GetTensorC(cLocal);                   //   拿片上结果（不落 GM）

            Epid(0, cLocal);                         // 片 0：gate → situ
            Epid(1, cLocal);                         // 片 1：up   → tanh × situ → 写回

            mm.FreeTensor(cLocal);
        }
    }

private:
    // 处理 tile 内第 `which` 个 subN 片
    __aicore__ inline void Epid(int which, LocalTensor<float>& cLocal) {
        // cLocal 布局：[BaseM, BaseN]，BaseN = 2*subN
        // which=0 取前 subN 列（gate），which=1 取后 subN 列（up）
        // ↓ 具体取值方式随 Matmul API 版本不同，见 §4 风险表
        LocalTensor<float> src = cLocal[which * subN * baseM];   // 示意

        if (which == 0) {
            // ---- situ(gate) = beta·tanh(gate/beta)·sigmoid(gate) ----
            // 这一段**直接搬自已验证的 SituAndMul kernel**，数学完全一致
            LocalTensor<float> g = gQueue.AllocTensor<float>();
            Muls(tmp, src, invBeta, n);  Tanh(tmp, tmp, n);
            Sigmoid(g, src, n);          Mul(tmp, tmp, g, n);
            Muls(g, tmp, beta, n);                    // g = situ(gate)
            gQueue.EnQue(g);
        } else {
            // ---- tanh_scale(up) = linear_beta·tanh(up/linear_beta) ----
            LocalTensor<float> u = uQueue.AllocTensor<float>();
            Muls(u, src, invLinearBeta, n);  Tanh(u, u, n);
            Muls(u, u, linearBeta, n);
            // ---- out = situ(gate) × tanh_scale(up) ----
            LocalTensor<float> g = gQueue.DeQue<float>();
            Mul(u, u, g, n);
            LocalTensor<bfloat16_t> o = outQueue.AllocTensor<bfloat16_t>();
            Cast(o, u, RoundMode::CAST_ROUND, n);
            outQueue.EnQue(o);
            // 写回 GM：列偏移 = tile 序号 × subN
            LocalTensor<bfloat16_t> oR = outQueue.DeQue<bfloat16_t>();
            DataCopy(yGm[rowOff * d + colOff], oR, n);
            ...
        }
    }
};
```

**注意**：`situ` 和 `tanh_scale` 那两段向量链**与已开发并验证过的 `SituAndMul` kernel 完全一致**
（`Muls → Tanh → Sigmoid → Mul → Muls`），所以这部分是**复用已验证代码**，
真正的新增只有 Matmul 集成 + 分片逻辑。

---

## 3. tiling 参数（由 `fusion_tiling_calc.py` 算出）

```
BaseM = 128     (L0C: 128×256×4B = 128KB，占满)
BaseN = 256     (2d=22016 → 86 块)
BaseK = 128     (H=4096 → 32 次迭代)
subN  = 64      (UB 约束：2×64×128×4B = 64KB)
L1 ≈ 192 KB ✓（512KB 够）
```

```bash
python3 kimi3_ops/fusion_tiling_calc.py           # 重算/调整
```

---

## 4. ⚠️ 风险与未验证项（诚实清单）

| # | 风险 | 说明 | 怎么处理 |
|---|---|---|---|
| 1 | **Matmul API 用法** | `Iterate()` / `GetTensorC()` / `FreeTensor()` 的签名和 tile 内取子片的方式**随 CANN 版本有差异**，我写的这段是按常见形态，**必须在目标版本上核对** | 查本机 `$ASCEND_HOME_PATH/include/lib/matmul_intf.h` 和 Matmul 开发指南 |
| 2 | host 侧 tiling | 融合版的 tiling 不能手写，要用 `MatmulApiTiling` / `MultiCoreMatmulTiling` 生成 `TCubeTiling` | 参考 CANN 样例里的 matmul 自定义算子 |
| 3 | **权重重排** | 需要一个预处理步骤把 `[W_gate; W_up]` 交错过 | 写成离线脚本（host 侧 Python 或 C++），推理期不重复 |
| 4 | 原地运算 | `Mul(tmp, tmp, g)` 依赖 Level-2 原地支持（与现有 kernel 同一约定） | 不支持就加缓冲 |
| 5 | 精度 | epilogue 多算一次 Cast 往返，与独立算子版可能差 1 个 ulp | 用**现有 `test_situ_real.py` 的 golden 对拍**融合版 |
| 6 | 零验证 | 这份设计**一行都没编译过** | 上机第一件事是编译，不是调参 |

---

## 5. 上机验证顺序（和别的工作不冲突）

```
① 先过原版            cd kernels/SituAndMul && bash build.sh && python test_situ_real.py
                      ← 这一步和融合无关，但必须先过（否则基准不可信）
② 建工程骨架          msopgen gen -i situ_glu_fused_op.json -f aclnn -c ai_core-<SOC> -lan cpp
③ 填三个文件          tiling.h / op_host / op_kernel（本设计第 2、3 节）
④ 权重重排脚本        把 [W_gate; W_up] 交错过
⑤ 对拍              复用 test_situ_real.py 的 harness（golden 不变，算子换成融合版）
⑥ 量收益            对比 ① 的基线，看省下的 4d 有没有兑现成时间
```

**⑤ 是关键**：golden（`reference/situ.py`）不用改，因为数学一样；
只是被测对象从"独立算子 + 前置 GEMM"换成了"融合算子"。**这样收益是可比的。**

---

## 6. 一句话总结（表述用）

> 「融合我设计完了，但**第一版想法是错的**：我原以为 GEMM 出 `[M, 2d]` 后 epilogue 顺手算就行，
> 但 gate 在 tile 0..42、up 在 43..85，处理 gate 时算出的 situ 要暂存到 up tile 才用得上，
> 那是 `[128, 11008]` fp32 = 5.6 MB，UB 只有 192 KB —— **差 30 倍，存不下**。
>
> 解法是**重排权重布局**：把 `[W_gate; W_up]` 按 subN 交错成 `[gate_0, up_0, gate_1, up_1, ...]`，
> 取 `BaseN = 2×subN`，每个 tile 里就同时有配对的 gate 和 up。代价只是离线重排一次权重，
> UB 需求降到 64 KB。
>
> **这个坑比 epilogue 本身重要** —— 融合的难点从来不是"把算子接上去"，
> 而是"改数据布局让两边能对上"。」
