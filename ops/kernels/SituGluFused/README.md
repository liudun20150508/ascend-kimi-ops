# SituGluFused —— GEMM + SiTU-GLU 融合算子（**已按官方 MIX 样例重建；上机对拍 PASS，通路验证版**）

> 用途：把 `gate_up = x @ [W_gate; W_up]^T` 和 `SiTU-GLU` 合成一个算子，
> 设计推导见 [`../../docs/13-SiTU-GLU融合实现设计.md`](../../docs/13-SiTU-GLU融合实现设计.md)，
> 收益量化见 [`../../benefit_quant.py`](../../benefit_quant.py)。

## ⚠️ 状态（截至 2026-09-19；详见 [`../../docs/14-上机实测记录.md`](../../docs/14-上机实测记录.md) §11、§13、§14.4、§16）

| 项 | 结果 |
|---|---|
| 代码状态 | **已按官方 MIX 样例骨架重写**，旧版「在 msopgen 旧模板上打补丁」的路线已废弃 |
| 编译 | ✅ **`EXIT=0`**（2026-09-19 实测；修掉 `Min()` 对 `uint32_t` 无重载的问题） |
| 安装 | ✅ 成功（`aclnn_situ_glu_fused.h` 落地） |
| **数值对拍** | ✅ **4 组 shape 全 PASS**，含真实尺寸 `M=128 H=1024 d=11008`（**0 / 1,409,024** 超容差，max_rel **3.890e-03**）→ `docs/14` §16.4 |
| 重复执行（bench） | ⛔ 第 2 次起报 `507057`；**空 kernel 也复现 ⇒ 与环境/框架层有关**，不影响数值结论（`docs/14` §16.4.1） |
| 历史已修缺陷（7 条） | workspace=NULL / cTile 未分配 UB / C 的 `TPosition` 声明错 / UB 未预留 / `coreM` 无符号下溢 / 缺 MIX 声明 / 缺 `REGIST_MATMUL_OBJ`（`docs/14` §13） |

**上机结果（2026-09-19，`../../scripts/verify_fused_rebuild.sh` 一键跑完）**：编译 `EXIT=0` → 安装 ✅ → 对拍
**4 组 shape 数值全 PASS**（真实尺寸 `M=128 H=1024 d=11008`：max_rel **3.890e-03**、**0 / 1,409,024** 超容差）。
上机又暴露并修掉 3 个真缺陷（`Min()` 无 `uint32_t` 重载 / Matmul 必须由 AIV 发起 / `B` 需 `isTrans=true`）；
⛔ 重复执行报 `507057`（**空 kernel 也复现** ⇒ 环境/框架层，不影响数值结论）；
⏸ 阶段 2（C 取 UB）三种组合都未通、已回退 ⇒ **本版是通路验证版，无 HBM 收益**。
一键重跑：`bash ../../scripts/verify_fused_rebuild.sh`（编译 → 安装 → 对拍 → 失败自动取证；从本目录所在的 `kimi3_ops/` 根跑也行）。

---

## 1. 重建后的骨架（每一处都有官方出处）

| # | 要点 | 官方出处（本机 `/usr/local/Ascend/cann-9.0.0/.../impl/...` 可直接核对） |
|---|---|---|
| ① | 入口：`TPipe` → `GET_TILING_DATA` → `GetUserWorkspace` → `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` | `upsample_linear1d.cpp`、`ffn.cpp`、`grouped_matmul_swiglu_quant.cpp` |
| ② | `REGIST_MATMUL_OBJ(&pipe, `**`GetSysWorkSpacePtr()`**`, mm, &tiling)` | 同上三个；**旧版把第二参传成了 user workspace —— 这是本轮修掉的关键一条** |
| ③ | Matmul 显式配置 `GetMDLConfig(true,false,0,false,false,false,true)` | `upsample_linear1d_mix.h` 的 `MDL_CFG_MIX`；`ffn.cpp` 的 `CFG_MDL` |
| ④ | C 声明为 `TPosition::GM` + `IterateAll<false>(cGm[...])`（C 落 workspace） | `upsample_linear1d_mix.h`、`grouped_matmul_swiglu_quant.h` |
| ⑤ | Cube 段（`if ASCEND_IS_AIC`）末尾 `SyncAll<false>()`；Vector 段（`if ASCEND_IS_AIV`）开头 `SyncAll<false>()` 握手 | `grouped_matmul_swiglu_quant.h` 的 `CubeProcess` / `VecProcess` |
| ⑥ | 逐 tile：`SetOrgShape/SetSingleShape/SetTensorA/SetTensorB/ClearBias/IterateAll/End` | `upsample_linear1d_mix.h`、`ffn_nonquant_nz.h` |
| ⑦ | epilogue 向量链（`Muls/Tanh/Sigmoid/Mul/Cast`）**原样搬自已真机验证的 `../SituAndMul`** | 本项目 `kernels/SituAndMul/op_kernel/situ_and_mul.cpp` |

> 上一版之所以挂住，是「Matmul 的 MIX 路径」与「msopgen 旧模板骨架」的组合问题
> （5 组配置全挂，见 `docs/14` §14.4）。重建版不再打补丁，而是照官方把**入口/注册/配置/分工/同步**五件事全部对齐。

## 2. ★ 代价：阶段 1 没有 HBM 收益（诚实说明）

重建版把 `C[M,2d] fp32` 落在 **workspace**，再由 AIV 读回来做 epilogue。这意味着：

- **中间张量仍然过了一遍 HBM**，甚至比「不融合」多一倍字节（fp32 写入 + fp32 读回）；
- 所以**这一版的性能没有收益**，它的唯一价值是「先把这个 MIX 通路跑通、把数值验证对」；
- 真正省掉 `[M,2d]` 往返要**阶段 2**：照 `ffn_nonquant_nz.h` 的 `IterateAll + GetTensorC<false>(ubTensor)`
  把 C 直接取到 UB（官方 FFN 就是这么把第一个 GEMM 的 C 喂给激活再喂第二个 GEMM 的）。
  那条路的同步更复杂（软同步 `SyncAll(syncGm, syncLocal)`），跑通阶段 1 之后再动。

## 3. 目录里有什么

```
SituGluFused/
├── situ_glu_fused_op.json          # msopgen 输入（算子定义）
└── op_kernel/
    ├── situ_glu_fused_tiling.h     # tiling 结构（Matmul + 分核 + epilogue）
    └── situ_glu_fused.cpp          # kernel：AIC 跑 Matmul 落 GM，AIV 跑 situ epilogue
└── op_host/
    └── situ_glu_fused.cpp          # host：InferShape + Tiling（Matmul tiling API + workspace）
```

`CMakeLists.txt` / `CMakePresets.json` / `build.sh` 用 msopgen 生成，别手抄
（照 `../SituAndMul/` 那套即可，生成后把 `ASCEND_COMPUTE_UNIT` 改成目标 SOC）。

## 4. 生成工程

```bash
source $ASCEND_HOME_PATH/set_env.sh
msopgen gen -i situ_glu_fused_op.json -f aclnn -c ai_core-<SOC> -lan cpp -out ./build_gen
# 把本目录的 op_kernel/ 与 op_host/ 覆盖进生成的骨架，再：
bash build.sh        # → build_out/custom_opp_<arch>.run
```

## 5. ★ 关键前置：权重必须**交错重排**，且两条不变量必须成立

```
W' 的行序 = [gate_0, up_0, gate_1, up_1, ...]   每片 subN 行
不变量 ①：W'[2k·subN + i] = gate[k·subN + i]，W'[(2k+1)·subN + i] = up[k·subN + i]
不变量 ②（本轮新增，host 侧强制检查）：d % subN == 0  且  d % 16 == 0
    ↑ subN 必须是 16 的倍数（UB 内每行 32B 对齐），且整除 d（否则 gate/up 错配）
    ↑ d 不是 16 的倍数时 host 直接 **GRAPH_FAILED 并打日志**，绝不静默算错
```

用仓库根目录（`kimi3_ops/`）的工具做（可跑的，带往返自校验）：

```bash
python3 ../../interleave_weight.py --mode interleave --subn 64   # 交错重排
python3 ../../interleave_weight.py --mode deinterleave --subn 64 # 逆变换（验证用）
```

**重排是一次性的**（权重在推理期不变），可以离线做完再喂给算子。
**如果忘了重排，结果会完全错**（gate/up 配不上对），而且不会报错——这是本方案最危险的一点。

> ✅ 这条现在可以**离线自检**（30 秒、纯 CPU、零依赖）：`python3 scripts/check_fused_index.py`
> —— 它照 kernel 的取数公式模拟一遍并与 golden 比对，还会故意「不做重排」确认真能报错（反向自证）。

## 6. 上机验证清单（2026-09-19 已上机：✅ 已确认 / ⛔ 未解决 / ❓ 仍待验证）

| # | 项 | 期望 | 如果不符 |
|---|---|---|---|
| 1 | 编译能过（`bash build.sh` → `EXIT=0`） | ✅ **已确认**（真机 `EXIT=0`；踩到并修掉 `Min()` 无 `uint32_t` 重载） | —— |
| 2 | `REGIST_MATMUL_OBJ` 的 KFC 分工是挂住的症结 | ✅ **已确认**：Matmul 必须由 **AIV（client）** 发起（AIC 在 `REGIST` 里就 `return` 了）→ 不再挂住 | —— |
| 3 | `GetBlockIdx()` 在 **AIV** 上 ∈ `[0, aivNum)`（即 0..47） | 48 个 AIV 各自分到不同的行块 | 若返回 0..23，一半行不会被处理 → 改用 `GetBlockIdx() / GetTaskRation()`（`ffn_nonquant_nz.h` 的写法）并配 `GetSubBlockIdx()` |
| 4 | `DataCopyPad` 的 `blockLen/srcStride` 单位与 UB 行距 | 字节；UB 行距 = `subN*sizeof(T)`（因 subN 是 16 倍数、32B 对齐） | 取数错位 → 改成逐行 `DataCopy` + `DataCopyParams`(datablock 单位) |
| 5 | `SyncAll<false>()` 能保证 AIC 的 C 写对 AIV 可见 | 无脏读 | 加 `DataCacheCleanAndInvalid`，或照 `ffn_nonquant_nz.h` 用 `SyncAll(syncGm, syncLocal)` 软同步 |
| 6 | Matmul 的 `IterateAll` 输出行距 = `N`（`C[M,2d]` 紧密） | 与 epilogue 的 `n2d` 假设一致 | 用 5 参 `SetOrgShape(orgM, orgN, orgKa, orgKb, orgKc)` 显式给 C 的行距（`upsample_linear1d_mix.h`） |
| 7 | 尾块（M 不是 `singleCoreM` 倍数） | `SetSingleShape(curM, curN, K)` 已按实际尺寸给 | 需要时补 `SetTail` |
| 8 | 数值对拍 `nBad == 0` | 容差 ±(1e-2 + 1e-2·\|ref\|) | 分段二分：先只跑 GEMM 通路，再开 epilogue |
| 9 | **删掉 `SetBufferSpace` 后 UB 的默认值够不够** | Matmul 在 MIX + C=GM 下不再需要 C 的 UB，默认值够 | 若编译期报 UB 超限或运行异常，按旧版给 Matmul 留一块 UB（`SetBufferSpace(-1,-1,ub)` + 二次 `GetTiling`） |
| 10 | `subN` 变宽（≈2752 而非几十/几百）不影响正确性 | 每片 = `epiRows` 行 × `subN` 列，`epiRows` 由 UB 预算算出 | 若 UB 超限就调小 `UB_BUDGET_DENOM` 的预算或强制 `subN` 上限 |
| 11 | **`GetUserWorkspace()` 的起点是否 == host 报的 `GetLibApiWorkSpaceSize()`** | 相等 —— 官方 `upsample_linear1d` / `ffn` 都是这套：host 报 `libWs + 用户区`，kernel 从 `GetUserWorkspace()` 起用 | 症状是**整片数据错乱/花屏**（不是数值精度问题）：C 的 fp32 缓冲和 sys workspace 重叠了 → 把 C 基址改成 `userWS + 固定偏移`，同时调整 host 的 workspace 计算 |

> **注意（与旧版调参经验不可比）**：旧版 `subN` 受「一个 `baseN` tile 内 gate/up 必须配对」约束
> （`subN ≤ baseN/2`，通常很小），新版 epilogue 已与 Matmul 分片**解耦**，`subN` 只受 UB 与 `d` 约束，
> 所以会大出一两个数量级（`d=11008` 时约 2752）。落盘里的 `kPairs` 现在**只是参考值**，不再是不变量。

## 7. 验证顺序（别跳步）

```bash
# ⓿ 上机前 30 秒自检（纯 CPU，不需要 NPU）：验证「重排规则 + 取数公式」自洽
#    ← 融合版最容易**静默算错**的就是 gate/up 配对，先把它排除掉
python3 scripts/check_fused_index.py

# ① 先过独立版（这是基准，不能被融合版影响）
cd ../SituAndMul && bash build.sh && cd ../.. && python3 test_situ_real.py

# ② 编译融合版
#    一键版（编译 → 安装 → 对拍 → 失败自动取证）：
bash scripts/verify_fused_rebuild.sh
#    或手动：
cd kernels/SituGluFused && bash build.sh

# ③ 对拍：golden 不变（reference/situ.py 数学相同），被测对象换成融合算子
#    scripts/verify_situ_glu_fused_aclnn.cpp（纯 aclrt，不需要 torch/torch_npu）：
#    先调一次让 host 把真实 subN 落盘到 /tmp/situ_glu_fused_tiling.txt，再按该 subN 重排权重
#    即：y_fused = SituGluFused(x, interleave(W_gate, W_up))
#    与 (situ_and_mul(x @ [W_gate;W_up]^T)) 的 golden 比对

# ④ 量收益：对比 ① 的基线（⚠️ 阶段 1 预期**无收益**，见 README §2）
python3 bench_situ_real.py
```

**③ 的黄金参考不用改**——数学一样，只是被测对象换成融合算子，这样收益才可比。
落盘 tiling 的格式**不能改**（`verify_situ_glu_fused_aclnn.cpp` 用 `sscanf` 按固定格式解析）。

## 8. 一句话

**这个融合的价值不在"把 epilogue 接上去"，而在"重排权重让 gate/up 落进同一个 tile"；
而这一版重建解决的是更前面的一件事——让 MIX（Cube+Vector）通路在 AscendC 上真的跑起来。**
