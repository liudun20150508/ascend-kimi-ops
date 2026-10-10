/* -------------------------------------------------------------------------
 * SituGluFused：GEMM + SiTU-GLU 融合算子的 tiling 结构
 *
 * 数学：
 *   gate_up = x @ W'^T          x:[M,H]  W':[2d,H]（**行序已交错重排**）
 *   out     = situ(gate) * tanh_scale(up)      out:[M,d]
 *
 * 为什么 W' 要重排：见 ../README.md 和 ../../docs/13-SiTU-GLU融合实现设计.md
 * ------------------------------------------------------------------------- */

#ifndef SITU_GLU_FUSED_TILING_H
#define SITU_GLU_FUSED_TILING_H

#include <cstdint>
// TCubeTiling 来自这里。注意 host / kernel 两侧 include path 不同：
//   kernel 侧：asc/include/adv_api/...（本文件按此路径写）
//   host  侧：tiling/tiling_api.h 会把它带进来
#include "adv_api/kernel_tiling.h"

struct SituGluFusedTilingData {
    // ---------- Matmul 侧 ----------
    // 由 host 端用 matmul_tiling::MultiCoreMatmulTiling 生成，**不能手写**
    // ★ 重建（2026-09-19）后 C 声明为 **GM**（落 workspace），与 kernel 侧
    //   SitGluCType = MatmulType<TPosition::GM, ...> 必须一致，否则 SetCType 与
    //   IterateAll 的目标地址类型对不上（官方 MIX 样例：upsample_linear1d /
    //   grouped_matmul_swiglu_quant 都是 C 落 GM）。
    TCubeTiling mmTiling;

    // ---------- 分核 ----------
    uint32_t coreNum;        // AIC 核数 = host 的 SetBlockDim 值（MIX 下 blockDim 按 AIC 计）
    uint32_t aivNum;         // AIV 核数；MIX_AIC_1_2 ⇒ = 2 × coreNum
                             //   epilogue 完全跑在 AIV 上（官方 swiglu 的 VectorProcess 同构）
    uint32_t epiRows;        // epilogue 每次处理的行数（由 host 按 UB 预算算出）

    // ---------- epilogue 侧 ----------
    uint32_t baseM;          // = mmTiling.baseM（Matmul 的 M 粒度，落盘文件要用）
    uint32_t baseN;          // = mmTiling.baseN（Matmul 的 N 粒度）
    uint32_t subN;           // epilogue 一次处理的列数
                             // ★ 不变量（重建后新增）：subN 是 16 的倍数，且 **d % subN == 0**
                             //   —— 权重重排工具 interleave_weight.py 的硬前提
                             //   （W'[2k*subN+i] = gate[k*subN+i]），否则 gate/up 静默错配。
    uint32_t kPairs;         // 一个 baseN tile 内含的 (gate,up) 对数（仅用于落盘/文档口径）

    uint32_t m;              // token 数（行数）
    uint32_t d;              // 输出通道 = intermediate_size
    uint32_t h;              // 输入特征维（K 维）

    float beta;              // situ 的 beta
    float linear_beta;       // up 分支的 linear_beta
    float invBeta;           // 1/beta（host 端算好，kernel 内不做除法）
    float invLinearBeta;     // 1/linear_beta
};

#endif // SITU_GLU_FUSED_TILING_H
