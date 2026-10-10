#include "kernel_operator.h"
#include "adv_api/matmul/matmul_intf.h"   // CANN 9.0：Matmul 高阶 API 在 asc/include/adv_api
#include "situ_glu_fused_tiling.h"

using namespace AscendC;

/* =========================================================================
 * SituGluFused —— GEMM + SiTU-GLU 一次做完（MIX（AIC+AIV））
 *
 *   入口 → AIC：按 tiling 分核跑 Matmul，C[M,2d] fp32 落 workspace
 *        → SyncAll → AIV：按行分核读 C，算 situ(gate)×tanh_scale(up) 写 y
 *
 * W' 是**行序交错重排**后的权重：W'[2k·subN+i] = gate[k·subN+i]，W'[(2k+1)·subN+i] = up[k·subN+i]
 * =========================================================================
 * ★ 2026-09-19 真机定位（三条，全部有设备侧证据）
 *
 * ① **AscendC 的 `Min()` 对 `uint32_t` 没有匹配重载**（官方 upsample 用 int64_t 所以能过）
 *    → 自写 `MinU32`。
 *
 * ② **Matmul 的两种集成模式差别巨大**：
 *      模式 A（KFC）：`REGIST_MATMUL_OBJ(...)`，AIC 进服务循环后 `return`，
 *                    **Matmul API 由 AIV 当 client 发起**（官方 upsample_linear1d / ffn 用）。
 *      模式 B（AIC 直执行）：`if ASCEND_IS_AIC { mm.Init(&tiling, &tpipe); }`，
 *                    AIC 自己跑 Matmul，AIV 完全不碰（官方 grouped_matmul_swiglu_quant 用）。
 *    **本算子用模式 A 时：数值 4 组 shape 全 PASS，但「重复执行」必失败** ——
 *    `mssanitizer` 直接指出 `Register FFTS_BASE_ADDR was not reset to default in block aiv(...)`
 *    （残留 4398046396416，应为 0）⇒ 第 2 次 launch 就报
 *    `fftsplus aicore error: The DDR address of the MTE instruction is out of range`。
 *    ⇒ 改用**模式 B**（AIV 不碰 KFC，FFTS_BASE_ADDR 不残留）。
 *
 * ③ **B = W' 是 [2d,H] = [N,K] 行主序，必须转置**（`SetTensorB(gm, true)` + host 侧
 *    `SetBType(..., isTrans=true)`），否则 Matmul 按 [K,N] 取数 → 结果全错。
 * ========================================================================= */

// ⚠️ 类型别名用独特前缀：AT/BT/CT 这类短名会和 CANN 头里的符号撞
using SitGluAType    = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;  // A = x   [M,K]
using SitGluBType    = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, false>;  // B = W'  [N,K]（转置）
using SitGluCType    = MatmulType<TPosition::GM, CubeFormat::ND, float>;              // C → workspace GM
using SitGluBiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

// 照官方 upsample_linear1d.cpp 的 MDL_CFG_MIX
constexpr MatmulConfig SITU_GLU_MM_CFG = GetMDLConfig(true, false, 0, false, false, false, true);

using SitGluMmType = Matmul<SitGluAType, SitGluBType, SitGluCType, SitGluBiasType, SITU_GLU_MM_CFG>;

// ★ 实测：AscendC 的 Min() 在 uint32_t 上没有匹配重载（见文件头 ①）
__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b) { return (a < b) ? a : b; }

class KernelSituGluFused {
public:
    __aicore__ inline KernelSituGluFused(SitGluMmType& mm_, TPipe* pipe_) : mm(mm_), pipe(pipe_) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR y, GM_ADDR cWs,
                                SituGluFusedTilingData* t)
    {
        subN          = t->subN;
        m             = t->m;
        d             = t->d;
        h             = t->h;
        coreNum       = t->coreNum;
        aivNum        = t->aivNum;
        epiRows       = t->epiRows;
        beta          = t->beta;
        linearBeta    = t->linear_beta;
        invBeta       = t->invBeta;
        invLinearBeta = t->invLinearBeta;

        singleCoreM = static_cast<uint32_t>(t->mmTiling.singleCoreM);
        singleCoreN = static_cast<uint32_t>(t->mmTiling.singleCoreN);
        n2d         = 2 * d;

        xGm.SetGlobalBuffer((__gm__ bfloat16_t*)x);
        wGm.SetGlobalBuffer((__gm__ bfloat16_t*)w);
        yGm.SetGlobalBuffer((__gm__ bfloat16_t*)y);
        cGm.SetGlobalBuffer((__gm__ float*)cWs);      // C 的 workspace：[M, 2d] 行主序 fp32

        if (epiRows == 0) {
            epiRows = 1;
        }

        // 只有 AIV 用这些缓冲（AIC 分支不碰；但 InitBuffer 在两个核上执行无副作用）
        const uint32_t tile = epiRows * subN;
        pipe->InitBuffer(gQueue,   1, tile * sizeof(float));
        pipe->InitBuffer(uQueue,   1, tile * sizeof(float));
        pipe->InitBuffer(tmp1Queue, 1, tile * sizeof(float));
        pipe->InitBuffer(tmp2Queue, 1, tile * sizeof(float));
        pipe->InitBuffer(outQueue, 1, tile * sizeof(bfloat16_t));
    }

    __aicore__ inline void Process()
    {
        // ★ 模式 A（官方 upsample_linear1d 的用法）：AIC 在 REGIST_MATMUL_OBJ 里进 KFC 服务
        //   循环并 return，**Matmul 必须由 AIV（client）发起**。见文件头 ②。
        if ASCEND_IS_AIV {
            const uint32_t aivIdx = GetBlockIdx();
            if (aivIdx >= aivNum) {
                return;
            }
            const uint32_t rowsPerCore = (m + aivNum - 1) / aivNum;
            const uint32_t rowStart = aivIdx * rowsPerCore;
            if (rowStart >= m) {
                return;
            }
            const uint32_t myRows = MinU32(rowsPerCore, m - rowStart);
            MatmulForRows(rowStart, myRows);      // ① Cube（KFC client 驱动 AIC）
            EpilogueRows(rowStart, myRows);       // ② Vector
        }
    }

    /* ---- ① Cube 段（AIV 发起）：本核负责的 myRows 行的 C 全部写进 workspace ---- */
    __aicore__ inline void MatmulForRows(uint32_t rowStart, uint32_t myRows)
    {
        // ★ 2026-09-19 性能修正（msprof 定位）：
        //   原来按 singleCoreN 逐块循环 + **每块 `IterateAll<true>`**，等于「发 KFC 请求 → 等 fixpipe」
        //   重复 (n2d/singleCoreN) 次（d=11008 时 **86 次**）⇒ 串行气泡 86×~10μs ≈ 860μs，
        //   与实测 847~857μs 吻合；msprof 也显示**所有管道利用率都 < 80%**（时间全花在等）。
        //   ⇒ 一次把整行 N 交给 Matmul，让它内部按 baseN 流水迭代（L1/L0 双缓冲由它自己管）。
        // ★ 2026-09-19 性能修正（msprof 实测）：原先把 7 个 Matmul API 全放在 baseN 循环里，
        //   每个 API 都要向 AIC 发一次 KFC 消息 ⇒ 每块固定开销 ≈10μs（而块内有效搬运仅 0.42μs）。
        //   实测：d=2048/5504/11008（24/43/86 块）= 283/436/830μs，**严格线性**。
        //   ⇒ 把**不随块变化**的 shape/A/bias 设置移出循环，每块只留 SetTensorB + IterateAll。
        const bool hasTail = (n2d % singleCoreN) != 0;
        mm.SetOrgShape(myRows, singleCoreN, h, h, n2d);
        mm.SetSingleShape(myRows, singleCoreN, h);
        mm.SetTensorA(xGm[static_cast<uint64_t>(rowStart) * h]);
        mm.ClearBias();

        for (uint32_t n0 = 0; n0 < n2d; n0 += singleCoreN) {
            const uint32_t curN = MinU32(singleCoreN, n2d - n0);
            if (hasTail && curN != singleCoreN) {
                mm.SetTail(myRows, curN, h);                 // 尾块：只有它需要改单核形状
            }
            mm.SetTensorB(wGm[static_cast<uint64_t>(n0) * h], true);
            mm.template IterateAll<true>(cGm[static_cast<uint64_t>(rowStart) * n2d + n0]);
        }
        mm.End();
    }

    /* ---- ② Vector 段：读本核那块 C，按 (gate_j, up_j) 配对算 SiTU-GLU 并写回 y ---- */
    __aicore__ inline void EpilogueRows(uint32_t rowStart, uint32_t myRows)
    {
        const uint32_t nPairs = d / subN;
        const uint32_t rowEnd = rowStart + myRows;
        for (uint32_t row0 = rowStart; row0 < rowEnd; row0 += epiRows) {
            const uint32_t rows = MinU32(epiRows, rowEnd - row0);
            for (uint32_t j = 0; j < nPairs; ++j) {
                EpilogueTile(row0, rows, j);
            }
        }
    }

private:
    /* 处理「行块 × 第 pairIdx 对 (gate,up) 片」 */
    __aicore__ inline void EpilogueTile(uint32_t row0, uint32_t rows, uint32_t pairIdx)
    {
        const uint32_t n = rows * subN;
        const uint32_t colGate = pairIdx * 2 * subN;
        const uint32_t colUp = colGate + subN;

        // DataCopyExtParams 的 blockLen/srcStride 单位是**字节**；
        // subN 是 16 的倍数 ⇒ subN*4 是 32B 的倍数 ⇒ UB 内是紧密的 rows×subN 布局
        LocalTensor<float> g = gQueue.AllocTensor<float>();
        LocalTensor<float> u = uQueue.AllocTensor<float>();

        DataCopyExtParams cp;
        cp.blockCount = static_cast<uint16_t>(rows);
        cp.blockLen = static_cast<uint32_t>(subN * sizeof(float));
        cp.srcStride = static_cast<uint32_t>((n2d - subN) * sizeof(float));
        cp.dstStride = 0;
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(g, cGm[static_cast<uint64_t>(row0) * n2d + colGate], cp, padParams);
        DataCopyPad(u, cGm[static_cast<uint64_t>(row0) * n2d + colUp], cp, padParams);
        gQueue.EnQue(g);                       // EnQue/DeQue 负责插 MTE2→V 同步
        uQueue.EnQue(u);

        LocalTensor<float> gr = gQueue.DeQue<float>();
        LocalTensor<float> ur = uQueue.DeQue<float>();
        LocalTensor<float> t1 = tmp1Queue.AllocTensor<float>();
        LocalTensor<float> t2 = tmp2Queue.AllocTensor<float>();

        // situ(gate) = beta · tanh(gate/beta) · sigmoid(gate)
        // 与真机验证过的 kernels/SituAndMul 数学一字不差（那边 max_rel 3.89e-3）
        Muls(t1, gr, invBeta, n);
        Tanh(t1, t1, n);
        Sigmoid(t2, gr, n);
        Mul(t1, t1, t2, n);
        Muls(t1, t1, beta, n);

        // up' = linear_beta · tanh(up/linear_beta)
        Muls(ur, ur, invLinearBeta, n);
        Tanh(ur, ur, n);
        Muls(ur, ur, linearBeta, n);

        // out = situ(gate) · up'
        Mul(t1, t1, ur, n);
        LocalTensor<bfloat16_t> o = outQueue.AllocTensor<bfloat16_t>();
        Cast(o, t1, RoundMode::CAST_ROUND, n);
        outQueue.EnQue(o);                     // V→MTE3 同步交给 DeQue

        LocalTensor<bfloat16_t> orr = outQueue.DeQue<bfloat16_t>();
        DataCopyExtParams wp;
        wp.blockCount = static_cast<uint16_t>(rows);
        wp.blockLen = static_cast<uint32_t>(subN * sizeof(bfloat16_t));
        wp.srcStride = 0;
        wp.dstStride = static_cast<uint32_t>((d - subN) * sizeof(bfloat16_t));
        DataCopyPad(yGm[static_cast<uint64_t>(row0) * d + pairIdx * subN], orr, wp);

        gQueue.FreeTensor(gr);
        uQueue.FreeTensor(ur);
        tmp1Queue.FreeTensor(t1);
        tmp2Queue.FreeTensor(t2);
        outQueue.FreeTensor(orr);
    }

private:
    SitGluMmType& mm;

    uint32_t subN;
    uint32_t m, d, h, n2d;
    uint32_t coreNum, aivNum, epiRows;
    uint32_t singleCoreM, singleCoreN;
    float beta, linearBeta, invBeta, invLinearBeta;

    GlobalTensor<bfloat16_t> xGm, wGm, yGm;
    GlobalTensor<float> cGm;
    TPipe* pipe = nullptr;

    TQue<QuePosition::VECIN, 1> gQueue;
    TQue<QuePosition::VECIN, 1> uQueue;
    TQue<QuePosition::VECCALC, 1> tmp1Queue;
    TQue<QuePosition::VECCALC, 1> tmp2Queue;
    TQue<QuePosition::VECOUT, 1> outQueue;
};

extern "C" __global__ __aicore__ void situ_glu_fused(GM_ADDR x, GM_ADDR w, GM_ADDR y,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    REGISTER_TILING_DEFAULT(SituGluFusedTilingData);
    GET_TILING_DATA(tilingData, tiling);

    GM_ADDR userWS = GetUserWorkspace(workspace);   // C 的 fp32 缓冲从这里开始
    if (userWS == nullptr) {
        return;
    }

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    // ★ 模式 A：REGIST_MATMUL_OBJ 在 AIC 上会进 KFC 服务循环并 return；
    //   Matmul 的调用由 AIV（client）发起（见文件头 ②）。
    SitGluMmType mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tilingData.mmTiling);

    KernelSituGluFused op(mm, &pipe);
    op.Init(x, w, y, userWS, &tilingData);
    op.Process();
}
