#include "kernel_operator.h"
#include "situ_and_mul_tiling.h"

using namespace AscendC;

constexpr int32_t QUEUE_DEPTH = 2;   // 与优化版一致（控制变量）

// ===== BASELINE：三段串行，不预取、不重叠 =====
// 与优化版的**唯一差别**：每处理一个 tile 都要走完
//   「copy in → compute → copy out」，
// 下一块必须等这一块全部结束。用来量化 double buffer 的净收益。
class KernelSituAndMul {
public:
    __aicore__ inline KernelSituAndMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, SituAndMulTilingData* tiling)
    {
        d = tiling->d;
        tileElem = tiling->tileElem;
        tilesPerRow = tiling->tilesPerRow;
        totalTiles = tiling->totalTiles;
        blockDim = tiling->blockDim;
        beta = tiling->beta;
        linearBeta = tiling->linear_beta;
        invBeta = tiling->invBeta;
        invLinearBeta = tiling->invLinearBeta;

        uint32_t blockIdx = GetBlockIdx();
        uint32_t tilesPerCore = (totalTiles + blockDim - 1) / blockDim;
        tileStart = blockIdx * tilesPerCore;
        tileEnd = tileStart + tilesPerCore;
        if (tileEnd > totalTiles) tileEnd = totalTiles;

        xGm.SetGlobalBuffer((__gm__ bfloat16_t*)x);
        yGm.SetGlobalBuffer((__gm__ bfloat16_t*)y);

        pipe.InitBuffer(gateQueue, QUEUE_DEPTH, tileElem * sizeof(bfloat16_t));
        pipe.InitBuffer(upQueue,   QUEUE_DEPTH, tileElem * sizeof(bfloat16_t));
        pipe.InitBuffer(outQueue,  QUEUE_DEPTH, tileElem * sizeof(bfloat16_t));
        pipe.InitBuffer(accQueue,  1, tileElem * sizeof(float));
        pipe.InitBuffer(tmp1Queue, 1, tileElem * sizeof(float));
        pipe.InitBuffer(tmp2Queue, 1, tileElem * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t t = tileStart; t < tileEnd; t++) {
            uint32_t row = t / tilesPerRow;
            uint32_t col = (t % tilesPerRow) * tileElem;
            uint32_t tile = tileElem;
            if (col + tile > d) tile = d - col;
            uint32_t rowInOff = row * 2 * d;

            // ---- Stage 1: copy in（等它完成才进 Stage 2）----
            LocalTensor<bfloat16_t> gateLocal = gateQueue.AllocTensor<bfloat16_t>();
            LocalTensor<bfloat16_t> upLocal = upQueue.AllocTensor<bfloat16_t>();
            DataCopy(gateLocal, xGm[rowInOff + col], tile);
            DataCopy(upLocal, xGm[rowInOff + d + col], tile);
            gateQueue.EnQue(gateLocal);
            upQueue.EnQue(upLocal);

            // ---- Stage 2: compute ----
            LocalTensor<bfloat16_t> gateReady = gateQueue.DeQue<bfloat16_t>();
            LocalTensor<bfloat16_t> upReady = upQueue.DeQue<bfloat16_t>();
            LocalTensor<float> acc = accQueue.AllocTensor<float>();
            LocalTensor<float> tmp1 = tmp1Queue.AllocTensor<float>();
            LocalTensor<float> tmp2 = tmp2Queue.AllocTensor<float>();
            LocalTensor<bfloat16_t> outLocal = outQueue.AllocTensor<bfloat16_t>();

            Cast(acc, gateReady, RoundMode::CAST_NONE, tile);
            Muls(tmp1, acc, invBeta, tile);
            Tanh(tmp1, tmp1, tile);
            Sigmoid(tmp2, acc, tile);
            Mul(tmp1, tmp1, tmp2, tile);
            Muls(tmp1, tmp1, beta, tile);

            Cast(acc, upReady, RoundMode::CAST_NONE, tile);
            Muls(acc, acc, invLinearBeta, tile);
            Tanh(acc, acc, tile);
            Muls(acc, acc, linearBeta, tile);

            Mul(tmp1, tmp1, acc, tile);
            Cast(outLocal, tmp1, RoundMode::CAST_ROUND, tile);
            outQueue.EnQue(outLocal);

            gateQueue.FreeTensor(gateReady);
            upQueue.FreeTensor(upReady);
            accQueue.FreeTensor(acc);
            tmp1Queue.FreeTensor(tmp1);
            tmp2Queue.FreeTensor(tmp2);

            // ---- Stage 3: copy out（等它完成才进下一轮 Stage 1）----
            LocalTensor<bfloat16_t> outReady = outQueue.DeQue<bfloat16_t>();
            DataCopy(yGm[row * d + col], outReady, tile);
            outQueue.FreeTensor(outReady);
        }
    }

private:
    uint32_t d, tileElem, tilesPerRow, totalTiles, blockDim;
    uint32_t tileStart, tileEnd;
    float beta, linearBeta, invBeta, invLinearBeta;
    GlobalTensor<bfloat16_t> xGm, yGm;
    TPipe pipe;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> gateQueue;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> upQueue;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> outQueue;
    TQue<QuePosition::VECCALC, 1> accQueue;
    TQue<QuePosition::VECCALC, 1> tmp1Queue;
    TQue<QuePosition::VECCALC, 1> tmp2Queue;
};

extern "C" __global__ __aicore__ void situ_and_mul(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SituAndMulTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelSituAndMul op;
    op.Init(x, y, &tilingData);
    op.Process();
}
