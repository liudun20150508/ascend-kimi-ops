#include "kernel_operator.h"
#include "situ_and_mul_tiling.h"

using namespace AscendC;

// d 维（切分维）上的分块长度，保证 UB 占用可控
constexpr uint32_t TILE_ELEM = 4096;

class KernelSituAndMul {
public:
    __aicore__ inline KernelSituAndMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, SituAndMulTilingData* tiling)
    {
        d = tiling->d;
        numRows = tiling->numRows;
        blockDim = tiling->blockDim;
        beta = tiling->beta;
        linear_beta = tiling->linear_beta;

        uint32_t blockIdx = GetBlockIdx();
        uint32_t rowsPerCore = (numRows + blockDim - 1) / blockDim;
        rowStart = blockIdx * rowsPerCore;
        rowEnd = rowStart + rowsPerCore;
        if (rowEnd > numRows) {
            rowEnd = numRows;
        }

        xGm.SetGlobalBuffer((__gm__ bfloat16_t*)x);
        yGm.SetGlobalBuffer((__gm__ bfloat16_t*)y);

        pipe.InitBuffer(gateQueue, 1, TILE_ELEM * sizeof(bfloat16_t));
        pipe.InitBuffer(upQueue, 1, TILE_ELEM * sizeof(bfloat16_t));
        pipe.InitBuffer(gateF32Queue, 1, TILE_ELEM * sizeof(float));
        pipe.InitBuffer(upF32Queue, 1, TILE_ELEM * sizeof(float));
        pipe.InitBuffer(tmp1Queue, 1, TILE_ELEM * sizeof(float));
        pipe.InitBuffer(tmp2Queue, 1, TILE_ELEM * sizeof(float));
        pipe.InitBuffer(outQueue, 1, TILE_ELEM * sizeof(bfloat16_t));
    }

    __aicore__ inline void Process()
    {
        if (rowStart >= rowEnd) {
            return;
        }
        for (uint32_t r = rowStart; r < rowEnd; r++) {
            uint32_t rowInOff = r * 2 * d;   // 本行 gate 起始偏移
            uint32_t rowOutOff = r * d;      // 本行输出起始偏移
            for (uint32_t c = 0; c < d; c += TILE_ELEM) {
                uint32_t tile = TILE_ELEM;
                if (c + tile > d) {
                    tile = d - c;
                }
                ProcessTile(rowInOff + c, rowInOff + d + c, rowOutOff + c, tile);
            }
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t gateOff, uint32_t upOff, uint32_t outOff, uint32_t tile)
    {
        // ---------- Stage 1: copy in（MTE2）----------
        LocalTensor<bfloat16_t> gateLocal = gateQueue.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> upLocal = upQueue.AllocTensor<bfloat16_t>();
        DataCopy(gateLocal, xGm[gateOff], tile);
        DataCopy(upLocal, xGm[upOff], tile);
        gateQueue.EnQue(gateLocal);
        upQueue.EnQue(upLocal);

        // ---------- Stage 2: compute（V）----------
        LocalTensor<bfloat16_t> gateReady = gateQueue.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> upReady = upQueue.DeQue<bfloat16_t>();

        LocalTensor<float> gateF32 = gateF32Queue.AllocTensor<float>();
        LocalTensor<float> upF32 = upF32Queue.AllocTensor<float>();
        LocalTensor<float> tmp1 = tmp1Queue.AllocTensor<float>();
        LocalTensor<float> tmp2 = tmp2Queue.AllocTensor<float>();
        LocalTensor<bfloat16_t> outLocal = outQueue.AllocTensor<bfloat16_t>();

        // gate 分支：situ(gate) = beta * tanh(gate/beta) * sigmoid(gate)
        Cast(gateF32, gateReady, RoundMode::CAST_NONE, tile);
        Muls(tmp1, gateF32, 1.0f / beta, tile);   // gate / beta
        Tanh(tmp1, tmp1, tile);                   // tanh(gate/beta)
        Sigmoid(tmp2, gateF32, tile);             // sigmoid(gate)
        Mul(gateF32, tmp1, tmp2, tile);           // tanh * sigmoid
        Muls(gateF32, gateF32, beta, tile);       // * beta

        // up 分支：up' = linear_beta * tanh(up/linear_beta)
        Cast(upF32, upReady, RoundMode::CAST_NONE, tile);
        Muls(tmp1, upF32, 1.0f / linear_beta, tile);  // up / linear_beta
        Tanh(tmp1, tmp1, tile);                      // tanh(up/linear_beta)
        Muls(tmp1, tmp1, linear_beta, tile);         // * linear_beta

        // out = situ(gate) * up'
        Mul(upF32, gateF32, tmp1, tile);
        Cast(outLocal, upF32, RoundMode::CAST_ROUND, tile);

        outQueue.EnQue(outLocal);   // 信号：V 已写完 outLocal，可交给 MTE3

        gateQueue.FreeTensor(gateReady);
        upQueue.FreeTensor(upReady);
        gateF32Queue.FreeTensor(gateF32);
        upF32Queue.FreeTensor(upF32);
        tmp1Queue.FreeTensor(tmp1);
        tmp2Queue.FreeTensor(tmp2);

        // ---------- Stage 3: copy out（MTE3）----------
        LocalTensor<bfloat16_t> outReady = outQueue.DeQue<bfloat16_t>();
        DataCopy(yGm[outOff], outReady, tile);
        outQueue.FreeTensor(outReady);
    }

private:
    uint32_t d;
    uint32_t numRows;
    uint32_t blockDim;
    float beta;
    float linear_beta;
    uint32_t rowStart;
    uint32_t rowEnd;

    GlobalTensor<bfloat16_t> xGm;
    GlobalTensor<bfloat16_t> yGm;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> gateQueue;
    TQue<QuePosition::VECIN, 1> upQueue;
    TQue<QuePosition::VECCALC, 1> gateF32Queue;
    TQue<QuePosition::VECCALC, 1> upF32Queue;
    TQue<QuePosition::VECCALC, 1> tmp1Queue;
    TQue<QuePosition::VECCALC, 1> tmp2Queue;
    TQue<QuePosition::VECOUT, 1> outQueue;
};

extern "C" __global__ __aicore__ void situ_and_mul(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SituAndMulTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelSituAndMul op;
    op.Init(x, y, &tilingData);
    op.Process();
}
