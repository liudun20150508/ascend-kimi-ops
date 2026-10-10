#include "kernel_operator.h"
#include "situ_and_mul_tiling.h"

using namespace AscendC;

// 队列深度 2 = double buffer：让 MTE2 预取下一块、当前块的 V 计算、上一块的 MTE3 写回
// 三段重叠。原来的实现 depth 全是 1，MTE2 / V / MTE3 完全串行。
// 类型写 int32_t：AscendC 的 TQue 模板签名是 `TQue<QuePosition, int32_t depth>`，
// 用 uint32_t 虽然能隐式转换，但直接对齐签名可以免掉一类挑剔的编译报错。
constexpr int32_t QUEUE_DEPTH = 2;

// ⚠️ 本 kernel 依赖两个前提，上机前确认：
//   ① d 必须是 16 的倍数（bf16 下 32B = 16 元素）。DataCopy 要求 32B 对齐，
//      d 不是 16 倍数时最后一个 tile 的拷贝长度会非对齐，可能越界。
//      KIMI3 的 intermediate_size = 11008 满足（11008 = 16 × 688）。
//   ② Level-2 向量接口允许原地运算（dst 与某个 src 同地址），本 kernel 靠这一点
//      把 fp32 缓冲从 4 块压到 3 块。原实现里 `Tanh(tmp1, tmp1, ...)` 已经是原地用法，
//      属于这套代码的既有约定；若目标 CANN 版本不支持，退回 4 块缓冲即可。

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

        // 任务粒度 = 一个 (row, tile) 分块，而不是整行：
        // 原来按 numRows/blockDim 均分整行，d 不是 blockDim 整数倍时尾核少算、其余核多算；
        // 现在按 tile 总数均分，长尾误差从「一行」降到「一个 tile」。
        uint32_t blockIdx = GetBlockIdx();
        uint32_t tilesPerCore = (totalTiles + blockDim - 1) / blockDim;
        tileStart = blockIdx * tilesPerCore;
        tileEnd = tileStart + tilesPerCore;
        if (tileEnd > totalTiles) {
            tileEnd = totalTiles;
        }

        xGm.SetGlobalBuffer((__gm__ bfloat16_t*)x);
        yGm.SetGlobalBuffer((__gm__ bfloat16_t*)y);

        pipe.InitBuffer(gateQueue, QUEUE_DEPTH, tileElem * sizeof(bfloat16_t));
        pipe.InitBuffer(upQueue, QUEUE_DEPTH, tileElem * sizeof(bfloat16_t));
        pipe.InitBuffer(outQueue, QUEUE_DEPTH, tileElem * sizeof(bfloat16_t));
        // fp32 计算缓冲 3 块（原实现 4 块）：见 Compute 里 acc 的复用
        pipe.InitBuffer(accQueue, 1, tileElem * sizeof(float));
        pipe.InitBuffer(tmp1Queue, 1, tileElem * sizeof(float));
        pipe.InitBuffer(tmp2Queue, 1, tileElem * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (tileStart >= tileEnd) {
            return;
        }
        Prefetch(tileStart);
        for (uint32_t t = tileStart; t < tileEnd; t++) {
            if (t + 1 < tileEnd) {
                // 先把下一块发出去（MTE2），再算当前块（V），靠 depth=2 的两块 buffer 重叠
                Prefetch(t + 1);
            }
            Compute(t);
        }
    }

private:
    // 把一个 tile 的 gate / up 两段从 GM 搬到 UB 并 EnQue（MTE2 段）
    __aicore__ inline void Prefetch(uint32_t t)
    {
        uint32_t row = t / tilesPerRow;
        uint32_t col = (t % tilesPerRow) * tileElem;
        uint32_t tile = tileElem;
        if (col + tile > d) {
            tile = d - col;
        }

        uint32_t rowInOff = row * 2 * d;  // 本行 gate 起始偏移（up 段在其后 d 处）
        LocalTensor<bfloat16_t> gateLocal = gateQueue.AllocTensor<bfloat16_t>();
        LocalTensor<bfloat16_t> upLocal = upQueue.AllocTensor<bfloat16_t>();
        DataCopy(gateLocal, xGm[rowInOff + col], tile);
        DataCopy(upLocal, xGm[rowInOff + d + col], tile);
        gateQueue.EnQue(gateLocal);
        upQueue.EnQue(upLocal);
    }

    // 向量计算 + 写回（V + MTE3 段）
    __aicore__ inline void Compute(uint32_t t)
    {
        uint32_t row = t / tilesPerRow;
        uint32_t col = (t % tilesPerRow) * tileElem;
        uint32_t tile = tileElem;
        if (col + tile > d) {
            tile = d - col;
        }

        LocalTensor<bfloat16_t> gateReady = gateQueue.DeQue<bfloat16_t>();
        LocalTensor<bfloat16_t> upReady = upQueue.DeQue<bfloat16_t>();

        LocalTensor<float> acc = accQueue.AllocTensor<float>();
        LocalTensor<float> tmp1 = tmp1Queue.AllocTensor<float>();
        LocalTensor<float> tmp2 = tmp2Queue.AllocTensor<float>();
        LocalTensor<bfloat16_t> outLocal = outQueue.AllocTensor<bfloat16_t>();

        // ---- gate 分支：situ(gate) = beta * tanh(gate/beta) * sigmoid(gate) ----
        Cast(acc, gateReady, RoundMode::CAST_NONE, tile);   // bf16 -> fp32（Tanh/Sigmoid 无 bf16 版本，必须升精度）
        if (beta == 4.0f) {
            // ★ 快路径（实测驱动；KIMI3 的 beta = 4.0）：**把 2 个超越函数压成 1 个 Exp**
            //   t = e^{−g/2}  ⇒  σ(g) = 1/(1+t²)，tanh(g/4) = (1−t)/(1+t)
            Muls(tmp1, acc, -0.5f, tile);                   // −g/2
            Mins(tmp1, tmp1, 30.0f, tile);                  // clamp：防 e^{大} → inf → NaN
            Maxs(tmp1, tmp1, -30.0f, tile);
            Exp(tmp1, tmp1, tile);                          // t = e^{−g/2}
            Mul(acc, tmp1, tmp1, tile);                     // t²
            Adds(acc, acc, 1.0f, tile);                     // 1 + t²
            Reciprocal(acc, acc, tile);                     // σ(g) = 1/(1+t²)
            Muls(tmp2, tmp1, -1.0f, tile);                  // −t
            Adds(tmp2, tmp2, 1.0f, tile);                   // 1 − t
            Adds(tmp1, tmp1, 1.0f, tile);                   // 1 + t
            Reciprocal(tmp1, tmp1, tile);                   // 1/(1+t)
            Mul(tmp1, tmp2, tmp1, tile);                    // tanh(g/4) = (1−t)/(1+t)
            Mul(tmp1, tmp1, acc, tile);                     // tanh · σ
            Muls(tmp1, tmp1, beta, tile);                   // situ(gate) 落在 tmp1
        } else {
            Muls(tmp1, acc, invBeta, tile);                 // 通用回退：gate / beta
            Tanh(tmp1, tmp1, tile);                         // tanh(gate/beta)
            Sigmoid(tmp2, acc, tile);                       // sigmoid(gate)
            Mul(tmp1, tmp1, tmp2, tile);                    // tanh * sigmoid
            Muls(tmp1, tmp1, beta, tile);                   // situ(gate) 落在 tmp1
        }

        // ---- up 分支：up' = linear_beta * tanh(up/linear_beta) ----
        // acc 在 gate 分支用完即废，直接复用为 up 的 fp32 输入 —— 省掉一整块 UB
        Cast(acc, upReady, RoundMode::CAST_NONE, tile);
        Muls(acc, acc, invLinearBeta, tile);
        Tanh(acc, acc, tile);
        Muls(acc, acc, linearBeta, tile);

        // ---- out = situ(gate) * up' ----
        Mul(tmp1, tmp1, acc, tile);
        Cast(outLocal, tmp1, RoundMode::CAST_ROUND, tile);
        outQueue.EnQue(outLocal);

        gateQueue.FreeTensor(gateReady);
        upQueue.FreeTensor(upReady);
        accQueue.FreeTensor(acc);
        tmp1Queue.FreeTensor(tmp1);
        tmp2Queue.FreeTensor(tmp2);

        LocalTensor<bfloat16_t> outReady = outQueue.DeQue<bfloat16_t>();
        DataCopy(yGm[row * d + col], outReady, tile);
        outQueue.FreeTensor(outReady);
    }

private:
    uint32_t d;
    uint32_t tileElem;
    uint32_t tilesPerRow;
    uint32_t totalTiles;
    uint32_t blockDim;
    float beta;
    float linearBeta;
    float invBeta;
    float invLinearBeta;
    uint32_t tileStart;
    uint32_t tileEnd;

    GlobalTensor<bfloat16_t> xGm;
    GlobalTensor<bfloat16_t> yGm;
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
