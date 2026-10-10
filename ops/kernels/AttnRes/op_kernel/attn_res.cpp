#include "kernel_operator.h"
#include "attn_res_tiling.h"

using namespace AscendC;

// softmax 缓冲按 32 对齐（向量 Exp/ReduceSum 需要对齐）
constexpr uint32_t SOFTMAX_ALIGN = 32;

class KernelAttnRes {
public:
    __aicore__ inline KernelAttnRes() {}

    __aicore__ inline void Init(GM_ADDR blockRes, GM_ADDR prefixSum, GM_ADDR scoreWeight, GM_ADDR out,
                                AttnResTilingData* tiling)
    {
        T = tiling->T;
        B = tiling->B;
        H = tiling->H;
        blockDim = tiling->blockDim;
        eps = tiling->eps;
        invH = tiling->invH;

        uint32_t blockIdx = GetBlockIdx();
        uint32_t rowsPerCore = (T + blockDim - 1) / blockDim;
        rowStart = blockIdx * rowsPerCore;
        rowEnd = rowStart + rowsPerCore;
        if (rowEnd > T) {
            rowEnd = T;
        }

        blockResGm.SetGlobalBuffer((__gm__ bfloat16_t*)blockRes);
        prefixGm.SetGlobalBuffer((__gm__ bfloat16_t*)prefixSum);
        scoreGm.SetGlobalBuffer((__gm__ bfloat16_t*)scoreWeight);
        outGm.SetGlobalBuffer((__gm__ bfloat16_t*)out);

        pipe.InitBuffer(vBf16Queue, 1, H * sizeof(bfloat16_t));
        pipe.InitBuffer(swBf16Queue, 1, H * sizeof(bfloat16_t));
        pipe.InitBuffer(vF32Queue, 1, H * sizeof(float));
        pipe.InitBuffer(swF32Queue, 1, H * sizeof(float));
        pipe.InitBuffer(tmpQueue, 1, H * sizeof(float));
        pipe.InitBuffer(outF32Queue, 1, H * sizeof(float));
        pipe.InitBuffer(outBf16Queue, 1, H * sizeof(bfloat16_t));
        pipe.InitBuffer(reduceDstQueue, 1, 32 * sizeof(float));
        pipe.InitBuffer(reduceWorkQueue, 1, H * sizeof(float));
        pipe.InitBuffer(softmaxBufQueue, 1, SOFTMAX_ALIGN * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (rowStart >= rowEnd) {
            return;
        }
        for (uint32_t t = rowStart; t < rowEnd; t++) {
            ProcessToken(t);
        }
    }

private:
    // 载入 v[b]（token t 第 b 个残差向量，H 维）并 cast 到 fp32，由调用方 FreeTensor。
    __aicore__ inline void LoadV(uint32_t t, uint32_t b, LocalTensor<float>& vF32)
    {
        LocalTensor<bfloat16_t> vBf16 = vBf16Queue.AllocTensor<bfloat16_t>();
        if (b < B) {
            DataCopy(vBf16, blockResGm[t * B * H + b * H], H);
        } else {
            DataCopy(vBf16, prefixGm[t * H], H);
        }
        vBf16Queue.EnQue(vBf16);
        vBf16 = vBf16Queue.DeQue<bfloat16_t>();
        vF32 = vF32Queue.AllocTensor<float>();
        Cast(vF32, vBf16, RoundMode::CAST_NONE, H);
        vBf16Queue.FreeTensor(vBf16);
    }

    // 向量 softmax：buf[0..n-1] 存 scores（其余位置已置 -1e30），返回后 buf[0..n-1] 为 probs。
    __aicore__ inline void Softmax(LocalTensor<float>& buf, LocalTensor<float>& reduceDst,
                                   LocalTensor<float>& reduceWork, uint32_t n)
    {
        float mx = buf.GetValue(0);
        for (uint32_t i = 1; i < n; i++) {
            float v = buf.GetValue(i);
            mx = (v > mx) ? v : mx;
        }
        Adds(buf, buf, -mx, SOFTMAX_ALIGN);
        Exp(buf, buf, SOFTMAX_ALIGN);
        ReduceSum(reduceDst, buf, reduceWork, SOFTMAX_ALIGN);
        float sum = reduceDst.GetValue(0);
        Muls(buf, buf, 1.0f / sum, SOFTMAX_ALIGN);
    }

    __aicore__ inline void ProcessToken(uint32_t t)
    {
        LocalTensor<float> vF32;
        LocalTensor<float> swF32 = swF32Queue.AllocTensor<float>();
        LocalTensor<float> tmp = tmpQueue.AllocTensor<float>();
        LocalTensor<float> outF32 = outF32Queue.AllocTensor<float>();
        LocalTensor<float> reduceDst = reduceDstQueue.AllocTensor<float>();
        LocalTensor<float> reduceWork = reduceWorkQueue.AllocTensor<float>();
        LocalTensor<float> softmaxBuf = softmaxBufQueue.AllocTensor<float>();

        // 载入 score_weight
        LocalTensor<bfloat16_t> swBf16 = swBf16Queue.AllocTensor<bfloat16_t>();
        DataCopy(swBf16, scoreGm, H);
        swBf16Queue.EnQue(swBf16);
        swBf16 = swBf16Queue.DeQue<bfloat16_t>();
        Cast(swF32, swBf16, RoundMode::CAST_NONE, H);
        swBf16Queue.FreeTensor(swBf16);

        uint32_t nb = B + 1;

        // Pass 1：score[b] = rsqrt(var + eps) * <v[b], score_weight>
        for (uint32_t b = 0; b < nb; b++) {
            LoadV(t, b, vF32);

            Mul(tmp, vF32, vF32, H);                       // v^2
            ReduceSum(reduceDst, tmp, reduceWork, H);      // sum(v^2)
            float sumSq = reduceDst.GetValue(0);

            Mul(tmp, vF32, swF32, H);                      // v * score_weight
            ReduceSum(reduceDst, tmp, reduceWork, H);      // sum(v*score_weight)
            float sumSw = reduceDst.GetValue(0);

            float var = sumSq * invH;
            softmaxBuf.SetValue(b, (1.0f / sqrt(var + eps)) * sumSw);

            vF32Queue.FreeTensor(vF32);
        }

        // softmax 在 B+1 维（对齐到 32，padding 置 -1e30）
        for (uint32_t b = nb; b < SOFTMAX_ALIGN; b++) {
            softmaxBuf.SetValue(b, -1e30f);
        }
        Softmax(softmaxBuf, reduceDst, reduceWork, nb);

        // Pass 2：out = sum_b probs[b] * v[b]
        for (uint32_t b = 0; b < nb; b++) {
            LoadV(t, b, vF32);
            float p = softmaxBuf.GetValue(b);
            if (b == 0) {
                Muls(outF32, vF32, p, H);
            } else {
                Muls(tmp, vF32, p, H);
                Add(outF32, outF32, tmp, H);
            }
            vF32Queue.FreeTensor(vF32);
        }

        // fp32 -> bf16 写回
        LocalTensor<bfloat16_t> outBf16 = outBf16Queue.AllocTensor<bfloat16_t>();
        Cast(outBf16, outF32, RoundMode::CAST_ROUND, H);
        outBf16Queue.EnQue(outBf16);
        outBf16 = outBf16Queue.DeQue<bfloat16_t>();
        DataCopy(outGm[t * H], outBf16, H);
        outBf16Queue.FreeTensor(outBf16);

        swF32Queue.FreeTensor(swF32);
        tmpQueue.FreeTensor(tmp);
        outF32Queue.FreeTensor(outF32);
        reduceDstQueue.FreeTensor(reduceDst);
        reduceWorkQueue.FreeTensor(reduceWork);
        softmaxBufQueue.FreeTensor(softmaxBuf);
    }

private:
    uint32_t T;
    uint32_t B;
    uint32_t H;
    uint32_t blockDim;
    float eps;
    float invH;
    uint32_t rowStart;
    uint32_t rowEnd;

    GlobalTensor<bfloat16_t> blockResGm;
    GlobalTensor<bfloat16_t> prefixGm;
    GlobalTensor<bfloat16_t> scoreGm;
    GlobalTensor<bfloat16_t> outGm;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> vBf16Queue;
    TQue<QuePosition::VECIN, 1> swBf16Queue;
    TQue<QuePosition::VECCALC, 1> vF32Queue;
    TQue<QuePosition::VECCALC, 1> swF32Queue;
    TQue<QuePosition::VECCALC, 1> tmpQueue;
    TQue<QuePosition::VECCALC, 1> outF32Queue;
    TQue<QuePosition::VECOUT, 1> outBf16Queue;
    TQue<QuePosition::VECCALC, 1> reduceDstQueue;
    TQue<QuePosition::VECCALC, 1> reduceWorkQueue;
    TQue<QuePosition::VECCALC, 1> softmaxBufQueue;
};

extern "C" __global__ __aicore__ void attn_res(GM_ADDR blockRes, GM_ADDR prefixSum, GM_ADDR scoreWeight,
                                               GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AttnResTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelAttnRes op;
    op.Init(blockRes, prefixSum, scoreWeight, out, &tilingData);
    op.Process();
}
