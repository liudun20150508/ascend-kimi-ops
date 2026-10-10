// verify_attn_res_aclnn.cpp —— AttnRes 真机对拍 + benchmark（纯 aclrt，不依赖 torch）
//
// AttnRes 的数学（与 reference/attn_res.py 一致）：
//   v = cat([block_residual(T,B,H), prefix_sum(T,1,H)], dim=1)      # (T, B+1, H)
//   k = v * rsqrt(mean(v^2, -1) + eps)                              # RMS-norm（沿 H）
//   scores = sum(k * score_weight, -1)                              # (T, B+1)
//   probs  = softmax(scores, -1)
//   out    = sum(probs[:,:,None] * v, 1)                            # (T, H)
// 其中 score_weight 是 H 维常量（= norm.weight ⊙ proj.weight）
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn_attn_res.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(expr, msg) do { \
    auto _r = (expr); \
    if (_r != ACL_SUCCESS) { printf("[FAIL] %s (ret=%d) @line %d\n", msg, (int)_r, __LINE__); return 1; } \
} while (0)

static inline float bf16_to_f32(uint16_t v) {
    uint32_t u = (uint32_t)v << 16; float f; std::memcpy(&f, &u, 4); return f;
}
static inline uint16_t f32_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t)((u + 0x8000u) >> 16);
}

int main(int argc, char** argv) {
    const int64_t T = (argc > 1) ? atoll(argv[1]) : 16;
    const int64_t B = (argc > 2) ? atoll(argv[2]) : 8;
    const int64_t H = (argc > 3) ? atoll(argv[3]) : 4096;
    const double eps = 1e-6;
    const int64_t NB = B + 1;

    const size_t brBytes = (size_t)T * B * H * 2, psBytes = (size_t)T * H * 2;
    const size_t swBytes = (size_t)H * 2, outBytes = (size_t)T * H * 2;

    std::vector<uint16_t> hbr((size_t)T * B * H), hps((size_t)T * H), hsw((size_t)H), hout((size_t)T * H, 0);
    std::mt19937 rng(7);
    std::normal_distribution<float> d1(0.0f, 2.0f), d2(0.0f, 0.5f);
    for (auto& v : hbr) v = f32_to_bf16(d1(rng));
    for (auto& v : hps) v = f32_to_bf16(d1(rng));
    for (auto& v : hsw) v = f32_to_bf16(d2(rng));

    CHECK(aclInit(nullptr), "aclInit");
    CHECK(aclrtSetDevice(0), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream), "aclrtCreateStream");

    void *dbr = nullptr, *dps = nullptr, *dsw = nullptr, *dout = nullptr;
    CHECK(aclrtMalloc(&dbr, brBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc br");
    CHECK(aclrtMalloc(&dps, psBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ps");
    CHECK(aclrtMalloc(&dsw, swBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc sw");
    CHECK(aclrtMalloc(&dout, outBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc out");
    CHECK(aclrtMemcpy(dbr, brBytes, hbr.data(), brBytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D br");
    CHECK(aclrtMemcpy(dps, psBytes, hps.data(), psBytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D ps");
    CHECK(aclrtMemcpy(dsw, swBytes, hsw.data(), swBytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D sw");
    CHECK(aclrtMemset(dout, outBytes, 0, outBytes), "memset out");

    int64_t brS[3] = {T, B, H}, brSt[3] = {B * H, H, 1};
    int64_t psS[2] = {T, H},    psSt[2] = {H, 1};
    int64_t swS[1] = {H},       swSt[1] = {1};
    int64_t outS[2] = {T, H},   outSt[2] = {H, 1};
    aclTensor *brT = aclCreateTensor(brS, 3, ACL_BF16, brSt, 0, ACL_FORMAT_ND, brS, 3, dbr);
    aclTensor *psT = aclCreateTensor(psS, 2, ACL_BF16, psSt, 0, ACL_FORMAT_ND, psS, 2, dps);
    aclTensor *swT = aclCreateTensor(swS, 1, ACL_BF16, swSt, 0, ACL_FORMAT_ND, swS, 1, dsw);
    aclTensor *outT = aclCreateTensor(outS, 2, ACL_BF16, outSt, 0, ACL_FORMAT_ND, outS, 2, dout);
    if (!brT || !psT || !swT || !outT) { printf("[FAIL] aclCreateTensor\n"); return 1; }

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    CHECK(aclnnAttnResGetWorkspaceSize(brT, psT, swT, eps, outT, &wsSize, &exec), "GetWorkspaceSize");
    void *ws = nullptr;
    if (wsSize > 0) CHECK(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ws");

    CHECK(aclnnAttnRes(ws, wsSize, exec, stream), "exec");
    CHECK(aclrtSynchronizeStream(stream), "sync");
    CHECK(aclrtMemcpy(hout.data(), outBytes, dout, outBytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H out");

    // ---------------- golden ----------------
    double maxAbs = 0.0, maxRel = 0.0;
    long long nBad = 0, nTotal = 0;
    std::vector<double> v((size_t)NB * H);   // 注意：每个残差向量是 H 维，共 NB=B+1 个
    std::vector<double> sc(NB);
    std::vector<double> sw(H);
    for (int64_t h = 0; h < H; ++h) sw[h] = bf16_to_f32(hsw[h]);

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t b = 0; b < NB; ++b) {
            const uint16_t* src = (b < B) ? &hbr[((size_t)t * B + b) * H] : &hps[(size_t)t * H];
            double ss = 0.0, dot = 0.0;
            for (int64_t h = 0; h < H; ++h) {
                double x = bf16_to_f32(src[h]);
                v[b * H + h] = x;
                ss += x * x;
                dot += x * sw[h];
            }
            double var = ss / (double)H;
            sc[b] = (1.0 / std::sqrt(var + eps)) * dot;      // score
        }
        // softmax（数值稳定）
        double mx = sc[0];
        for (int64_t b = 1; b < NB; ++b) mx = std::max(mx, sc[b]);
        double sum = 0.0;
        for (int64_t b = 0; b < NB; ++b) { sc[b] = std::exp(sc[b] - mx); sum += sc[b]; }
        for (int64_t b = 0; b < NB; ++b) sc[b] /= sum;

        for (int64_t h = 0; h < H; ++h) {
            double gold = 0.0;
            for (int64_t b = 0; b < NB; ++b) gold += sc[b] * v[b * H + h];
            double got = bf16_to_f32(hout[(size_t)t * H + h]);
            double err = std::fabs(got - gold);
            double rel = err / (std::fabs(gold) + 1e-3);      // 保留：用来展示"纯相对误差"多夸张
            double tol = 1e-2 + 1e-2 * std::fabs(gold);        // 标准判据 atol + rtol·|ref|
            if (err > maxAbs) maxAbs = err;
            if (rel > maxRel) maxRel = rel;
            if (err > tol) ++nBad;
            ++nTotal;
        }
    }
    printf("shape: T=%lld B=%lld H=%lld (残差向量数 %lld)\n", (long long)T, (long long)B, (long long)H, (long long)NB);
    printf("max_abs        = %.3e    ← 这就是 bf16 的正常绝对精度（|x|~4 时 4×2^-8≈1.6e-2）\n", maxAbs);
    printf("max_rel(纯相对) = %.3e    ← 分母趋零时会被放大（docs/04 说的「假失败」）\n", maxRel);
    printf("超容差元素      = %lld / %lld  (%.4f%%)\n", nBad, nTotal, 100.0 * nBad / (double)nTotal);
    printf("判定(±1e-2+1e-2|ref|) : %s\n", nBad == 0 ? "✅ PASS" : "❌ FAIL");

    // ---------------- benchmark ----------------
    for (int i = 0; i < 5; ++i) CHECK(aclnnAttnRes(ws, wsSize, exec, stream), "warmup");
    CHECK(aclrtSynchronizeStream(stream), "sync warmup");
    const int iters = 50, rounds = 7;
    std::vector<double> rms;
    for (int r = 0; r < rounds; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) CHECK(aclnnAttnRes(ws, wsSize, exec, stream), "exec");
        CHECK(aclrtSynchronizeStream(stream), "sync");
        auto t1 = std::chrono::high_resolution_clock::now();
        rms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
    }
    std::sort(rms.begin(), rms.end());
    double ms = rms[rms.size() / 2];
    // 必要搬运：读一遍 v（T*(B+1)*H 个 bf16）+ 写 out（T*H 个 bf16）
    const size_t needed = (size_t)T * NB * H * 2 + (size_t)T * H * 2;
    const double PEAK = 1208.7;
    double bw = needed / (ms * 1e-3) / 1e9;
    printf("\n单次耗时(中位): %.4f ms  [min %.4f / max %.4f]\n", ms, rms.front(), rms.back());
    printf("必要搬运      : %.2f MB（读 v 一遍 + 写 out）\n", needed / 1048576.0);
    printf("理论下限      : %.4f ms  (÷ 实测带宽 %.1f GB/s)\n", needed / (PEAK * 1e9) * 1e3, PEAK);
    printf("有效带宽      : %.1f GB/s\n", bw);
    printf("★ 达标率      : %.1f%%   %s\n", bw / PEAK * 100,
           (bw / PEAK) >= 0.8 ? "达标" : (bw / PEAK) >= 0.6 ? "接近（60~80%）" : "未达标（<60%）");
    printf("\n注：kernel 内部 Pass1/Pass2 各读一遍 v（共 2 遍），所以「有效带宽」偏低是符合预期的——\n");
    printf("    这也正是 docs/04 §7 指出的 AttnRes 主要浪费点（分块/驻留是它的下一步优化方向）。\n");

    aclrtFree(dbr); aclrtFree(dps); aclrtFree(dsw); aclrtFree(dout);
    if (ws) aclrtFree(ws);
    aclDestroyTensor(brT); aclDestroyTensor(psT); aclDestroyTensor(swT); aclDestroyTensor(outT);
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return maxRel < 1e-2 ? 0 : 1;
}
