// baseline_twostage_aclnn.cpp —— 「不融合」两段式基线，用于量化 SituGluFused 的收益
//
// 两段式：gu  = x @ W2          (aclnnMatmul    → gu[M,2d] 落 HBM)
//         out = situ(gu)        (aclnnSituAndMul→ out[M,d])
// 融合版：out = SituGluFused(x, W'[2d,H], beta, linearBeta)   ← C 理想上不落 HBM（阶段 2）
//
// 目的：**同 shape、同计时口径**地量出「不融合」要花多少，这样融合省了多少才有分母。
//   · matmul-only / situ-only / 两段式合计 —— 三项分开测，便于拆解瓶颈
//   · 与 verify_situ_glu_fused_aclnn.cpp 用同一套计时（50 iters × 7 rounds 取中位）
//
// ⚠️ 权重排布差异（不影响性能，只影响数值对照）：
//     融合算子的 W' 是 [2d,H] 且 kernel 内 isTrans=true；
//     本程序直接用 [H,2d] 的 W2（数学等价：x@W'^T == x@W2），避免 host 侧再转置。
//
// 用法：
//   ./baseline2s [M] [H] [d]        默认 128 1024 11008
//   ./baseline2s 128 1024 11008 --nocheck   跳过 host golden（只为性能时用）
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnn_situ_and_mul.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// 50 iters × 7 rounds 取中位（与 verify_situ_glu_fused_aclnn.cpp 一致）
template <typename F>
static double benchMs(aclrtStream stream, F&& call, const char* tag) {
    int warmFail = 0, iterFail = 0; int firstRc = 0;
    for (int i = 0; i < 5; ++i) { int r = (int)call(); if (r != ACL_SUCCESS) { ++warmFail; if (!firstRc) firstRc = r; } }
    aclrtSynchronizeStream(stream);
    const int iters = 50, rounds = 7;
    std::vector<double> rms;
    for (int r = 0; r < rounds; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) { int rc = (int)call(); if (rc != ACL_SUCCESS) { ++iterFail; if (!firstRc) firstRc = rc; } }
        aclrtSynchronizeStream(stream);
        auto t1 = std::chrono::high_resolution_clock::now();
        rms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
    }
    std::sort(rms.begin(), rms.end());
    double ms = rms[rms.size() / 2];
    printf("  %-22s 中位 %8.4f ms   [min %.4f / max %.4f]", tag, ms, rms.front(), rms.back());
    if (warmFail || iterFail) printf("   ⚠️ 失败调用: warmup %d / bench %d（首个错误码 %d）", warmFail, iterFail, firstRc);
    printf("\n");
    return ms;
}

int main(int argc, char** argv) {
    const int64_t M = (argc > 1) ? atoll(argv[1]) : 128;
    const int64_t H = (argc > 2) ? atoll(argv[2]) : 1024;
    const int64_t d = (argc > 3) ? atoll(argv[3]) : 11008;
    bool doCheck = true;
    for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "--nocheck")) doCheck = false;

    const double beta = 4.0, linearBeta = 25.0;
    const size_t xBytes  = (size_t)M * H * 2;
    const size_t wBytes  = (size_t)H * 2 * d * 2;   // W2[H,2d] bf16
    const size_t guBytes = (size_t)M * 2 * d * 2;   // gu[M,2d] bf16（两段式：落 HBM）
    const size_t yBytes  = (size_t)M * d * 2;

    std::vector<uint16_t> hx((size_t)M * H), hw((size_t)H * 2 * d), hy((size_t)M * d, 0);
    std::mt19937 rng(20260919);
    std::normal_distribution<float> dx_(0.0f, 1.0f), dw_(0.0f, 0.02f);
    for (auto& v : hx) v = f32_to_bf16(dx_(rng));
    for (auto& v : hw) v = f32_to_bf16(dw_(rng));

    CHECK(aclInit(nullptr), "aclInit");
    CHECK(aclrtSetDevice(0), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream), "aclrtCreateStream");

    void *dX = nullptr, *dW = nullptr, *dGu = nullptr, *dY = nullptr;
    CHECK(aclrtMalloc(&dX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc x");
    CHECK(aclrtMalloc(&dW, wBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc w");
    CHECK(aclrtMalloc(&dGu, guBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc gu");
    CHECK(aclrtMalloc(&dY, yBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc y");
    CHECK(aclrtMemcpy(dX, xBytes, hx.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D x");
    CHECK(aclrtMemcpy(dW, wBytes, hw.data(), wBytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D w");
    CHECK(aclrtMemset(dGu, guBytes, 0, guBytes), "memset gu");
    CHECK(aclrtMemset(dY, yBytes, 0, yBytes), "memset y");

    // ---------- 张量：x[M,H] × W2[H,2d] → gu[M,2d]；situ: gu → y[M,d] ----------
    int64_t xS[2]  = {M, H},   xSt[2]  = {H, 1};
    int64_t wS[2]  = {H, 2 * d}, wSt[2] = {2 * d, 1};
    int64_t gS[2]  = {M, 2 * d}, gSt[2] = {2 * d, 1};
    int64_t yS[2]  = {M, d},   ySt[2]  = {d, 1};
    aclTensor* xT   = aclCreateTensor(xS, 2, ACL_BF16, xSt, 0, ACL_FORMAT_ND, xS, 2, dX);
    aclTensor* wT   = aclCreateTensor(wS, 2, ACL_BF16, wSt, 0, ACL_FORMAT_ND, wS, 2, dW);
    aclTensor* guT  = aclCreateTensor(gS, 2, ACL_BF16, gSt, 0, ACL_FORMAT_ND, gS, 2, dGu);
    aclTensor* yT   = aclCreateTensor(yS, 2, ACL_BF16, ySt, 0, ACL_FORMAT_ND, yS, 2, dY);
    if (!xT || !wT || !guT || !yT) { printf("[FAIL] aclCreateTensor\n"); return 1; }

    uint64_t wsMM = 0, wsSM = 0;
    aclOpExecutor *exMM = nullptr, *exSM = nullptr;
    CHECK(aclnnMatmulGetWorkspaceSize(xT, wT, guT, 0, &wsMM, &exMM), "Matmul GetWorkspaceSize");
    CHECK(aclnnSituAndMulGetWorkspaceSize(guT, beta, linearBeta, yT, &wsSM, &exSM), "SituAndMul GetWorkspaceSize");
    void *wsM = nullptr, *wsS = nullptr;
    if (wsMM) CHECK(aclrtMalloc(&wsM, wsMM, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ws matmul");
    if (wsSM) CHECK(aclrtMalloc(&wsS, wsSM, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ws situ");

    // ★ 实测（2026-09-19）：aclnnMatmul 的 executor **不能复用** ——
    //   GetWorkspaceSize 一次 + 多次 exec 的写法，第 2 次起全部返回 161001（PARAM_NULLPTR），
    //   表现为耗时塌到 ~3 μs 的假数字。改成「每次调用重新 GetWorkspaceSize」（torch dispatch 的做法）。
    //   自定义算子（libcust_opapi）无此问题：aclnnSituAndMul 连跑 350 次全成功。
    auto runMM = [&]() -> aclnnStatus {
        uint64_t ws = 0; aclOpExecutor* ex = nullptr;
        aclnnStatus rc = aclnnMatmulGetWorkspaceSize(xT, wT, guT, 0, &ws, &ex);
        if (rc != ACL_SUCCESS) return rc;
        return aclnnMatmul(wsM, ws, ex, stream);
    };
    auto runSM = [&]() -> aclnnStatus { return aclnnSituAndMul(wsS, wsSM, exSM, stream); };

    printf("[diag] workspace: matmul=%llu B, situ=%llu B\n",
           (unsigned long long)wsMM, (unsigned long long)wsSM);

    // ---------- 一次性正确性（host golden：fp32 累加 → situ）----------
    if (doCheck && M <= 256) {
        CHECK(runMM(), "exec matmul"); CHECK(runSM(), "exec situ");
        CHECK(aclrtSynchronizeStream(stream), "sync");
        CHECK(aclrtMemcpy(hy.data(), yBytes, dY, yBytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H y");
        // diag：matmul 到底写没写 gu？
        std::vector<uint16_t> hgu((size_t)M * 2 * d, 0);
        CHECK(aclrtMemcpy(hgu.data(), guBytes, dGu, guBytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H gu");
        double guSum = 0.0, guAbsMax = 0.0;
        for (size_t i = 0; i < hgu.size(); ++i) { double v = bf16_to_f32(hgu[i]); guSum += v; guAbsMax = std::max(guAbsMax, std::fabs(v)); }
        printf("[diag] gu[M,2d] 均值=%.6f  |max|=%.6f  （若全 0 ⇒ matmul 没写进去）\n",
               guSum / (double)hgu.size(), guAbsMax);
        printf("[diag] y[0..3] = %.5f %.5f %.5f %.5f\n", bf16_to_f32(hy[0]), bf16_to_f32(hy[1]),
               bf16_to_f32(hy[2]), bf16_to_f32(hy[3]));
        double maxRel = 0.0; long long nBad = 0;
        for (int64_t r = 0; r < M; ++r) {
            for (int64_t c = 0; c < d; ++c) {
                double g = 0.0, u = 0.0;
                for (int64_t h = 0; h < H; ++h) {
                    double xv = (double)bf16_to_f32(hx[(size_t)r * H + h]);
                    g += xv * (double)bf16_to_f32(hw[(size_t)h * 2 * d + c]);        // gate = x @ W2[:, c]
                    u += xv * (double)bf16_to_f32(hw[(size_t)h * 2 * d + d + c]);    // up   = x @ W2[:, d+c]
                }
                double situ = beta * std::tanh(g / beta) * (1.0 / (1.0 + std::exp(-g)));
                double up2  = linearBeta * std::tanh(u / linearBeta);
                double gold = situ * up2;                                            // ★ 漏了 up 就会全错
                double got  = bf16_to_f32(hy[(size_t)r * d + c]);
                double err  = std::fabs(got - gold);
                if (err > 1e-2 + 1e-2 * std::fabs(gold)) ++nBad;
                maxRel = std::max(maxRel, err / (std::fabs(gold) + 1e-3));
            }
        }
        printf("正确性 sanity（两段式 vs host golden）: max_rel=%.3e  超容差=%lld/%lld  %s\n",
               maxRel, nBad, (long long)M * d, nBad == 0 ? "✅ PASS" : "❌ FAIL");
    } else if (doCheck) {
        printf("（M=%lld 太大，跳过 host golden；正确性见 verify_situ_glu_fused_aclnn.cpp）\n", (long long)M);
    }

    // ---------- bench ----------
    printf("\n=== shape M=%lld H=%lld d=%lld ===\n", (long long)M, (long long)H, (long long)d);
    double tMM  = benchMs(stream, runMM, "matmul-only");
    double tSM  = benchMs(stream, runSM, "situ-only");
    double t2S  = benchMs(stream, [&]() -> aclnnStatus {
                      auto a = runMM(); if (a != ACL_SUCCESS) return a; return runSM();
                  }, "两段式合计");
    CHECK(aclrtSynchronizeStream(stream), "final sync");

    // ---------- 搬运量账（bf16 口径）----------
    const double MB = 1048576.0;
    double wMB  = (double)wBytes / MB;
    double guMB = (double)guBytes / MB;
    double xMB  = (double)xBytes / MB;
    double yMB  = (double)yBytes / MB;
    // 两段式必要搬运：读 x + 读 W2 + 写 gu + 读 gu + 写 y
    double twoStageMB = xMB + wMB + 2 * guMB + yMB;
    // 融合（理想，阶段 2，C 不落 HBM）：读 x + 读 W + 写 y
    double fusedIdealMB = xMB + wMB + yMB;
    double bw = 1208.7;   // GB/s，与 docs/14 同口径
    printf("\n--- 搬运量账（bf16）---\n");
    printf("  x %.2f MB + W %.2f MB + y %.2f MB ；gu（中间张量）%.2f MB\n", xMB, wMB, yMB, guMB);
    printf("  两段式  搬运 %.2f MB → 理论下限 %7.1f μs\n", twoStageMB, twoStageMB / bw * 1000.0);
    printf("  融合理想搬运 %.2f MB → 理论下限 %7.1f μs   （省掉 gu 写+读 = %.2f MB）\n",
           fusedIdealMB, fusedIdealMB / bw * 1000.0, 2 * guMB);
    printf("  ★ 中间张量往返占两段式实测的比例 : %.1f%%（%.4f ms / %.4f ms）\n",
           100.0 * (2 * guMB / bw * 1000.0) / (t2S * 1000.0), (2 * guMB / bw * 1000.0) / 1000.0, t2S);

    aclrtFree(dX); aclrtFree(dW); aclrtFree(dGu); aclrtFree(dY);
    if (wsM) aclrtFree(wsM);
    if (wsS) aclrtFree(wsS);
    aclDestroyTensor(xT); aclDestroyTensor(wT); aclDestroyTensor(guT); aclDestroyTensor(yT);
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return 0;
}
