// verify_situ_glu_fused_aclnn.cpp —— 融合算子（GEMM + SiTU epilogue）真机对拍 + bench
//
// 接口：aclnnSituGluFused(x[M,H], weight[2d,H], beta, linearBeta, out[M,d])
//   ⚠️ weight 必须是**交错重排后**的：W' = [gate_0, up_0, gate_1, up_1, ...]（每片 subN 行）
//      而 subN 由 kernel 的 tiling 决定 → 本程序先调一次触发 host 把 tiling 落盘，
//      读回真实 subN 再重排（否则 gate/up 配不上对，且**不会报错，只会算错**）。
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn_situ_glu_fused.h>
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

static int g_subN = 0, g_baseM = 0, g_baseN = 0, g_kPairs = 0;

static bool readTiling() {
    FILE* f = std::fopen("/tmp/situ_glu_fused_tiling.txt", "r");
    if (!f) return false;
    char buf[256];
    int baseK = 0, scm = 0, ucn = 0;
    while (std::fgets(buf, sizeof buf, f)) {
        std::sscanf(buf, "baseM=%d baseN=%d baseK=%d subN=%d kPairs=%d singleCoreM=%d usedCoreNum=%d",
                    &g_baseM, &g_baseN, &baseK, &g_subN, &g_kPairs, &scm, &ucn);
    }
    std::fclose(f);
    printf("[host tiling] baseM=%d baseN=%d baseK=%d subN=%d kPairs=%d\n",
           g_baseM, g_baseN, baseK, g_subN, g_kPairs);
    return g_subN > 0;
}

// 原始布局 W[2d,H]（前 d 行 gate、后 d 行 up）→ 交错 W'
static void interleave(const std::vector<uint16_t>& Wo, int64_t d, int64_t H, int subN,
                       std::vector<uint16_t>& Wp) {
    Wp.assign((size_t)2 * d * H, 0);
    for (int64_t k = 0; k * subN < d; ++k) {
        for (int64_t i = 0; i < subN && k * subN + i < d; ++i) {
            int64_t gRow = k * subN + i, uRow = d + k * subN + i;
            int64_t dG = 2 * k * subN + i, dU = (2 * k + 1) * subN + i;
            for (int64_t h = 0; h < H; ++h) {
                Wp[(size_t)dG * H + h] = Wo[(size_t)gRow * H + h];
                Wp[(size_t)dU * H + h] = Wo[(size_t)uRow * H + h];
            }
        }
    }
}

int main(int argc, char** argv) {
    const int64_t M = (argc > 1) ? atoll(argv[1]) : 8;
    const int64_t H = (argc > 2) ? atoll(argv[2]) : 32;
    const int64_t d = (argc > 3) ? atoll(argv[3]) : 64;
    const double beta = 4.0, linearBeta = 25.0;

    std::mt19937 rng(3);
    std::normal_distribution<float> dx(0.0f, 1.0f), dw(0.0f, 0.1f);
    std::vector<uint16_t> hx((size_t)M * H), hWo((size_t)2 * d * H), hy((size_t)M * d, 0);
    for (auto& v : hx) v = f32_to_bf16(dx(rng));
    for (auto& v : hWo) v = f32_to_bf16(dw(rng));

    CHECK(aclInit(nullptr), "aclInit");
    CHECK(aclrtSetDevice(0), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream), "aclrtCreateStream");

    void *dx_ = nullptr, *dw_ = nullptr, *dy_ = nullptr;
    CHECK(aclrtMalloc(&dx_, (size_t)M * H * 2, ACL_MEM_MALLOC_HUGE_FIRST), "malloc x");
    CHECK(aclrtMalloc(&dw_, (size_t)2 * d * H * 2, ACL_MEM_MALLOC_HUGE_FIRST), "malloc w");
    CHECK(aclrtMalloc(&dy_, (size_t)M * d * 2, ACL_MEM_MALLOC_HUGE_FIRST), "malloc y");
    CHECK(aclrtMemcpy(dx_, (size_t)M * H * 2, hx.data(), (size_t)M * H * 2, ACL_MEMCPY_HOST_TO_DEVICE), "H2D x");
    CHECK(aclrtMemset(dy_, (size_t)M * d * 2, 0, (size_t)M * d * 2), "memset y");

    int64_t xS[2] = {M, H}, xSt[2] = {H, 1};
    int64_t wS[2] = {2 * d, H}, wSt[2] = {H, 1};
    int64_t yS[2] = {M, d}, ySt[2] = {d, 1};
    aclTensor* xT = aclCreateTensor(xS, 2, ACL_BF16, xSt, 0, ACL_FORMAT_ND, xS, 2, dx_);
    aclTensor* wT = aclCreateTensor(wS, 2, ACL_BF16, wSt, 0, ACL_FORMAT_ND, wS, 2, dw_);
    aclTensor* yT = aclCreateTensor(yS, 2, ACL_BF16, ySt, 0, ACL_FORMAT_ND, yS, 2, dy_);
    if (!xT || !wT || !yT) { printf("[FAIL] aclCreateTensor\n"); return 1; }

    // ---- 第 1 次调用：只为触发 host 把 tiling 落盘（权重随便填）----
    {
        std::vector<uint16_t> dummy((size_t)2 * d * H, f32_to_bf16(0.0f));
        CHECK(aclrtMemcpy(dw_, (size_t)2 * d * H * 2, dummy.data(), (size_t)2 * d * H * 2,
                          ACL_MEMCPY_HOST_TO_DEVICE), "H2D dummy");
        uint64_t ws0 = 0; aclOpExecutor* e0 = nullptr;
        CHECK(aclnnSituGluFusedGetWorkspaceSize(xT, wT, beta, linearBeta, yT, &ws0, &e0), "GetWS(probe)");
        void* wsb = nullptr;
        if (ws0) CHECK(aclrtMalloc(&wsb, ws0, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ws(probe)");
        CHECK(aclnnSituGluFused(wsb, ws0, e0, stream), "exec(probe)");
        CHECK(aclrtSynchronizeStream(stream), "sync(probe)");
        if (wsb) aclrtFree(wsb);
    }
    if (!readTiling()) { printf("[FAIL] 没读到 host tiling（/tmp/situ_glu_fused_tiling.txt）\n"); return 1; }
    if (g_subN <= 0)   { printf("[FAIL] subN 非法: %d\n", g_subN); return 1; }

    // ---- 按真实 subN 重排权重，正式调用 ----
    std::vector<uint16_t> hWp;
    interleave(hWo, d, H, g_subN, hWp);
    CHECK(aclrtMemcpy(dw_, (size_t)2 * d * H * 2, hWp.data(), (size_t)2 * d * H * 2,
                      ACL_MEMCPY_HOST_TO_DEVICE), "H2D w'");
    uint64_t wsSize = 0; aclOpExecutor* exec = nullptr;
    CHECK(aclnnSituGluFusedGetWorkspaceSize(xT, wT, beta, linearBeta, yT, &wsSize, &exec), "GetWorkspaceSize");
    void* ws = nullptr;
    if (wsSize) CHECK(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ws");
    CHECK(aclnnSituGluFused(ws, wsSize, exec, stream), "exec");
    CHECK(aclrtSynchronizeStream(stream), "sync");
    CHECK(aclrtMemcpy(hy.data(), (size_t)M * d * 2, dy_, (size_t)M * d * 2, ACL_MEMCPY_DEVICE_TO_HOST), "D2H y");

    // ---- golden：用**原始布局**算（等价，且直观）----
    double maxAbs = 0.0, maxRel = 0.0;
    long long nBad = 0, nTotal = 0;
    for (int64_t r = 0; r < M; ++r) {
        for (int64_t c = 0; c < d; ++c) {
            double g = 0.0, u = 0.0;
            for (int64_t h = 0; h < H; ++h) {
                double xv = bf16_to_f32(hx[(size_t)r * H + h]);
                g += xv * bf16_to_f32(hWo[(size_t)c * H + h]);
                u += xv * bf16_to_f32(hWo[(size_t)(d + c) * H + h]);
            }
            double situ = beta * std::tanh(g / beta) * (1.0 / (1.0 + std::exp(-g)));
            double up2  = linearBeta * std::tanh(u / linearBeta);
            double gold = situ * up2;
            double got  = bf16_to_f32(hy[(size_t)r * d + c]);
            double err  = std::fabs(got - gold);
            double rel  = err / (std::fabs(gold) + 1e-3);
            double tol  = 1e-2 + 1e-2 * std::fabs(gold);
            if (err > maxAbs) maxAbs = err;
            if (rel > maxRel) maxRel = rel;
            if (err > tol) ++nBad;
            ++nTotal;
        }
    }
    printf("\nshape: M=%lld H=%lld d=%lld\n", (long long)M, (long long)H, (long long)d);
    printf("max_abs = %.3e   max_rel(纯相对) = %.3e\n", maxAbs, maxRel);
    printf("超容差 = %lld / %lld\n", nBad, nTotal);
    printf("判定(±1e-2+1e-2|ref|) : %s\n", nBad == 0 ? "✅ PASS" : "❌ FAIL");

    // ---- bench（仅当 shape 够大才有意义）----
    if (M >= 64) {
        for (int i = 0; i < 5; ++i) CHECK(aclnnSituGluFused(ws, wsSize, exec, stream), "warmup");
        CHECK(aclrtSynchronizeStream(stream), "sync warmup");
        const int iters = 50, rounds = 7;
        std::vector<double> rms;
        for (int r = 0; r < rounds; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) CHECK(aclnnSituGluFused(ws, wsSize, exec, stream), "exec");
            CHECK(aclrtSynchronizeStream(stream), "sync");
            auto t1 = std::chrono::high_resolution_clock::now();
            rms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
        }
        std::sort(rms.begin(), rms.end());
        double ms = rms[rms.size() / 2];
        // 融合算子的必要搬运：读 x + 读 W'(2d×H) + 写 out。权重只在 M 大时才被摊薄。
        const double wMb = (double)2 * d * H * 2 / 1048576.0;
        printf("\n单次耗时(中位): %.4f ms   （权重 %.3f MB，M=%lld 时权重读取占大头）\n",
               ms, wMb, (long long)M);
        printf("算力：%.2f GFLOP   → %.1f TFLOPS\n",
               2.0 * M * (2 * d) * H / 1e9, 2.0 * M * (2 * d) * H / (ms * 1e-3) / 1e12);
    }

    aclrtFree(dx_); aclrtFree(dw_); aclrtFree(dy_);
    if (ws) aclrtFree(ws);
    aclDestroyTensor(xT); aclDestroyTensor(wT); aclDestroyTensor(yT);
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return nBad == 0 ? 0 : 1;
}
