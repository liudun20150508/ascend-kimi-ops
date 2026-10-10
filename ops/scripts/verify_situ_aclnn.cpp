// verify_situ_aclnn.cpp —— 真机正确性对拍（纯 aclrt，不依赖 torch / torch_npu）
//
// 为什么需要它：test_situ_real.py 要 torch_npu，而本机装不上。
// 但"正确性"这条红线不能因为缺 Python 绑定就跳过 —— 用 C++ 直接对拍即可。
//
// 编译 / 运行见同目录 README 或 bench_situ_aclnn.cpp 顶部注释。
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn_situ_and_mul.h>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#define CHECK(expr, msg) do { \
    auto _r = (expr); \
    if (_r != ACL_SUCCESS) { printf("[FAIL] %s (ret=%d) @line %d\n", msg, (int)_r, __LINE__); return 1; } \
} while (0)

// bf16 <-> fp32（bf16 就是 fp32 的高 16 位）
static inline float bf16_to_f32(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float f; std::memcpy(&f, &u, 4); return f;
}
static inline uint16_t f32_to_bf16(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    return (uint16_t)((u + 0x8000u) >> 16);   // round-to-nearest（简化版）
}

int main(int argc, char** argv) {
    const int64_t rows = (argc > 1) ? atoll(argv[1]) : 64;
    const int64_t d    = (argc > 2) ? atoll(argv[2]) : 128;
    const double beta = 4.0, linearBeta = 25.0;
    const size_t inBytes = (size_t)rows * 2 * d * 2, outBytes = (size_t)rows * d * 2;

    // ---- host 侧造数据（bf16）----
    std::vector<uint16_t> hx(rows * 2 * d), hy(rows * d, 0);
    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.0f, 6.0f);   // 放大值域以覆盖 tanh/sigmoid 饱和区
    for (auto& v : hx) v = f32_to_bf16(dist(rng));

    CHECK(aclInit(nullptr), "aclInit");
    CHECK(aclrtSetDevice(0), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream), "aclrtCreateStream");

    void *xDev = nullptr, *yDev = nullptr;
    CHECK(aclrtMalloc(&xDev, inBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc x");
    CHECK(aclrtMalloc(&yDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc y");
    CHECK(aclrtMemcpy(xDev, inBytes, hx.data(), inBytes, ACL_MEMCPY_HOST_TO_DEVICE), "H2D x");
    CHECK(aclrtMemset(yDev, outBytes, 0, outBytes), "memset y");

    int64_t xShape[2] = {rows, 2 * d}, yShape[2] = {rows, d};
    int64_t xStr[2] = {2 * d, 1},      yStr[2] = {d, 1};
    aclTensor *xT = aclCreateTensor(xShape, 2, ACL_BF16, xStr, 0, ACL_FORMAT_ND, xShape, 2, xDev);
    aclTensor *yT = aclCreateTensor(yShape, 2, ACL_BF16, yStr, 0, ACL_FORMAT_ND, yShape, 2, yDev);
    if (!xT || !yT) { printf("[FAIL] aclCreateTensor\n"); return 1; }

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    CHECK(aclnnSituAndMulGetWorkspaceSize(xT, beta, linearBeta, yT, &wsSize, &exec), "GetWorkspaceSize");
    void *ws = nullptr;
    if (wsSize > 0) CHECK(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ws");
    CHECK(aclnnSituAndMul(ws, wsSize, exec, stream), "exec");
    CHECK(aclrtSynchronizeStream(stream), "sync");
    CHECK(aclrtMemcpy(hy.data(), outBytes, yDev, outBytes, ACL_MEMCPY_DEVICE_TO_HOST), "D2H y");

    // ---- host 侧 golden（与 reference/situ.py 同一套数学）----
    //   situ(gate) = beta * tanh(gate/beta) * sigmoid(gate)
    //   up'        = linearBeta * tanh(up/linearBeta)
    //   out        = situ(gate) * up'
    double maxAbs = 0.0, maxRel = 0.0;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < d; ++c) {
            double g = bf16_to_f32(hx[r * 2 * d + c]);
            double u = bf16_to_f32(hx[r * 2 * d + d + c]);
            double situ = beta * std::tanh(g / beta) * (1.0 / (1.0 + std::exp(-g)));
            double up2  = linearBeta * std::tanh(u / linearBeta);
            double gold = situ * up2;
            double got  = bf16_to_f32(hy[r * d + c]);
            double err  = std::fabs(got - gold);
            double rel  = err / (std::fabs(gold) + 1e-3);   // 带绝对项，避免跨零假失败
            if (err > maxAbs) maxAbs = err;
            if (rel > maxRel) maxRel = rel;
        }
    }

    printf("shape = [%lld, %lld]  d=%lld  beta=%.1f linear_beta=%.1f\n",
           (long long)rows, (long long)(2 * d), (long long)d, beta, linearBeta);
    printf("max_abs = %.3e\n", maxAbs);
    printf("max_rel = %.3e   (%s)\n", maxRel, maxRel < 1e-2 ? "✅ PASS（bf16 容差内）" : "❌ FAIL");
    printf("\n判据与 reference/situ.py 的 harness 一致：max_rel = |got-gold| / (|gold| + 1e-3)，阈值 1e-2。\n");

    aclrtFree(xDev); aclrtFree(yDev);
    if (ws) aclrtFree(ws);
    aclDestroyTensor(xT); aclDestroyTensor(yT);
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return maxRel < 1e-2 ? 0 : 1;
}
