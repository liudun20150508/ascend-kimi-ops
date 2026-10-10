// 真机直调 CANN 内置 **FFN 融合算子**（`aclnnFFNV3`），并与「不融合三段」对比
//   ⇒ 目标：量出「华为的整层 MLP 融合」在这台机器上值多少 —— 图级收益对比的参照系。
//
// 华为 `FFN` 语义（头文件 aclnn_ffn_v3.h）：
//     y = activation(xW1 + b1) W2 + b2      （weight 无专家形态即 dense MLP）
//     activation ∈ {fastgelu, gelu, relu, silu, geglu, swiglu, reglu}   ← **不含 SiTU**
//
// 编译：
//   g++ -O2 -std=c++17 scripts/bench_ffn_aclnn.cpp \
//     -I$ASCEND_HOME_PATH/include -L$ASCEND_HOME_PATH/lib64 \
//     -lopapi -lnnopbase -lascendcl -o /tmp/bench_ffn_aclnn
//   ⚠️ 坑：内置算子的 aclnn 实现在 **libopapi.so**（不是 libnnopbase.so），不加会 undefined reference
//
// 运行：
//   /tmp/bench_ffn_aclnn [M] [K] [N] [activation] [use_e1]
//     use_e1=1 → 走 **E=1 的 MoE 形态**（weight 带专家维）
//
// ★ 形态与 dtype 的实测约束（撞出来的）：
//   - **dense 形态（无 expertTokens）只支持 fp16**：bf16 报
//     "x only support dtype float16 without expert tokens, but found DT_BFLOAT16"
//   - 想用 **bf16**，就得走"有 expertTokens"的形态 ⇒ 用 **E=1**（等价 dense FFN，但支持 bf16）
//   - GLU 形态要求 `innerPrecise=1`（传 0 报 "Detected glu, inner precision is not right"）
//
// ⚠️ 自我纠错记录：我一度判定"本机 SOC 没有 FFN kernel"，**那是错的（查法错）** ——
//   内置 kernel 按分类目录组织：kernel/<soc>/ops_transformer/ffn/，不是按算子名放顶层。
//   按同样错查法 swi_glu / matmul 也"查不到"，但它们都能跑。正确查法可确认 FFN 的 kernel 就在本机。
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "aclnnop/level2/aclnn_ffn_v3.h"
#include "aclnnop/level2/aclnn_matmul.h"
#include "aclnnop/level2/aclnn_swi_glu.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define CHECK(expr, msg) do { \
    auto _r = (expr); \
    if (_r != ACL_SUCCESS) { printf("[FAIL] %s (ret=%d) @line %d\n", msg, (int)_r, __LINE__); return 1; } \
} while (0)

static const double PEAK_BW = 1271.6;   // 本卡实测峰值带宽（GB/s）

static aclTensor* mkTensor(void* dev, std::vector<int64_t> shape, aclDataType dt) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) strides[i] = strides[i + 1] * shape[i + 1];
    return aclCreateTensor(shape.data(), (uint64_t)shape.size(), dt, strides.data(), 0,
                           ACL_FORMAT_ND, shape.data(), (uint64_t)shape.size(), dev);
}

int main(int argc, char** argv) {
    const int64_t M = (argc > 1) ? atoll(argv[1]) : 128;
    const int64_t K = (argc > 2) ? atoll(argv[2]) : 7168;    // hidden（K3 = 7168）
    const int64_t N = (argc > 3) ? atoll(argv[3]) : 11008;   // intermediate（K3 = 11008）
    const std::string act = (argc > 4) ? argv[4] : "swiglu";
    const bool use_e1 = (argc > 5) ? (atoll(argv[5]) != 0) : false;
    const bool is_glu = act.find("glu") != std::string::npos;
    const int64_t N1 = is_glu ? 2 * N : N;

    // dtype：dense 形态只支持 fp16；带 expertTokens（E=1）时可用 bf16
    const aclDataType DT = use_e1 ? ACL_BF16 : ACL_FLOAT16;

    CHECK(aclInit(nullptr), "aclInit");
    CHECK(aclrtSetDevice(0), "device");
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream), "stream");

    const size_t bX = (size_t)M * K * 2, bW1 = (size_t)K * N1 * 2, bW2 = (size_t)N * K * 2;
    const size_t bG = (size_t)M * N1 * 2, bU = (size_t)M * N * 2, bY = (size_t)M * K * 2;
    void *x, *w1, *w2, *g, *u, *y, *y2;
    auto alloc = [](void** p, size_t bytes) { return aclrtMalloc(p, bytes, ACL_MEM_MALLOC_HUGE_FIRST); };
    CHECK(alloc(&x, bX), "x"); CHECK(alloc(&w1, bW1), "w1"); CHECK(alloc(&w2, bW2), "w2");
    CHECK(alloc(&g, bG), "g"); CHECK(alloc(&u, bU), "u");
    CHECK(alloc(&y, bY), "y"); CHECK(alloc(&y2, bY), "y2");
    aclrtMemset(x, bX, 0x3C, bX); aclrtMemset(w1, bW1, 0x3C, bW1); aclrtMemset(w2, bW2, 0x3C, bW2);
    aclrtMemset(g, bG, 0x3C, bG); aclrtMemset(u, bU, 0x3C, bU);
    aclrtMemset(y, bY, 0x00, bY); aclrtMemset(y2, bY, 0x00, bY);

    printf("shape: M=%lld K=%lld N=%lld  act=%s  is_glu=%d N1=%lld  form=%s  dtype=%s\n",
           (long long)M, (long long)K, (long long)N, act.c_str(), (int)is_glu, (long long)N1,
           use_e1 ? "MoE(E=1)" : "dense", use_e1 ? "bf16" : "fp16");

    aclTensor* tX = mkTensor(x, {M, K}, DT);
    aclTensor* tW1 = use_e1 ? mkTensor(w1, {1, K, N1}, DT) : mkTensor(w1, {K, N1}, DT);
    aclTensor* tW2 = use_e1 ? mkTensor(w2, {1, N, K}, DT) : mkTensor(w2, {N, K}, DT);
    aclTensor* tG = mkTensor(g, {M, N1}, DT);
    aclTensor* tU = mkTensor(u, {M, N}, DT);
    aclTensor* tY = mkTensor(y, {M, K}, DT);
    aclTensor* tY2 = mkTensor(y2, {M, K}, DT);

    // E=1 形态需要一个 expertTokens 张量（各专家 token 数，int64）
    aclTensor* tET = nullptr;
    void* etDev = nullptr;
    if (use_e1) {
        int64_t etHost[1] = { M };
        CHECK(alloc(&etDev, sizeof(int64_t)), "et");
        CHECK(aclrtMemcpy(etDev, sizeof(int64_t), etHost, sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE), "et cpy");
        tET = mkTensor(etDev, {1}, ACL_INT64);
    }

    auto bench = [&stream](std::function<aclnnStatus()> fn, int iters = 50, int rounds = 7) {
        for (int i = 0; i < 5; ++i) fn();
        aclrtSynchronizeStream(stream);
        std::vector<double> v;
        for (int r = 0; r < rounds; ++r) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < iters; ++i) fn();
            aclrtSynchronizeStream(stream);
            auto t1 = std::chrono::high_resolution_clock::now();
            v.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
        }
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    // ---------------- ① 融合：内置 FFN 一次调用 ----------------
    uint64_t ws1 = 0;
    aclOpExecutor* e1 = nullptr;
    CHECK(aclnnFFNV3GetWorkspaceSize(tX, tW1, tW2, tET, nullptr, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                                     act.c_str(), 0, false, tY, &ws1, &e1), "FFN GetWorkspaceSize");
    printf("FFN workspace = %llu B\n", (unsigned long long)ws1);
    void* wsp1 = nullptr;
    if (ws1 > 0) CHECK(alloc(&wsp1, ws1), "ws1");
    double t_fused = bench([&] { return aclnnFFNV3(wsp1, ws1, e1, stream); });
    const double bytes_fused = (double)(bX + bW1 + bW2 + bY);
    printf("① 融合（aclnnFFNV3 %s）: %8.4f ms   等效搬运 %.1f MB ⇒ %.1f GB/s\n",
           act.c_str(), t_fused, bytes_fused / 1048576.0, bytes_fused / (t_fused * 1e-3) / 1e9);

    // ---------------- ② 不融合：matmul + glu + matmul（都用 CANN 内置算子）----------------
    if (is_glu) {
        uint64_t wm1 = 0, wsg = 0, wm2 = 0;
        aclOpExecutor *em1 = nullptr, *esg = nullptr, *em2 = nullptr;
        aclTensor* tW1m = mkTensor(w1, {K, N1}, DT);      // matmul 用 2 维权重
        CHECK(aclnnMatmulGetWorkspaceSize(tX, tW1m, tG, 0, &wm1, &em1), "matmul1");
        CHECK(aclnnSwiGluGetWorkspaceSize(tG, -1, tU, &wsg, &esg), "swiglu");
        aclTensor* tW2m = mkTensor(w2, {N, K}, DT);
        CHECK(aclnnMatmulGetWorkspaceSize(tU, tW2m, tY2, 0, &wm2, &em2), "matmul2");
        void *pm1 = nullptr, *psg = nullptr, *pm2 = nullptr;
        if (wm1) CHECK(alloc(&pm1, wm1), "pm1");
        if (wm2) CHECK(alloc(&pm2, wm2), "pm2");
        if (wsg) CHECK(alloc(&psg, wsg), "psg");
        double t_un = bench([&] {
            aclnnMatmul(pm1, wm1, em1, stream);
            aclnnSwiGlu(psg, wsg, esg, stream);
            return aclnnMatmul(pm2, wm2, em2, stream);
        });
        const double saved = (double)M * (N1 + N1 + N + N) * 2;   // 写g+读g+写u+读u
        printf("② 不融合（matmul+swiglu+matmul）: %8.4f ms\n", t_un);
        printf("⇒ 融合收益                      : %8.4f ms (%+.1f%%)\n", t_un - t_fused,
               (t_un - t_fused) / t_un * 100.0);
        printf("   省掉的中间张量往返              : %.2f MB/次（%.1f KB/token）\n",
               saved / 1048576.0, saved / M / 1024.0);
        printf("   按带宽折算（@%.0f GB/s）        : %8.4f ms\n", PEAK_BW, saved / (PEAK_BW * 1e9) * 1e3);
    }

    printf("\n⚠️ 这是**华为的融合**（%s，整层 MLP）；我的 SituGluFused 是**半个 MLP**\n", act.c_str());
    printf("   （gate_up GEMM + SiTU，down 在外面，账面省 4d vs 华为 6d）；两者激活不同，\n");
    printf("   比的是**融合收益的幅度**，不是谁算得快。\n");

    aclDestroyTensor(tX); aclDestroyTensor(tW1); aclDestroyTensor(tW2);
    aclDestroyTensor(tG); aclDestroyTensor(tU); aclDestroyTensor(tY); aclDestroyTensor(tY2);
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return 0;
}
