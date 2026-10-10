// C++ 直调 CANN 内置 **SwiGlu**（`aclnnSwiGlu`）—— 与 torch 路径的 `npu_swiglu` 互证。
//
// 为什么补这一条：本项目自己的标准是「每个结论要有**两条独立路径**互证」。
// "相对华为"那格此前只有 torch 路径的数字（`torch_npu.npu_swiglu`），
// 缺 C++ 直调（不经 torch / torch_npu）的第二条路径 —— 本文件补上。
//
// 编译：
//   g++ -O2 -std=c++17 scripts/bench_swiglu_aclnn.cpp \
//     -I$ASCEND_HOME_PATH/include -L$ASCEND_HOME_PATH/lib64 \
//     -lopapi -lnnopbase -lascendcl -o /tmp/bench_swiglu_aclnn
//   ⚠️ 内置算子的 aclnn 实现在 libopapi.so（不是 libnnopbase.so）
//
// 运行：
//   /tmp/bench_swiglu_aclnn [rows] [d]     默认 4096 11008（K3 真实 intermediate_size）
//
// 语义：x [rows, 2d] → out [rows, d]，out = silu(x[..., :d]) * x[..., d:]   （SwiGLU）
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "aclnnop/level2/aclnn_swi_glu.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#define CHECK(expr, msg) do { \
    auto _r = (expr); \
    if (_r != ACL_SUCCESS) { printf("[FAIL] %s (ret=%d) @line %d\n", msg, (int)_r, __LINE__); return 1; } \
} while (0)

static const double PEAK_BW = 1271.6;   // 本卡实测峰值（GB/s，torch copy 口径）

int main(int argc, char** argv) {
    const int64_t rows = (argc > 1) ? atoll(argv[1]) : 4096;
    const int64_t d = (argc > 2) ? atoll(argv[2]) : 11008;   // K3 真实 intermediate_size
    const size_t inBytes = (size_t)rows * 2 * d * 2;          // bf16
    const size_t outBytes = (size_t)rows * d * 2;

    CHECK(aclInit(nullptr), "aclInit");
    CHECK(aclrtSetDevice(0), "device");
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream), "stream");

    void *xDev = nullptr, *yDev = nullptr;
    CHECK(aclrtMalloc(&xDev, inBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc x");
    CHECK(aclrtMalloc(&yDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc y");
    aclrtMemset(xDev, inBytes, 0x3C, inBytes);
    aclrtMemset(yDev, outBytes, 0x00, outBytes);

    int64_t xShape[2] = {rows, 2 * d}, yShape[2] = {rows, d};
    int64_t xStr[2] = {2 * d, 1}, yStr[2] = {d, 1};
    aclTensor* xT = aclCreateTensor(xShape, 2, ACL_BF16, xStr, 0, ACL_FORMAT_ND, xShape, 2, xDev);
    aclTensor* yT = aclCreateTensor(yShape, 2, ACL_BF16, yStr, 0, ACL_FORMAT_ND, yShape, 2, yDev);

    uint64_t wsSize = 0;
    aclOpExecutor* exec = nullptr;
    // SwiGlu：沿最后一维切两半，dim = -1；out = silu(前一半) * 后一半
    CHECK(aclnnSwiGluGetWorkspaceSize(xT, -1, yT, &wsSize, &exec), "SwiGlu GetWorkspaceSize");
    void* ws = nullptr;
    if (wsSize > 0) CHECK(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ws");

    printf("SwiGlu(aclnn)  shape=[%lld, %lld] d=%lld  输入 %.1f MB 输出 %.1f MB ws=%llu B\n",
           (long long)rows, (long long)(2 * d), (long long)d,
           inBytes / 1048576.0, outBytes / 1048576.0, (unsigned long long)wsSize);

    for (int i = 0; i < 5; ++i) CHECK(aclnnSwiGlu(ws, wsSize, exec, stream), "warmup");
    CHECK(aclrtSynchronizeStream(stream), "sync warmup");

    const int iters = 50, rounds = 7;
    std::vector<double> v;
    for (int r = 0; r < rounds; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) CHECK(aclnnSwiGlu(ws, wsSize, exec, stream), "exec");
        CHECK(aclrtSynchronizeStream(stream), "sync");
        auto t1 = std::chrono::high_resolution_clock::now();
        v.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
    }
    std::sort(v.begin(), v.end());
    const double ms = v[v.size() / 2];
    // 必要搬运：读 x(2d) + 写 out(d) = 3d 元素（与自研算子的严格口径 B 一致）
    const double moved = (double)rows * (2 * d + d) * 2;
    const double bw = moved / (ms * 1e-3) / 1e9;
    printf("单次耗时(中位): %.4f ms   [min %.4f / max %.4f, %d 轮×%d 次]\n",
           ms, v.front(), v.back(), rounds, iters);
    printf("严格口径搬运  : %.1f MB ⇒ 有效带宽 %.1f GB/s（峰值 %.1f，占比 %.1f%%）\n",
           moved / 1048576.0, bw, PEAK_BW, bw / PEAK_BW * 100);
    printf("★ 与 torch 路径（npu_swiglu）的数字互证：两条路径应同量级\n");

    aclrtFree(xDev); aclrtFree(yDev);
    if (ws) aclrtFree(ws);
    aclDestroyTensor(xT); aclDestroyTensor(yT);
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return 0;
}
