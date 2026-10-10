// 真机直调 aclnnSituAndMul（不依赖 torch / torch_npu）
// 这在「Python 绑定缺失」时是唯一能拿到真实性能数字的路。
//
// 编译：
//   g++ -O2 -std=c++17 bench_situ_aclnn.cpp \
//     -I$ASCEND_HOME_PATH/include \
//     -I$ASCEND_HOME_PATH/opp/vendors/customize/op_api/include \
//     -L$ASCEND_HOME_PATH/lib64 \
//     -L$ASCEND_HOME_PATH/opp/vendors/customize/op_api/lib \
//     -lascendcl -lcust_opapi -o bench_situ_aclnn
// 运行前：
//   export ASCEND_CUSTOM_OPP_PATH=$ASCEND_HOME_PATH/opp/vendors/customize
//   export LD_LIBRARY_PATH=$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnn_situ_and_mul.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>

#define CHECK(expr, msg) do { \
    auto _r = (expr); \
    if (_r != ACL_SUCCESS) { printf("[FAIL] %s (ret=%d) @line %d\n", msg, (int)_r, __LINE__); return 1; } \
} while (0)

static double g_peak_bw = 1208.7;   // 实测 D2D 带宽（GB/s），见 bw_test.cpp

int main(int argc, char** argv) {
    // KIMI3 真实 shape：rows 个 token，最后一维 2d（前半 gate、后半 up）
    const int64_t rows = (argc > 1) ? atoll(argv[1]) : 3584;
    const int64_t d    = (argc > 2) ? atoll(argv[2]) : 11008;
    const size_t inBytes  = (size_t)rows * 2 * d * 2;   // bf16
    const size_t outBytes = (size_t)rows * d * 2;

    CHECK(aclInit(nullptr), "aclInit");
    CHECK(aclrtSetDevice(0), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream), "aclrtCreateStream");

    void *xDev = nullptr, *yDev = nullptr;
    CHECK(aclrtMalloc(&xDev, inBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc x");
    CHECK(aclrtMalloc(&yDev, outBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc y");
    aclrtMemset(xDev, inBytes, 0x3C, inBytes);          // 填点值（不校验数值，只看性能）

    int64_t xShape[2] = {rows, 2 * d}, yShape[2] = {rows, d};
    int64_t xStr[2] = {2 * d, 1},      yStr[2] = {d, 1};
    aclTensor *xT = aclCreateTensor(xShape, 2, ACL_BF16, xStr, 0, ACL_FORMAT_ND, xShape, 2, xDev);
    aclTensor *yT = aclCreateTensor(yShape, 2, ACL_BF16, yStr, 0, ACL_FORMAT_ND, yShape, 2, yDev);
    if (!xT || !yT) { printf("[FAIL] aclCreateTensor\n"); return 1; }

    uint64_t wsSize = 0;
    aclOpExecutor *exec = nullptr;
    CHECK(aclnnSituAndMulGetWorkspaceSize(xT, 4.0, 25.0, yT, &wsSize, &exec), "GetWorkspaceSize");
    void *ws = nullptr;
    if (wsSize > 0) CHECK(aclrtMalloc(&ws, wsSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc ws");

    printf("shape = [%lld, %lld] (d=%lld)  输入 %.1f MB  输出 %.1f MB  workspace %llu B\n",
           (long long)rows, (long long)(2 * d), (long long)d,
           inBytes / 1048576.0, outBytes / 1048576.0, (unsigned long long)wsSize);

    for (int i = 0; i < 5; ++i) CHECK(aclnnSituAndMul(ws, wsSize, exec, stream), "warmup exec");
    CHECK(aclrtSynchronizeStream(stream), "sync warmup");

    const int iters = 50, rounds = 7;
    std::vector<double> rms;
    for (int r = 0; r < rounds; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i) CHECK(aclnnSituAndMul(ws, wsSize, exec, stream), "exec");
        CHECK(aclrtSynchronizeStream(stream), "sync bench");
        auto t1 = std::chrono::high_resolution_clock::now();
        rms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count() / iters);
    }
    std::sort(rms.begin(), rms.end());
    double ms = rms[rms.size() / 2];
    double ms_min = rms.front(), ms_max = rms.back();
    // 必要搬运：读 gate(2d) + 读 up(2d) + 写 out(d)，bf16
    const size_t moved = (size_t)rows * (2 * d + 2 * d + d) * 2;
    double bw = moved / (ms * 1e-3) / 1e9;
    double lower = moved / (g_peak_bw * 1e9) * 1e3;

    printf("\n单次耗时(中位): %.4f ms  [min %.4f / max %.4f, %d 轮×%d 次]\n", ms, ms_min, ms_max, rounds, iters);
    printf("理论下限      : %.4f ms  (搬运 %.1f MB ÷ 实测带宽 %.1f GB/s)\n",
           lower, moved / 1048576.0, g_peak_bw);
    printf("有效带宽      : %.1f GB/s\n", bw);
    printf("★ 带宽达标率  : %.1f%%   %s\n", bw / g_peak_bw * 100,
           (bw / g_peak_bw) >= 0.8 ? "达标" : (bw / g_peak_bw) >= 0.6 ? "接近（60~80%）" : "未达标（<60%）");

    aclrtFree(xDev); aclrtFree(yDev);
    if (ws) aclrtFree(ws);
    aclDestroyTensor(xT); aclDestroyTensor(yT);
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return 0;
}
