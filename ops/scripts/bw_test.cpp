// HBM 带宽实测（纯 aclrt，不依赖 torch_npu）
// 编译：g++ -O2 -std=c++17 bw_test.cpp -I$ASCEND_HOME_PATH/include -L$ASCEND_HOME_PATH/lib64 -lascendcl -o bw_test
#include <acl/acl.h>
#include <chrono>
#include <cstdio>
#include <cstdint>

static const char* kindName(aclrtMemcpyKind k) {
    switch (k) {
        case ACL_MEMCPY_HOST_TO_DEVICE: return "H2D";
        case ACL_MEMCPY_DEVICE_TO_HOST: return "D2H";
        case ACL_MEMCPY_DEVICE_TO_DEVICE: return "D2D";
        default: return "?";
    }
}

int main() {
    if (aclInit(nullptr) != ACL_SUCCESS) { printf("aclInit 失败\n"); return 1; }
    if (aclrtSetDevice(0) != ACL_SUCCESS) { printf("aclrtSetDevice 失败\n"); return 1; }
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) { printf("aclrtCreateStream 失败\n"); return 1; }

    const size_t n = 256ull * 1024 * 1024;   // 256 MB
    void *d_a = nullptr, *d_b = nullptr, *h = nullptr;
    if (aclrtMalloc(&d_a, n, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { printf("malloc d_a 失败\n"); return 1; }
    if (aclrtMalloc(&d_b, n, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) { printf("malloc d_b 失败\n"); return 1; }
    if (aclrtMallocHost(&h, n) != ACL_SUCCESS) { printf("mallocHost 失败\n"); return 1; }
    aclrtMemset(d_a, n, 0x5A, n);

    struct Case { aclrtMemcpyKind kind; void* dst; void* src; const char* label; };
    Case cases[] = {
        { ACL_MEMCPY_DEVICE_TO_DEVICE, d_b, d_a, "D2D copy (≈HBM 读写)" },
        { ACL_MEMCPY_DEVICE_TO_HOST,   h,   d_a, "D2H        (≈HBM 读)"  },
        { ACL_MEMCPY_HOST_TO_DEVICE,   d_a, h,   "H2D        (≈HBM 写)"  },
    };

    printf("%-24s %10s %14s %14s\n", "case", "ms/iter", "GB/s(单向)", "GB/s(读+写口径)");
    for (auto& c : cases) {
        for (int i = 0; i < 5; ++i)
            aclrtMemcpyAsync(c.dst, n, c.src, n, c.kind, stream);
        aclrtSynchronizeStream(stream);

        const int iters = 20;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iters; ++i)
            aclrtMemcpyAsync(c.dst, n, c.src, n, c.kind, stream);
        aclrtSynchronizeStream(stream);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
        double one_way = (double)n / (ms * 1e-3) / 1e9;
        double both = one_way * 2;   // memcpy 实际是读 + 写
        printf("%-24s %10.3f %14.1f %14.1f\n", c.label, ms, one_way, both);
    }

    printf("\n说明：D2D 那行 GB/s(读+写) 就是 HBM 的可用带宽量级；\n");
    printf("      用它做分母即可把「省的搬运字节数」换算成时间。\n");
    aclrtFree(d_a); aclrtFree(d_b); aclrtFreeHost(h);
    aclrtDestroyStream(stream); aclrtResetDevice(0); aclFinalize();
    return 0;
}
