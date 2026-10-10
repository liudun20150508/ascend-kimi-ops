// SiTU-GLU (SituAndMul) 的 torch_extension 包装。
// 调用自定义 aclnn 算子 aclnnSituAndMul，输入 [..., 2d] -> 输出 [..., d]。
// 用 torch.utils.cpp_extension.load 直接 JIT 编译，绕开完整 package import。

#include <torch/extension.h>
#include "aclnn_common.h"
#include "aclnn_situ_and_mul.h"

namespace op_api {

at::Tensor situ_and_mul(const at::Tensor &x, double beta, double linear_beta)
{
    // x: [..., 2d] -> out: [..., d]
    auto sizes = x.sizes().vec();
    sizes.back() /= 2;
    auto out = at::empty(sizes, x.options());
    ACLNN_CMD(aclnnSituAndMul, x, beta, linear_beta, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("situ_and_mul", &situ_and_mul, "SituAndMul (aclnnSituAndMul wrapper)");
}

}  // namespace op_api
