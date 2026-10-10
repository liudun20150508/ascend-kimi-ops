// SituGluFused（融合算子：GEMM + SiTU epilogue）的 torch_extension 包装。
// 调用自定义 aclnn 算子 aclnnSituGluFused：x[M,H] + weight[2d,H] -> out[M,d]。
//
// ⚠️ 与 aclnn 语义一致：**weight 必须是交错重排后的 W'**
//     （W'[2k·subN+i] = gate[k·subN+i]，W'[(2k+1)·subN+i] = up[k·subN+i]）。
//     重排工具见 interleave_weight.py；忘了重排**不会报错、只会算错**。
//     另外 host 侧要求 d % 16 == 0 且 subN | d，否则会 GRAPH_FAILED。
//
// 用 torch.utils.cpp_extension.load 直接 JIT 编译（见 test_situ_glu_fused_real.py）。

#include <torch/extension.h>
#include "aclnn_common.h"
#include "aclnn_situ_glu_fused.h"

namespace op_api {

at::Tensor situ_glu_fused(const at::Tensor &x, const at::Tensor &w, double beta, double linear_beta)
{
    // x: [M, H] + w: [2d, H]  ->  out: [M, d]
    auto sizes = x.sizes().vec();
    sizes.back() = w.size(0) / 2;                 // 2d -> d
    auto out = at::empty(sizes, x.options());
    ACLNN_CMD(aclnnSituGluFused, x, w, beta, linear_beta, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("situ_glu_fused", &situ_glu_fused,
          "SituGluFused (aclnnSituGluFused wrapper): x[M,H] + Wp[2d,H] -> y[M,d]");
}

}  // namespace op_api
