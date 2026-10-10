// AttnRes 残差栈 (aclnnAttnRes) 的 torch_extension 包装。
// block_residual (T,B,H) + prefix_sum (T,H) + score_weight (H,) -> out (T,H)。

#include <torch/extension.h>
#include "aclnn_common.h"
#include "aclnn_attn_res.h"

namespace op_api {

at::Tensor attn_res(const at::Tensor &block_residual, const at::Tensor &prefix_sum,
                    const at::Tensor &score_weight, double eps)
{
    // block_residual: (T, B, H) -> out: (T, H)
    auto T = block_residual.size(0);
    auto H = block_residual.size(2);
    auto out = at::empty({T, H}, block_residual.options());
    ACLNN_CMD(aclnnAttnRes, block_residual, prefix_sum, score_weight, eps, out);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("attn_res", &attn_res, "AttnRes (aclnnAttnRes wrapper)");
}

}  // namespace op_api
