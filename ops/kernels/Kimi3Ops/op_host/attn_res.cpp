
#include "../op_kernel/attn_res_tiling.h"
#include "register/op_def_registry.h"
// PlatformAscendC：host 端查 AIV 核数（原来硬编码 blockDim=8，换 SOC 就废）
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  AttnResTilingData *tiling = context->GetTilingData<AttnResTilingData>();
  // block_residual: (T, B, H)
  const gert::StorageShape* br_shape = context->GetInputShape(0);
  const gert::Shape &br = br_shape->GetStorageShape();
  tiling->T = static_cast<uint32_t>(br.GetDim(0));
  tiling->B = static_cast<uint32_t>(br.GetDim(1));
  tiling->H = static_cast<uint32_t>(br.GetDim(2));

  // 核数按实际 AIV 核数取，不再写死 8；T 少于核数时 kernel 内会自行退出
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
  tiling->blockDim = (coreNum > 0) ? coreNum : 8;

  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  tiling->eps = *attrs->GetFloat(0);
  tiling->invH = 1.0f / static_cast<float>(tiling->H);

  context->SetBlockDim(tiling->blockDim);
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    // block_residual: (T, B, H) -> out: (T, H)
    const gert::Shape* br = context->GetInputShape(0);
    gert::Shape* out = context->GetOutputShape(0);
    out->SetDimNum(2);
    out->SetDim(0, br->GetDim(0));   // T
    out->SetDim(1, br->GetDim(2));   // H
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class AttnRes : public OpDef {
public:
    explicit AttnRes(const char* name) : OpDef(name)
    {
        this->Input("block_residual")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("prefix_sum")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("score_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("eps").Float();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_93");

    }
};

OP_ADD(AttnRes);
}
