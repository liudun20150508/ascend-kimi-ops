
#include "../op_kernel/situ_and_mul_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  SituAndMulTilingData *tiling = context->GetTilingData<SituAndMulTilingData>();
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  const gert::Shape &storageShape = x1_shape->GetStorageShape();
  size_t dimNum = storageShape.GetDimNum();

  // 输入最后一维 = 2d，输出最后一维 = d
  int64_t lastDim = storageShape.GetDim(dimNum - 1);
  tiling->d = static_cast<uint32_t>(lastDim / 2);

  // 行数 = 除最后一维外各维的乘积
  tiling->numRows = 1;
  for (size_t i = 0; i + 1 < dimNum; i++) {
    tiling->numRows *= static_cast<uint32_t>(storageShape.GetDim(i));
  }

  tiling->blockDim = 8;

  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  tiling->beta = *attrs->GetFloat(0);
  tiling->linear_beta = *attrs->GetFloat(1);

  context->SetBlockDim(tiling->blockDim);
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = 0;
  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    // SiTU：输入 [..., 2d] -> 输出 [..., d]，最后一维减半
    size_t dimNum = x1_shape->GetDimNum();
    y_shape->SetDim(dimNum - 1, x1_shape->GetDim(dimNum - 1) / 2);
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
class SituAndMul : public OpDef {
public:
    explicit SituAndMul(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("beta").Float();
        this->Attr("linear_beta").AttrType(OPTIONAL).Float(1.0);

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_93");

    }
};

OP_ADD(SituAndMul);
}
