
#include "../op_kernel/situ_and_mul_tiling.h"
#include "register/op_def_registry.h"
// PlatformAscendC：host 端查 UB 容量与 AIV 核数，用于自适应 tiling。
// 头文件路径随 CANN 版本略有差异（tiling/platform/ 或 platform/），如编译报缺头，
// 按本机 $ASCEND_HOME_PATH/include 下的实际路径调整即可。
#include "tiling/platform/platform_ascendc.h"


namespace optiling {

// UB 每元素预算（字节）：
//   输入 gate/up 队列 depth=2：2 路 × 2 块 × 2B =  8 B/元素
//   输出 out 队列      depth=2：     1 路 × 2 块 × 2B =  4 B/元素
//   片内 fp32 计算缓冲        ：     3 块 × 4B        = 12 B/元素
// --------------------------------------------------------------
//   合计                                              = 24 B/元素
static constexpr uint32_t UB_BYTES_PER_ELEM = 24;
// UB 只用到一半：留余量给框架自身的同步/临时空间
static constexpr uint32_t UB_BUDGET_DENOM = 2;
// tileElem 对齐粒度（元素）：bf16 下 256 元素 = 512B，32B 对齐绰绰有余
static constexpr uint32_t TILE_ALIGN_ELEM = 256;
static constexpr uint32_t FALLBACK_TILE_ELEM = 4096;
static constexpr uint32_t FALLBACK_CORE_NUM = 8;

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

  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  tiling->beta = *attrs->GetFloat(0);
  tiling->linear_beta = *attrs->GetFloat(1);
  // 倒数在 host 端算好：aicore 上不允许 uint->float cast，且省掉 kernel 内每 tile 一次除法
  tiling->invBeta = (tiling->beta != 0.0f) ? (1.0f / tiling->beta) : 0.0f;
  tiling->invLinearBeta = (tiling->linear_beta != 0.0f) ? (1.0f / tiling->linear_beta) : 0.0f;

  // ---------- 硬件能力探测：UB 容量 + AIV 核数 ----------
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  uint64_t ubSize = 0;
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

  // ---------- tileElem：按 UB 容量反推，而不是编译期写死 ----------
  uint32_t tileElem = FALLBACK_TILE_ELEM;
  if (ubSize > 0) {
    uint32_t maxElem = static_cast<uint32_t>(ubSize / UB_BUDGET_DENOM / UB_BYTES_PER_ELEM);
    tileElem = maxElem / TILE_ALIGN_ELEM * TILE_ALIGN_ELEM;
    if (tileElem == 0) {
      tileElem = TILE_ALIGN_ELEM;
    }
  }
  // 不必超过 d 本身；d 不是对齐倍数时向上取整到对齐粒度（尾块由 kernel 内裁剪）
  if (tiling->d > 0 && tileElem > tiling->d) {
    tileElem = (tiling->d + TILE_ALIGN_ELEM - 1) / TILE_ALIGN_ELEM * TILE_ALIGN_ELEM;
  }

  // ⚠️ 前提：d 必须是 16 的倍数（bf16 下 32B = 16 元素，DataCopy 的对齐要求）。
  // 不满足时最后一个 tile 的拷贝长度会非对齐（可能越界读 GM），需要在 host 侧
  // pad 输入、或改走非对齐拷贝路径。KIMI3 的 d = 11008 满足（11008 = 16 × 688）。
  // 这里不做静默兜底 —— 静默兜底会把越界问题藏到运行期，不如在文档里写明前提。
  tiling->tileElem = tileElem;
  tiling->tilesPerRow = (tiling->d + tileElem - 1) / tileElem;
  tiling->totalTiles = tiling->numRows * tiling->tilesPerRow;

  tiling->blockDim = (coreNum > 0) ? coreNum : FALLBACK_CORE_NUM;

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
