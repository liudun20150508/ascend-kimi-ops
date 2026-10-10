
#include "../op_kernel/situ_glu_fused_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include <cstdio>   // 调试落盘用

namespace optiling {

// epilogue 同时持有 g/u/tmp1/tmp2 四片 fp32，外加 out 的 bf16
static constexpr uint32_t EPI_BYTES_PER_ELEM = 4 * sizeof(float) + sizeof(uint16_t);   // = 18
static constexpr uint32_t UB_BUDGET_DENOM = 2;      // UB 一半给 Matmul、一半给 epilogue
static constexpr uint32_t SUBN_ALIGN = 16;          // bf16 下 32B = 16 元素；fp32 下 32B = 8 元素
static constexpr uint32_t FALLBACK_CORE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  SituGluFusedTilingData *tiling = context->GetTilingData<SituGluFusedTilingData>();

  // ---- 形状：x[M,H]、weight[2d,H]、y[M,d] ----
  const gert::Shape &xShape = context->GetInputShape(0)->GetStorageShape();
  const gert::Shape &wShape = context->GetInputShape(1)->GetStorageShape();
  uint32_t M = static_cast<uint32_t>(xShape.GetDim(0));
  uint32_t H = static_cast<uint32_t>(xShape.GetDim(1));
  uint32_t d = static_cast<uint32_t>(wShape.GetDim(0) / 2);
  tiling->m = M;
  tiling->h = H;
  tiling->d = d;

  // ---- attrs ----
  const gert::RuntimeAttrs* attrs = context->GetAttrs();
  tiling->beta = *attrs->GetFloat(0);
  tiling->linear_beta = *attrs->GetFloat(1);
  tiling->invBeta = (tiling->beta != 0.0f) ? (1.0f / tiling->beta) : 0.0f;
  tiling->invLinearBeta = (tiling->linear_beta != 0.0f) ? (1.0f / tiling->linear_beta) : 0.0f;

  // ---- ★ 新不变量（重建后）：d 必须是 16 的倍数 ----
  //   epilogue 的 DataCopyPad 依赖「UB 内每行 32B 对齐」⇒ 片宽 subN 必须是 16 的倍数；
  //   而权重重排工具 interleave_weight.py 的硬前提是 d % subN == 0。
  //   两者合起来要求 d 是 16 的倍数 —— KIMI3 的 intermediate_size = 11008 = 16×688 ✅。
  //   不满足就**明确失败**，绝不静默算错（忘了重排会静默错配 gate/up，这是本方案最大的坑）。
  if (d == 0 || (d % SUBN_ALIGN) != 0) {
    std::fprintf(stderr,
                 "[SituGluFused] d=%u 不是 %u 的倍数：当前实现不支持（重排/UB 对齐前提不成立）\n",
                 d, SUBN_ALIGN);
    return ge::GRAPH_FAILED;
  }

  // ---- 平台 ----
  auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  // ★ MIX（Cube+Vector）算子：blockDim 与 Matmul 的 SetDim 都用 **AIC 核数**，
  //   AIV 数由 MIX 配比自动推出（1_2 ⇒ AIV = 2×AIC）。
  uint32_t coreNumAic = platform.GetCoreNumAic();
  uint32_t coreNumAiv = platform.GetCoreNumAiv();
  uint32_t coreNum = coreNumAic ? coreNumAic : (coreNumAiv ? coreNumAiv : FALLBACK_CORE_NUM);
  // MIX_AIC_1_2 ⇒ 每个 AIC 配 2 个 AIV（拿不到平台 AIV 数时按 2× 推）
  uint32_t aivNum = coreNumAiv ? coreNumAiv : (coreNum * 2);

  // ---- UB 容量 ----
  uint64_t ubSize = 0;
  platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

  // ---- Matmul tiling：**必须用官方 API 生成，不能手写** ----
  matmul_tiling::MultiCoreMatmulTiling mmTiling(platform);
  mmTiling.SetDim(coreNum);
  mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_BF16);
  // B = Wp 是 [2d, H] = [N, K] 行主序 ⇒ 转置（Matmul 按 [N,K] 取数）
  mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_BF16, true);
  // ★ 重建：C 声明为 **GM**（落 workspace），与 kernel 侧 SitGluCType 一致。
  //   旧版声明 VECIN（想 GetTensorC 到 UB 做融合）在 MIX 下挂住，见 docs/14 §14.4。
  mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                    matmul_tiling::DataType::DT_FLOAT);
  mmTiling.SetShape(M, 2 * d, H);          // [M] × [2d] × [H]
  mmTiling.SetOrgShape(M, 2 * d, H);
  mmTiling.SetBias(false);
  if (mmTiling.GetTiling(tiling->mmTiling) == -1) {
    return ge::GRAPH_FAILED;
  }
  // 注意：C 落 GM 后不再需要 SetBufferSpace 给 C 预留 UB，
  //      也不再需要为它二次 GetTiling（旧版的两次 GetTiling 一并去掉）。

  tiling->baseM = tiling->mmTiling.baseM;
  tiling->baseN = tiling->mmTiling.baseN;
  tiling->coreNum = coreNum;
  tiling->aivNum = aivNum;

  // ---- epilogue 分片：UB 约束决定 subN ----
  // 片 = (若干行) × subN 列；一对 (gate_j, up_j) 各占一片。
  // subN 越大 → 每行 DataCopyPad 越长；越小 → 片数越多。允许一个片跨多行以摊薄启动开销。
  uint32_t subN = 0;
  if (ubSize > 0) {
    uint32_t budget = static_cast<uint32_t>(ubSize / UB_BUDGET_DENOM);
    subN = budget / EPI_BYTES_PER_ELEM;            // 按「一行」估的元素数上限
    subN = subN / SUBN_ALIGN * SUBN_ALIGN;
  }
  if (subN > d) {
    subN = d;
  }
  // ★ 新不变量：subN 必须是 16 的倍数，且 **整除 d**（权重重排的前提）
  while (subN > SUBN_ALIGN && (d % subN) != 0) {
    subN -= SUBN_ALIGN;
  }
  if (subN < SUBN_ALIGN || (d % subN) != 0) {
    subN = d;                                      // d 是 16 的倍数 ⇒ 这个也满足不变量
  }
  tiling->subN = subN;
  // kPairs 只用于落盘/文档口径（epilogue 现在不依赖它与 baseN 的关系）
  uint32_t kPairs = (tiling->baseN / 2) / subN;
  tiling->kPairs = (kPairs == 0) ? 1 : kPairs;

  // ---- epilogue 每次处理的行数（UB 预算 / 行开销） ----
  uint32_t epiRows = 1;
  if (ubSize > 0) {
    uint32_t epiBudget = static_cast<uint32_t>(ubSize / UB_BUDGET_DENOM);
    uint32_t perRowBytes = subN * EPI_BYTES_PER_ELEM;
    if (perRowBytes > 0) {
      epiRows = epiBudget / perRowBytes;
    }
  }
  if (epiRows == 0) {
    epiRows = 1;
  }
  if (epiRows > M) {
    epiRows = M;
  }
  tiling->epiRows = epiRows;

  // ---- 调试输出：把 matmul/epilogue 的关键 tiling 值落盘 ----
  // ⚠️ 格式**不能改**：scripts/verify_situ_glu_fused_aclnn.cpp 用 sscanf 按这个格式解析，
  //    对拍程序要按同一个 subN 做权重重排，否则 gate/up 配对不上（会静默算错）。
  if (FILE *fp = std::fopen("/tmp/situ_glu_fused_tiling.txt", "w")) {
    std::fprintf(fp, "baseM=%u baseN=%u baseK=%u subN=%u kPairs=%u singleCoreM=%u usedCoreNum=%u\n",
                 tiling->baseM, tiling->baseN, tiling->mmTiling.baseK, tiling->subN,
                 tiling->kPairs, tiling->mmTiling.singleCoreM, tiling->mmTiling.usedCoreNum);
    std::fclose(fp);
  }

  context->SetBlockDim(coreNum);

  // ---- workspace：sys（lib API）+ 用户区（C 的 fp32 缓冲） ----
  // ★ 根因修复保留：原代码 `currentWorkspace[0] = 0` 会让 args[4] 为 NULL，
  //   而 MIX 下的 Matmul 需要 workspace（设备侧曾报 507015，见 docs/14 §13.2）。
  //   现在用户区的大小不再拍脑袋（旧版写死 16 MB），而是按 C 的真实尺寸给：
  //     C = [M, 2d] fp32，行主序 —— kernel 里 cGm 就从 GetUserWorkspace() 的起点开始。
  size_t libWsBytes = static_cast<size_t>(platform.GetLibApiWorkSpaceSize());
  // ★ 2026-09-19 真机定位：Matmul 的 IterateAll 按 **baseN 粒度**写 C，尾块（curN < baseN）会在
  //   行末多写 `baseN - curN` 列 ⇒ 越过 C 的尾部。第一次执行只是"污染了后面那块内存"
  //   （自己的结果仍然正确），第二次执行才炸（`MTE DDR address out of range`）。
  //   ⇒ C 之后必须留出至少一整行的余量。
  size_t cBytes = static_cast<size_t>(M) * (2 * static_cast<size_t>(d)) * sizeof(float);
  const size_t kWsAlign = 512;                     // 用户 workspace 起点按 512B 对齐，留够余量
  cBytes = ((cBytes + kWsAlign - 1) / kWsAlign) * kWsAlign;
  // 注：2026-09-19 曾试过 libWs*2 给 sys 区留余量，实测**不解决问题**（变成挂住），已回退。
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  currentWorkspace[0] = libWsBytes + cBytes;
  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    // x:[M,H]  weight:[2d,H]  →  y:[M,d]
    const gert::Shape* x = context->GetInputShape(0);
    const gert::Shape* w = context->GetInputShape(1);
    gert::Shape* y = context->GetOutputShape(0);
    y->SetDimNum(2);
    y->SetDim(0, x->GetDim(0));                    // M
    y->SetDim(1, w->GetDim(0) / 2);                // d
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}


namespace ops {
class SituGluFused : public OpDef {
public:
    explicit SituGluFused(const char* name) : OpDef(name)
    {
        // x:[M,H]、weight:[2d,H]（**行序已交错重排**，见 README）
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("weight")
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
        // 换 SOC 时改这一行（A2 → "ascend910b"），同时改 CMakePresets.json
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SituGluFused);
}
