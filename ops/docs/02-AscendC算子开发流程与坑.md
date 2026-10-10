# AscendC 自定义算子开发流程与坑（A3 / ascend910_93）

在昇腾 A3 上从零开发自定义 AscendC 算子的已验证流程，及踩过的坑。

## 完整流程

`msopgen 生成工程 → 填 kernel/host → build.sh → 装 .run → JIT 包装 → golden 对拍`

1. 写算子定义 JSON（IR 格式：`op` / `input_desc` / `output_desc` / `attr`）。
2. `msopgen gen -i op.json -f aclnn -c ai_core-ascend910_93 -lan cpp -out <dir>` 生成工程骨架（`op_kernel` + `op_host` + `build.sh`）。
3. 填三个文件：
   - `op_kernel/*_tiling.h`：tiling 结构（kernel 需要的量：d/numRows/blockDim/beta 等）。
   - `op_host/*.cpp`：`InferShape`（改输出 shape）+ `TilingFunc`（算 tiling、`context->GetAttrs()->GetFloat(i)` 读 attr、`SetBlockDim`）。
   - `op_kernel/*.cpp`：AscendC kernel 本体。
4. `bash build.sh` → 出 `custom_opp_ubuntu_aarch64.run`（SOC 在 CMakePresets.json 的 `ASCEND_COMPUTE_UNIT=ascend910_93`）。
5. `./xxx.run --install-path=/usr/local/Ascend/cann-9.0.0/opp --quiet` 装到 `opp/vendors/customize/`。
6. 写 C++ torch 包装（`ACLNN_CMD(aclnnXxx, ...)`，include `aclnn_common.h` + 生成的 `aclnn_xxx.h`），用 `torch.utils.cpp_extension.load` JIT 编译，vs golden 对拍。

## 关键坑（都踩过）

1. **`DT_BF16` 不是类型**：是宏 `#define DT_BF16 27`。真正的 bf16 类型是 `bfloat16_t`。
2. **`REGISTER_TILING_DEFAULT(TilingStruct)` 必须写在 `GET_TILING_DATA(tilingData, tiling)` 之前**，它负责声明 `tilingData` 变量（删了会报 undeclared identifier）。
3. **高层 API（LocalTensor + TQue）的跨引擎同步靠 `EnQue`/`DeQue`，漏了输出全零**：
   - 输入（MTE2→V）：`AllocTensor → DataCopy → EnQue → DeQue → compute → FreeTensor`
   - 输出（V→MTE3）：`AllocTensor → compute → EnQue → DeQue → DataCopy → FreeTensor`
   - 中间 V-only 缓冲（Cast/Tanh/Mul 等）不需要 EnQue/DeQue，但 QuePosition 用 `VECCALC`（输入 `VECIN`、输出 `VECOUT`）。
   - 底层 `__ubuf__` 内建函数才用 `pipe_barrier(PIPE_V)`，高层 API 不要用。
4. **标量 `exp` 在 aicore 上不存在**（只有 `sqrt`）。softmax 等要用**向量 `AscendC::Exp`** 在 32 对齐的小 buffer 上做；标量 `sqrt` 可用。
5. **uint→float 的 cast 不允许**（`static_cast<float>(H)` 报错）。把 `invH` 这类量移到 host 端算好，存进 tiling 结构传进来。
6. **归约**：`ReduceSum(dst, src, workBuf, count)`（sum all elements），结果读 `dst.GetValue(0)`，`workBuf` 分配与 src 同量级即可。

## 参考完整样例

高层 API 三段式（AllocTensor/EnQue/DeQue/DataCopy）完整样例：
`/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/ascendc/apply_came_part2/apply_came_part2.cpp`

归约 + GetValue 用法：
`.../apply_came_part1/apply_came_part1_fp32.h`（`ReduceSum` + `GetValue(0)`）

## 里程碑

- **SiTU-GLU（SituAndMul）已开发并验证通过**：`kernels/SituAndMul/` + `situ_and_mul_wrapper.cpp` + `test_situ_real.py`。vs golden 误差 bf16 内（d=11008 时 max_rel≈7.5e-3）。
- **AttnRes 残差栈已开发并验证通过**：`kernels/AttnRes/` + `attn_res_wrapper.cpp` + `test_attn_res_real.py`。mini-attention（RMS-norm + score 点积 + softmax + 加权求和）。vs golden bf16 内（H=4096 时 max_rel≈7.7e-3）。
- 剩余缺口：**LatentMoE**（图级组合，非新 kernel）。
