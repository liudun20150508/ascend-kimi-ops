# 环境与 SOC

本机 Ascend 设备是 **Ascend910_9382**，对应 CANN SOC 是 **`ascend910_93`**（SOC_MAP_EXT 中 `"ascend910_93" -> "Ascend910_9391"`），**不是** `ascend910b`。`npu-smi` 的 Name 列显示 "Ascend910"。

## 关键事实

- ops-transformer 的 `build.sh` 默认 `--soc=ascend910b`，在此设备上编译出的 `.run` 内核跑不起来（报 "Op ChunkKdaFwd does not has any binary"）。必须 `--soc=ascend910_93` 重新编译。
- 自定义算子安装路径：`/usr/local/Ascend/cann-9.0.0/opp/vendors/`（本项目用 vendor `customize`）。
- 调用前需设置环境变量：
  - `ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/customize`
  - `LD_LIBRARY_PATH=/usr/local/Ascend/cann-9.0.0/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH`
- 本机缺 `torchair`（pip 装不了，网络受限），因此 `import cann_ops_transformer` 会失败。绕法：用 `torch.utils.cpp_extension.load` 直接 JIT 编译 csrc（复刻 builder 的 include_paths/extra_cflags/extra_ldflags）。

## 运行 harness 的外部依赖

`test_situ_real.py` / `test_attn_res_real.py` 用 `torch.utils.cpp_extension.load` JIT 编译 wrapper，wrapper 里 `#include "aclnn_common.h"`（提供 `ACLNN_CMD` 宏）。这个头来自 **ops-transformer** 仓库（`torch_extension/cann_ops_transformer/common/inc/aclnn_common.h`），**不是** torch_npu 自带的。运行前把该 include 路径和 wrapper 源文件路径改成本机实际位置即可（脚本里 `pkg_path` 和 `src` 两个变量）。昇腾团队本就有 ops-transformer，按此对齐路径即可。

## 环境信息

| 项 | 值 |
|---|---|
| CANN | `/usr/local/Ascend/cann-9.0.0` |
| torch / torch_npu | 2.10.0+cpu / 2.10.0 |
| 设备 | Ascend910_9382 × 2 |
| SOC | `ascend910_93` |
| AscendC 编译器 | `ccec`、`bisheng` |
| 算子脚手架 | `msopgen` |
