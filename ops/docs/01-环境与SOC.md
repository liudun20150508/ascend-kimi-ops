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

`aclnn_ops.py`（三个算子共用的运行时；harness 与接入层都从它调）用 `torch.utils.cpp_extension.load` JIT 编译 wrapper，wrapper 里 `#include "aclnn_common.h"`（提供 `ACLNN_CMD` 宏）。这个头来自 **ops-transformer** 仓库（`torch_extension/cann_ops_transformer/common/inc/aclnn_common.h`），**不是** torch_npu 自带的。

**路径已参数化，换机器不用改代码**（原来是写死的绝对路径）：

| 环境变量 | 作用 | 默认值 |
|---|---|---|
| `ASCEND_HOME_PATH` | CANN 安装目录 | `/usr/local/Ascend/cann-9.0.0` |
| `OPS_TRANSFORMER_PKG` | **ops-transformer 的包路径（换机器必设这一条）** | `/tmp/ops-transformer/torch_extension/cann_ops_transformer` |
| `KIMI3_VENDOR` | 三个算子**共用**的自定义算子安装目录（优先于下面那些单算子变量） | —— |
| `SITU_VENDOR` / `ATTNRES_VENDOR` / `SGF_VENDOR` | 单算子的自定义算子安装目录（覆盖） | `$ASCEND_HOME_PATH/opp/vendors/customize` |
| `SITU_WRAPPER_SRC` / `ATTNRES_WRAPPER_SRC` / `SGF_WRAPPER_SRC` | wrapper 源文件 | 仓库根目录下那份 |

```bash
# 换机器时先找到 aclnn_common.h，把它的父级两层填进 OPS_TRANSFORMER_PKG
export OPS_TRANSFORMER_PKG=$(dirname $(dirname $(find / -name aclnn_common.h 2>/dev/null | head -1)))/..
# 上机前跑一遍环境体检（会自动找 aclnn_common.h 并报 SOC）
bash scripts/preflight_check.sh
```

## 环境信息

| 项 | 值 |
|---|---|
| CANN | `/usr/local/Ascend/cann-9.0.0` |
| torch / torch_npu | 2.10.0+cpu / 2.10.0 |
| 设备 | Ascend910_9382 × 2 |
| SOC | `ascend910_93` |
| AscendC 编译器 | `ccec`、`bisheng` |
| 算子脚手架 | `msopgen` |
