# Kimi3Ops —— 三个算子合成**一个包**

> 为什么需要它：**昇腾的 `opp/vendors/<vendor>` 目录一次只装得下一套算子包**。
> 分开装会互相覆盖 —— 实测过：装完 `SituGluFused` 后，`SituAndMul` / `AttnRes` 的
> `aclnn_*.h` 就消失了，连 `torch_ops_kimi3.py` 的自检都会在第一段就挂。
> 所以**交付/验证一律用本包**。

## 包含哪三个算子

| 算子 | kernel / host | 状态 |
|---|---|---|
| `SituAndMul`（SiTU-GLU） | `op_kernel/situ_and_mul.cpp` + `op_host/situ_and_mul.cpp` | ✅ 真机对拍 PASS（max_rel 3.89e-3，达标率 81.5%） |
| `AttnRes`（残差栈） | `op_kernel/attn_res.cpp` + `op_host/attn_res.cpp` | ✅ 真机对拍 PASS（max_rel 7.7e-3） |
| `SituGluFused`（GEMM+SiTU epilogue） | `op_kernel/situ_glu_fused.cpp` + `op_host/situ_glu_fused.cpp` | ✅ 真机对拍 PASS（4 组 shape，含真实 d=11008；`docs/14` §16.4） |

## 构建 / 安装

```bash
source $ASCEND_HOME_PATH/set_env.sh
cd kimi3_ops/kernels/Kimi3Ops
bash build.sh                                       # → build_out/custom_opp_*.run
./build_out/custom_opp_ubuntu_aarch64.run --install-path=$ASCEND_HOME_PATH/opp --quiet
```

**验收**（三个都要在）：

```bash
ls $ASCEND_HOME_PATH/opp/vendors/customize/op_api/include/     # 应有 aclnn_{attn_res,situ_and_mul,situ_glu_fused}.h
```

## ⚠️ 与开发目录的关系（同步要求）

`kernels/SituAndMul/`、`kernels/AttnRes/`、`kernels/SituGluFused/` 是**各自的开发目录**
（能单独 `build.sh`、单独对拍）；本目录是**打包分发目录**，源码是它们的副本。

⇒ **改代码时以各自的开发目录为准**（那边有独立的 kernel/host/验证脚本），改完**同步到本目录**再打包。
（本目录的 `op_kernel/CMakeLists.txt` 比单个算子多一行 `-I$ASCEND_CANN_PACKAGE_PATH/aarch64-linux/asc/include`，
因为融合算子要用 Matmul 高阶 API —— 这条**不能删**。）

## 一条命令跑全（推荐）

```bash
cd kimi3_ops
source $ASCEND_HOME_PATH/set_env.sh
/usr/local/python3.11.15/bin/python3 integration/torch_ops_kimi3.py   # torch.ops 注册 + fake + 图捕获 + 融合算子（①~⑤）
/usr/local/python3.11.15/bin/python3 test_situ_real.py                # SiTU-GLU 对拍
/usr/local/python3.11.15/bin/python3 test_attn_res_real.py            # AttnRes 对拍
/usr/local/python3.11.15/bin/python3 test_situ_glu_fused_real.py      # 融合算子对拍（含自动重排）
```
