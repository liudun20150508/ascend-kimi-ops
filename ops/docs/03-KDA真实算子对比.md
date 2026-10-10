# KDA 真实算子对比（华为 chunk_kda_fwd vs 参考实现）

KIMI3 KDA 算子的关键结果：真实华为 `chunk_kda_fwd`（aclnnChunkKdaFwd，AscendC 实现）与参考实现 `reference/kda_chunked.py` 的 `chunk_kda_reference` 在 NPU 上对比通过，attn_out / final_state 相对误差 ~4e-3（bf16 精度内）。

## 结果

| case | attn_out | final_state |
|---|---|---|
| 单 chunk K16 | 4.41e-3 | 3.38e-3 |
| 多 chunk K16 | 4.39e-3 | 3.54e-3 |
| GVA H2→HV4 K32 | 4.46e-3 | 3.27e-3 |
| KIMI3 真实 shape (H=96,K=128,V=128) | 4.37e-3 | 3.37e-3 |

## 关键约定

- 参考实现 `chunk_kda_reference` 用朴素 `torch.linalg.inv(I+L)`，**慢衰减输入下才有限**（gate 快衰减会导致 E=exp(cumsum) 下溢、k_hat=k/E 爆炸、(I+L) 奇异 → NaN）。对比必须用慢衰减输入：`g = randn*0.5 - 3.0`、`dt_bias = -5.0`、`a_log = 0`。
- 真实算子 final_state 输出 fp32（aclnn 规范），参考实现返回 bf16 → dtype 差异导致的 ~3e-3 是 bf16 舍入，非 bug。
- torch_extension 包装：`ACLNN_CMD` 调 `aclnnChunkKdaFwd`，支持 BSND/BNSD/TND/NTD。
- 对比脚本：`compare_real_chunk_kda.py`（standalone JIT load，绕过缺 torchair 的 package import）。
