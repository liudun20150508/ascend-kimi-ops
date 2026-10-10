#!/usr/bin/env python3
"""负对照（negative control）：**故意用错前提，判据必须报警**。

为什么必须有这一项
------------------
"对拍通过"只有在**判据有分辨力**时才有意义。判据分辨不出错误前提，
就是在给自己发合格证 —— 这是"缺乏验证机制设计"最典型的病。

本脚本做两个反向检查：
  ① 故意用错 `beta`（4.0 → 4.5）调自研算子 ⇒ **必须超容差**；
  ② 故意用错 `linear_beta`（25.0 → 20.0）⇒ **必须超容差**。
若任一"看起来也对"，说明对拍判据太松（或算子根本没算这个参数），前面的 PASS 不可信。

用法：
    /usr/local/python3.11.15/bin/python3 scripts/negative_control.py
退出码：0 = 两个负对照都按预期报警（= 正常）；1 = 有一个没报出来（= 判据有问题）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _c in ('/workspace/ops-transformer/torch_extension/cann_ops_transformer',
           '/tmp/ops-transformer/torch_extension/cann_ops_transformer'):
    if os.path.isdir(_c):
        os.environ.setdefault('OPS_TRANSFORMER_PKG', _c)
        break
os.environ.setdefault('ASCEND_HOME_PATH', '/usr/local/Ascend/cann-9.0.0')
os.environ.setdefault('KIMI3_VENDOR', os.environ['ASCEND_HOME_PATH'] + '/opp/vendors/customize')
os.environ.setdefault('ASCEND_CUSTOM_OPP_PATH', os.environ['KIMI3_VENDOR'])

import torch          # noqa: E402
import torch_npu      # noqa: E402,F401
import aclnn_ops      # noqa: E402
from reference.situ import situ_and_mul as golden  # noqa: E402

TOL = 1e-2


def max_rel(out, ref):
    return ((out.float() - ref.float()).abs() / (ref.float().abs() + 1e-3)).max().item()


def main():
    torch.manual_seed(0)
    x = (torch.randn(2, 7, 22016, dtype=torch.bfloat16) * 6.0).npu()
    ref = golden(x.cpu(), beta=4.0, linear_beta=25.0).to(torch.bfloat16)

    # 前置：正确参数必须通过（否则负对照无从谈起）
    rel_ok = max_rel(aclnn_ops.situ_and_mul(x, 4.0, 25.0).cpu(), ref)
    print(f"[前置] 正确参数 max_rel = {rel_ok:.3e}  (判据 < {TOL:g})")
    if rel_ok >= TOL:
        print("❌ 前置就不通过，负对照无意义")
        return 1

    bad = 0
    for tag, b, lb in (('错误 beta (4.0→4.5)', 4.5, 25.0),
                       ('错误 linear_beta (25.0→20.0)', 4.0, 20.0)):
        rel = max_rel(aclnn_ops.situ_and_mul(x, b, lb).cpu(), ref)
        ok = rel > TOL
        print(f"[负对照] {tag:28s} max_rel = {rel:.3e}  -> "
              f"{'✅ 按预期超容差' if ok else '❌ 竟然也通过（判据无分辨力）'}")
        if not ok:
            bad += 1

    if bad:
        print(f"\n❌ 有 {bad} 个负对照没报警 ⇒ 判据太松，对拍结论不可信")
        return 1
    print("\n✅ PASS —— 负对照全部按预期报警（判据有分辨力）")
    return 0


if __name__ == '__main__':
    sys.exit(main())
