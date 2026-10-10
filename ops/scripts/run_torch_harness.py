#!/usr/bin/env python3
"""在"缺 triton"的环境里跑 torch 侧 harness（test_situ_real.py / bench_situ_real.py 等）。

背景（实测，见 kimi3_ops/docs/14 §4 / §12）
------------------------------------------------
torch 2.10 的 `import torch` 会连带 `torch._dynamo` / `_inductor` 去 import triton，
而本机（aarch64、内网离线）没有 triton ⇒ `import torch` 直接失败，
于是 test_situ_real.py / bench_situ_real.py 全都跑不起来。

但这两个脚本**根本不用 torch.compile / inductor** —— triton 只是 import 期的依赖。

本脚本做三件事
------------------------------------------------
1. 先照常 `import torch`；成功就用真环境，**不注入任何东西**；
2. 失败且报错指向 triton 时，注入一个**惰性 stub**（任意 `triton.*` 属性都自动存在），
   再 import 一次；
3. import 成功 → 打印版本 / device_count / 冒烟算一发 → 用 runpy 执行目标脚本。

⚠️ 这是"绕开环境环"的手段，不是"让 inductor 能用"：
   打桩后的 triton 是空壳，`torch.compile` / `torch._inductor` 仍然不可用（本任务也不需要）。

用法
------------------------------------------------
    # 解释器必须是和 torch/torch_npu 配套的那个（本机是 py3.11.15，不是系统 python3！）
    /usr/local/python3.11.15/bin/python3 scripts/run_torch_harness.py test_situ_real.py
    /usr/local/python3.11.15/bin/python3 scripts/run_torch_harness.py bench_situ_real.py --bench-bw-only
    /usr/local/python3.11.15/bin/python3 scripts/run_torch_harness.py bench_situ_real.py --peak-bw 1208.7

    退出码：0 成功；2 参数/目标脚本问题；3 报错与 triton 无关；4 打桩后仍失败；5 NPU 不可用。
"""

import importlib
import os
import runpy
import sys
import types

# 本文件在 kimi3_ops/scripts/ 下 ⇒ 包根是上一层
KIMI3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 已知会被 torch._dynamo / _inductor 在 import 期摸到的 triton 子模块
TRITON_MODULES = (
    "triton",
    "triton.language",
    "triton.language.core",
    "triton.language.extra",
    "triton.compiler",
    "triton.compiler.compiler",
    "triton.backends",
    "triton.backends.compiler",
    "triton.runtime",
    "triton.runtime.driver",
)


class _Stub(types.ModuleType):
    """万能桩模块：任何属性访问都返回（可调用的）子桩，任何 import 都不抛错。"""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)          # 别拦 dunder，否则 import 机制会坏
        child = _Stub(f"{self.__name__}.{name}")
        setattr(self, name, child)              # 缓存，保证多次访问是同一个对象
        return child

    def __call__(self, *a, **k):
        return _Stub(self.__name__ + "()")

    def __repr__(self):
        return f"<triton stub {self.__name__}>"


def _install_triton_stub():
    for mod in TRITON_MODULES:
        sys.modules.setdefault(mod, _Stub(mod))
    # 最常见的硬依赖写法： from triton.backends.compiler import ...
    sys.modules["triton"].language = sys.modules["triton.language"]
    sys.modules["triton"].backends = sys.modules["triton.backends"]
    sys.modules["triton.backends"].compiler = sys.modules["triton.backends.compiler"]
    print("[harness] 已注入 triton stub（只补 import 期依赖；inductor/torch.compile 仍不可用）")


def _import_torch():
    """返回 (ok, err_text)。"""
    try:
        importlib.import_module("torch")
        return True, ""
    except Exception as e:                      # noqa: BLE001 —— 这里就是要抓一切
        return False, f"{type(e).__name__}: {e}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    target = sys.argv[1]
    if not os.path.isabs(target):
        cand = os.path.join(KIMI3, target)
        target = cand if os.path.exists(cand) else os.path.abspath(target)
    if not os.path.exists(target):
        print(f"[harness] 找不到目标脚本: {target}")
        return 2

    ok, err = _import_torch()
    if ok:
        print("[harness] `import torch` 直接成功 → 用真环境，不注入 stub")
    else:
        print(f"[harness] `import torch` 失败：{err}")
        if "triton" not in err.lower():
            print("[harness] 这个报错与 triton 无关 —— 不是本脚本要绕的问题，先解决上面这条")
            return 3
        _install_triton_stub()
        ok2, err2 = _import_torch()
        if not ok2:
            print(f"[harness] 打桩后仍然失败：{err2}")
            print("[harness] 说明还缺别的依赖（不止 triton）。把上面这条报错贴回来再定位")
            return 4
        print("[harness] 打桩后 `import torch` 成功")

    import torch          # noqa: E402
    import torch_npu      # noqa: E402

    n = torch.npu.device_count()
    print(f"[harness] torch {torch.__version__} / torch_npu {torch_npu.__version__} / "
          f"npu devices = {n}")
    if n == 0:
        print("[harness] device_count = 0 —— NPU 不可用，先查 torch_npu 与驱动，后面必然失败")
        return 5
    smoke = (torch.ones(4, dtype=torch.bfloat16).npu() * 2).cpu().tolist()
    print(f"[harness] 冒烟 ones(4)*2 = {smoke}（期望全是 2.0）")

    # 目标脚本按 __main__ 执行，argv 原样透传
    sys.argv = [target] + sys.argv[2:]
    if KIMI3 not in sys.path:
        sys.path.insert(0, KIMI3)
    print(f"[harness] 开始跑 {os.path.relpath(target, KIMI3)} {' '.join(sys.argv[1:])}".rstrip())
    try:
        runpy.run_path(target, run_name="__main__")
    except SystemExit as e:                     # 目标脚本自己 sys.exit(...)
        return int(e.code) if isinstance(e.code, int) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
