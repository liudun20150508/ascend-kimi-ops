"""三个自研 AscendC 算子的 aclnn 运行时 —— **JIT 包装的唯一实现**。

为什么要有这一层
--------------
接入层 `integration/torch_ops_kimi3.py` 原先是这样转发 NPU 实现的：

    def _situ_and_mul_npu(x, beta, linear_beta):
        from test_situ_real import run_real      # ← 把测试脚本当生产包装用

三个 `test_*_real.py` 各自复制了一份「路径解析 + `torch.utils.cpp_extension.load`」，
既重复又让"已验证"的那份逻辑散在测试脚本里。本模块把它收成一处：
**接入层、测试 harness、bench 全从这里调**，测试脚本退回去只当测试。

| 算子 | Python 入口 | aclnn 算子 | wrapper 源 |
|---|---|---|---|
| SiTU-GLU | `situ_and_mul(x, beta, linear_beta)` | `aclnnSituAndMul` | `situ_and_mul_wrapper.cpp` |
| AttnRes | `attn_res(block_residual, prefix_sum, score_weight, eps)` | `aclnnAttnRes` | `attn_res_wrapper.cpp` |
| 融合算子 | `situ_glu_fused(x, w_permuted, beta, linear_beta)` | `aclnnSituGluFused` | `situ_glu_fused_wrapper.cpp` |

路径解析（换机器**只设环境变量、不改代码**）
------------------------------------------
| 环境变量 | 作用 | 默认 |
|---|---|---|
| `ASCEND_HOME_PATH` | CANN 安装目录 | `/usr/local/Ascend/cann-9.0.0` |
| `OPS_TRANSFORMER_PKG` | 提供 `aclnn_common.h`（即 `ACLNN_CMD` 宏，**不是** torch_npu 自带的） | `/tmp/ops-transformer/torch_extension/cann_ops_transformer` |
| `KIMI3_VENDOR` | 三个算子共用的 vendor 目录 | —— |
| `SITU_VENDOR` / `ATTNRES_VENDOR` / `SGF_VENDOR` | 单算子覆盖（优先于 `KIMI3_VENDOR`） | `<CANN>/opp/vendors/customize` |
| `SITU_WRAPPER_SRC` / `ATTNRES_WRAPPER_SRC` / `SGF_WRAPPER_SRC` | wrapper 源文件 | 本目录下对应的 `.cpp` |

★ **vendor 目录名不统一**是实测踩过的坑：`opp/vendors/` 下可能同时有 `custom_transformer`
与 `customize`，而且**一次只装得下一套算子包**（`docs/14` §11、§16.5）。所以解析顺序是
「环境变量 → 默认路径 → 在 `vendors/*/` 里按**头文件**自动探测」，探测命中就不必再手填。

诊断入口（不需要 NPU、不触发 JIT 编译）
------------------------------------
```bash
python3 kimi3_ops/aclnn_ops.py      # 列出三个算子各自的安装状态与"该怎么装"
```

⚠️ 验证状态（诚实标注）
--------------------
本模块的代码**在本机没有运行过** —— 开发容器里宿主沙箱拒绝 `bash`（见仓库根
`reasonix.toml` 注释）。它是从三份**已上机验证过**的 `_build()`（`test_situ_real.py` /
`test_attn_res_real.py` / `test_situ_glu_fused_real.py`）合并而来，`include_paths` /
`ldflags` / JIT 模块名**逐项保持不变**；但"合并后仍等价"这件事本身必须重跑对拍确认：

```bash
python3 test_situ_real.py && python3 test_attn_res_real.py && python3 test_situ_glu_fused_real.py
```
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

CANN_PATH = os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0")
PKG_PATH = os.environ.get("OPS_TRANSFORMER_PKG",
                          "/tmp/ops-transformer/torch_extension/cann_ops_transformer")

# key -> 该算子的全部静态信息（顺序即 vendors/*/ 探测顺序，无所谓优先级）
_SPECS = {
    "situ_and_mul": {
        "label": "SiTU-GLU",
        "kernels_dir": "kernels/SituAndMul",
        "wrapper": "situ_and_mul_wrapper.cpp",
        "wrapper_env": "SITU_WRAPPER_SRC",
        "module": "situ_and_mul_op",            # JIT 模块名（与原 harness 一致，命中缓存）
        "vendor_env": "SITU_VENDOR",
        "header": "aclnn_situ_and_mul.h",
    },
    "attn_res": {
        "label": "AttnRes",
        "kernels_dir": "kernels/AttnRes",
        "wrapper": "attn_res_wrapper.cpp",
        "wrapper_env": "ATTNRES_WRAPPER_SRC",
        "module": "attn_res_op",
        "vendor_env": "ATTNRES_VENDOR",
        "header": "aclnn_attn_res.h",
    },
    "situ_glu_fused": {
        "label": "SituGluFused",
        "kernels_dir": "kernels/SituGluFused",
        "wrapper": "situ_glu_fused_wrapper.cpp",
        "wrapper_env": "SGF_WRAPPER_SRC",
        "module": "situ_glu_fused_op",
        "vendor_env": "SGF_VENDOR",
        "header": "aclnn_situ_glu_fused.h",
    },
}


def vendor_dir(key):
    """解析该算子的 vendor 目录（环境变量 → 默认 → `vendors/*/` 按头文件探测）。

    探测不到时**返回默认路径而不抛错** —— 让 `_build()` 或 `available()` 给出可操作的提示。
    """
    spec = _SPECS[key]
    explicit = os.environ.get(spec["vendor_env"]) or os.environ.get("KIMI3_VENDOR")
    if explicit:
        return explicit
    default = os.path.join(CANN_PATH, "opp/vendors/customize")
    if os.path.exists(os.path.join(default, "op_api/include", spec["header"])):
        return default
    root = os.path.join(CANN_PATH, "opp/vendors")
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            cand = os.path.join(root, name)
            if os.path.exists(os.path.join(cand, "op_api/include", spec["header"])):
                return cand
    return default


def wrapper_src(key):
    spec = _SPECS[key]
    return os.environ.get(spec["wrapper_env"]) or os.path.join(_HERE, spec["wrapper"])


def _export_env(key):
    """把 vendor 挂进 `ASCEND_CUSTOM_OPP_PATH` / `LD_LIBRARY_PATH`（等价 `source set_env.bash`）。

    幂等：同一个 vendor 重复设置不会把路径越拼越长。
    """
    vendor = vendor_dir(key)
    for var, value in (
        ("ASCEND_CUSTOM_OPP_PATH", vendor),
        ("LD_LIBRARY_PATH", os.path.join(vendor, "op_api/lib")),
    ):
        parts = [p for p in os.environ.get(var, "").split(":") if p]
        if value not in parts:
            os.environ[var] = ":".join([value] + parts)


for _key in _SPECS:                      # import 即生效（aclnn 加载依赖这两个变量）
    _export_env(_key)


_OPS = {}                                # key -> 已加载的 JIT 模块（懒加载 + 缓存）


def _build(key):
    spec = _SPECS[key]
    if key in _OPS:
        return _OPS[key]

    import torch_npu                      # 延迟到真正要编译时才 import（available() 用不到）
    from torch.utils.cpp_extension import load

    src = wrapper_src(key)
    if not os.path.exists(src):
        raise FileNotFoundError(
            f"找不到 {spec['label']} 的 wrapper 源文件: {src}\n"
            f"用 {spec['wrapper_env']}=<路径> 指定，或确认在仓库里运行"
        )

    vendor = vendor_dir(key)
    header = os.path.join(vendor, "op_api/include", spec["header"])
    if not os.path.exists(header):
        raise FileNotFoundError(
            f"{vendor} 里没有 {spec['header']} —— 需要先装 {spec['label']} 算子包：\n"
            f"  cd {spec['kernels_dir']} && bash build.sh && "
            f"./build_out/custom_opp_*.run --install-path=$ASCEND_HOME_PATH/opp --quiet\n"
            f"（注意：vendor 目录一次只装得下一套算子包，多算子要用 kernels/Kimi3Ops 合并包；"
            f"见 docs/14 §11、§16.5）\n"
            f"（也可用 KIMI3_VENDOR / {spec['vendor_env']} 直接指定 vendor 目录）"
        )

    tn = os.path.dirname(os.path.abspath(torch_npu.__file__))
    abi = 1 if torch.compiled_with_cxx11_abi() else 0
    op = load(
        name=spec["module"],
        sources=[src],
        extra_include_paths=[
            os.path.join(tn, "include"),
            os.path.join(tn, "include/third_party/hccl/inc"),
            os.path.join(tn, "include/third_party/acl/inc"),
            os.path.join(CANN_PATH, "include"),
            os.path.join(PKG_PATH, "common/inc"),
            os.path.join(vendor, "op_api/include"),
        ],
        extra_cflags=["-fPIC", "-O2", "-w", f"-D_GLIBCXX_USE_CXX11_ABI={abi}"],
        extra_ldflags=[
            "-L" + os.path.join(CANN_PATH, "lib64"), "-lascendcl",
            "-L" + os.path.join(tn, "lib"), "-ltorch_npu",
            "-L" + os.path.join(vendor, "op_api/lib"), "-lcust_opapi",
        ],
        verbose=False,
    )
    _OPS[key] = op
    return op


# --------------------------------------------------------------------------
# 对外 API：三个算子各一个（签名与各自 harness 的 run_real 一致）
# --------------------------------------------------------------------------
def situ_and_mul(x, beta, linear_beta):
    """`x[..., 2d]` → `[..., d]`（gate 在前、up 在后；`situ(gate) * tanh_scale(up)`）。"""
    return _build("situ_and_mul").situ_and_mul(x, float(beta), float(linear_beta))


def attn_res(block_residual, prefix_sum, score_weight, eps=1e-6):
    """`(T,B,H) + (T,H) + (H,)` → `(T,H)`。"""
    return _build("attn_res").attn_res(block_residual, prefix_sum, score_weight, float(eps))


def situ_glu_fused(x, w_permuted, beta, linear_beta):
    """`x[M,H]` + `w_permuted[2d,H]` → `y[M,d]`。

    ⚠️ `w_permuted` 必须是**已按 subN 交错重排**的 `W'`。重排规则与两段式（先空跑读 subN）
    见 `test_situ_glu_fused_real.py` 的 `interleave()` / `run_fused()` 与 `interleave_weight.py`。
    忘了重排**不会报错、只会算错**——这是本方案最危险的一点。
    """
    return _build("situ_glu_fused").situ_glu_fused(
        x, w_permuted, float(beta), float(linear_beta))


# --------------------------------------------------------------------------
# 融合算子的**权重重排**（算子特有的前置约束 ⇒ 放运行时，而不是留给每个调用方各写一遍）
#   W'[2k·subN + i] = gate[k·subN + i] ;  W'[(2k+1)·subN + i] = up[k·subN + i]
#   规则与 `interleave_weight.py` 一致。忘了重排**不会报错、只会算错**。
# --------------------------------------------------------------------------
TILING_DUMP = "/tmp/situ_glu_fused_tiling.txt"      # host tiling 落盘（由 op_host 写）


def interleave(gate, up, subn):
    """把 `gate[d,H]` / `up[d,H]` 按 `subN` 交错成算子要的 `W'[2d,H]`。"""
    d = gate.shape[0]
    assert d % subn == 0, f"d={d} 不是 subN={subn} 的整数倍（host 会直接 GRAPH_FAILED）"
    blocks = []
    for j in range(0, d, subn):
        blocks.append(gate[j:j + subn])
        blocks.append(up[j:j + subn])
    return torch.cat(blocks, dim=0)


def read_subn():
    """从 host 落盘的 tiling 文件里读真实的 subN（文件不在 / 没写就返回 None）。"""
    if not os.path.exists(TILING_DUMP):
        return None
    with open(TILING_DUMP, encoding="utf-8") as f:
        for line in f:
            for kv in line.split():
                if kv.startswith("subN="):
                    return int(kv.split("=")[1])
    return None


def situ_glu_fused_auto(x, gate, up, beta=4.0, linear_beta=25.0, subn=None):
    """便捷入口：给**原始布局**的 `gate`/`up`（各 `[d,H]`），自动重排后再调算子。

    ⚠️ 重排需要 `subN`，而它只有 host tiling 算完才知道 ⇒ 两段式：
       先空跑一次让 host 落盘 tiling → 读 `subN` → 重排 → 正式跑。
    返回 `(y, subn)`（`subn` 可复用，避免重复空跑）。
    """
    if subn is None:
        if os.path.exists(TILING_DUMP):
            os.remove(TILING_DUMP)
        # 空跑只为让 host 落盘 tiling；weight 形状必须合法（[2d,H]），值与重排与否都无所谓
        _build("situ_glu_fused").situ_glu_fused(
            x, torch.cat([gate, up], dim=0), float(beta), float(linear_beta))
        torch.npu.synchronize()
        subn = read_subn()
        if subn is None:
            raise RuntimeError(f"没读到 {TILING_DUMP} 里的 subN")
    w_p = interleave(gate, up, subn)
    y = _build("situ_glu_fused").situ_glu_fused(x, w_p, float(beta), float(linear_beta))
    return y, subn


def available():
    """三个算子的安装状态（**只查文件系统**：不加载 JIT、不需要 NPU、不触发编译）。"""
    out = {}
    for key, spec in _SPECS.items():
        vendor = vendor_dir(key)
        src = wrapper_src(key)
        header = os.path.join(vendor, "op_api/include", spec["header"])
        out[key] = {
            "label": spec["label"],
            "kernels_dir": spec["kernels_dir"],
            "vendor": vendor,
            "header": header,
            "installed": os.path.exists(header),
            "wrapper_src": src,
            "wrapper_present": os.path.exists(src),
        }
    return out


def main():
    print("=" * 72)
    print("aclnn_ops 安装状态（只查文件系统，不加载 JIT、不需要 NPU）")
    print(f"  CANN_PATH = {CANN_PATH}")
    print(f"  PKG_PATH  = {PKG_PATH}")
    print("-" * 72)
    all_ok = True
    for st in available().values():
        print(f"  [{'OK  ' if st['installed'] else 'MISS'}] {st['label']:<14} vendor={st['vendor']}")
        if not st["installed"]:
            print(f"         缺头文件: {st['header']}")
            print(f"         装法: cd {st['kernels_dir']} && bash build.sh && \\")
            print("               ./build_out/custom_opp_*.run --install-path=$ASCEND_HOME_PATH/opp --quiet")
        if not st["wrapper_present"]:
            print(f"         缺 wrapper 源文件: {st['wrapper_src']}")
        all_ok = all_ok and st["installed"] and st["wrapper_present"]
    print("-" * 72)
    print("全部就绪" if all_ok else "有缺失项：按上面的提示装算子包后再跑一次")
    print("=" * 72)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
