"""LatentMoE 的「图接线」层 —— 第③层（图/模型层），**不是算子层**。

它回答的是缺口清单里的第三项：`LatentMoE 投影`。判定是「**不需要新 kernel**」——
整条链路全用现成算子拼（Linear=GEMM、MoE、RMSNorm、expert 内用我们自研的 SiTU-GLU），
所以这里交付的是**接线代码**，不是 kernel。详见 `../docs/12-算子如何接进vLLM-SGLang.md` §4。

官方出处（逐行对齐，权威来源是仓库里的 `reference/modeling_kimi_linear.py`）
---------------------------------------------------------------------
| 官方位置 | 行号 | 对应什么 |
|---|---|---|
| `KimiSparseMoeBlock.__init__` | 776-813 | `use_latent_moe` / `moe_hidden_size` / down,up,norm 的定义 |
| `KimiSparseMoeBlock.forward` | 815-838 | 时序：gate → down → moe_infer → [norm] → up → +shared |
| `KimiBlockSparseMLP` | 242-270 | **expert 工作在 latent 维**；`hidden_act=="situ"` ⇒ 用我们的 SiTU-GLU |
| `KimiMLP`（shared_experts） | 273-301 | 走**原始 hidden**，绕过 latent 投影 |
| `KimiRMSNorm` | 226-236 | norm 的确切算法（本文件 `KimiStyleRMSNorm` 逐行照抄） |

★ 三条最容易写错的不变量（本模块的自检就是**专门盯这三条**的探针）
----------------------------------------------------------------
1. **`gate` 用投影前的 hidden（H 维）算 topk**，不是 latent 空间（官方行 818 在 down 之前）；
2. **experts 的 `hidden_size = moe_hidden_size`**（latent 维），不是 H（官方行 786-793）；
3. **`shared_experts` 走原始 hidden**，不经过 down/up 投影（官方行 836-838 用的是 `identity`）。

为什么"图接线"要单独写一层
--------------------------
vLLM/SGLang 的 KIMI3 模型文件里，MoE 那一段是框架自己的类（vLLM 是 `FusedMoE`，SGLang 同构）。
我们**不去改框架主干**，而是把官方的 latent 语义接在框架的 MoE 前后——
这就是 `docs/12` §6 说的「不用改 vLLM 主干，昇腾的活都在 vllm-ascend 这层 plugin 里」。

落地到框架时要改的地方（两处，都是模型文件）
------------------------------------------
```python
# ① 初始化：加两个投影（+可选 norm）
self.routed_expert_down_proj = nn.Linear(hidden_size, moe_hidden_size, bias=False)
self.routed_expert_up_proj   = nn.Linear(moe_hidden_size, hidden_size, bias=False)
self.moe_hidden_size = config.routed_expert_hidden_size     # ← 框架 MoE 的 hidden 要用这个

# ② forward：按本文件 latent_moe_forward() 的时序接
```

✅ 验证状态（2026-09-19，**两段全部跑通**）
----------------------------------------
**段 1**（不变量探针）**11 项全 OK**；**段 2**（与官方 `KimiSparseMoeBlock` 灌同一份权重的数值
对照）**超容差 0 / 384 元素** —— 接线时序 gate → down → moe → norm → up 与官方一致。跑法：

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.0.0   # torch_npu 自动加载需要它
/workspace/venv_l2/bin/python kimi3_ops/integration/latent_moe.py
```

* 接线语义是**照官方源码逐行核对**出来的（上面那张表就是核对结果，行号可复查）；
* 段 2 需要 `transformers`（已装在 `/workspace/venv_l2`）与 `reference/` 可导入 —— 后者会
  拉起 `fla`→triton driver，而本机 `triton-ascend 3.2.0` 与 CANN 9.0 不配套（`docs/14` §15.3 更正），
  所以段 2 里**先给 fla 打了 stub**（`KimiSparseMoeBlock` 不使用 fla，只有 KDA 相关类才用）；
* ⚠️ 官方 gate 要求 **3D** 输入（`modeling_kimi_linear.py:704` 解包 3 维）⇒ 段 2 用 `(2,3,H)`；
  我们的 `latent_moe_forward` 2D/3D 都支持（段 1 两种都测过）；
* 缺依赖 ⇒ 显式 SKIP；**其它异常 ⇒ 判 FAIL**（不会把真 bug 当"环境问题"放过去）。

> 历史备注：原先此处写「本模块的自检在本机从未运行过」—— 当时开发容器的宿主沙箱拒绝执行
> `bash`（见仓库根 `reasonix.toml` 的注释）。该阻塞已于 2026-09-19 解除。
"""

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_KIMI3 = os.path.dirname(_HERE)          # kimi3_ops/
if _KIMI3 not in sys.path:
    sys.path.insert(0, _KIMI3)


class KimiStyleRMSNorm(nn.Module):
    """与官方 `KimiRMSNorm`（`reference/modeling_kimi_linear.py:226-236`）逐行对齐。

    注意两处细节，写错就会静默偏差：
    * 方差在 **fp32** 里算（`x.float()`），算完再 cast 回原 dtype；
    * `weight` 乘在 **cast 之后**（官方 `self.weight * x.to(dtype)`）。
    """

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        dtype = hidden_states.dtype
        x = hidden_states.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dtype)


class LatentMoeProjection(nn.Module):
    """LatentMoE 的**投影段**：`down(H→mh)` → `[RMSNorm(mh)]` → `up(mh→H)`。

    它**不含** gate 与 experts —— 那两个是框架 MoE 层的事，不由本层实现（不重复造）。
    因此对外是两个入口而不是一个 `forward`：`down()` 在 MoE 之前调，`up()` 在之后调。

    参数名与官方**逐字对齐**（`routed_expert_down_proj` / `routed_expert_up_proj` /
    `routed_expert_norm`），这样官方权重可以直接灌进来（自检段 2 就是这么做的）。
    """

    def __init__(self, hidden_size, moe_hidden_size, use_norm=False, eps=1e-6, dtype=None):
        super().__init__()
        if hidden_size <= 0 or moe_hidden_size <= 0:
            raise ValueError(f"hidden_size/moe_hidden_size 必须为正: {hidden_size}/{moe_hidden_size}")
        self.hidden_size = hidden_size
        self.moe_hidden_size = moe_hidden_size
        self.routed_expert_down_proj = nn.Linear(hidden_size, moe_hidden_size, bias=False, dtype=dtype)
        self.routed_expert_up_proj = nn.Linear(moe_hidden_size, hidden_size, bias=False, dtype=dtype)
        # 官方是「可选」的：latent_moe_use_norm（config 默认 False，行 781/810-813）
        self.routed_expert_norm = KimiStyleRMSNorm(moe_hidden_size, eps=eps) if use_norm else None

    def down(self, hidden_states):
        """投影进 latent 空间。`[..., H] → [..., mh]`。"""
        # ★ 形状防线：docs/12 §4 说 LatentMoE 的「风险在形状对不对，不在算子」
        #   ⇒ 这里宁可 raise，也不让它静默算错（错配的 Linear 会直接抛，但维数刚好相等
        #   而语义错的情况——比如把 latent 当成 hidden 传进来——只有这个检查能拦住）
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"down() 期望最后一维是 hidden_size={self.hidden_size}，"
                f"实际收到 {hidden_states.shape[-1]}（是不是把 latent 当成 hidden 传进来了？）"
            )
        return self.routed_expert_down_proj(hidden_states)

    def up(self, latent_states):
        """从 latent 空间投回 hidden。`[..., mh] → [..., H]`；若启用 norm，则 **up 之前** 做。"""
        if latent_states.shape[-1] != self.moe_hidden_size:
            raise ValueError(
                f"up() 期望最后一维是 moe_hidden_size={self.moe_hidden_size}，"
                f"实际收到 {latent_states.shape[-1]}"
            )
        if self.routed_expert_norm is not None:
            latent_states = self.routed_expert_norm(latent_states)
        return self.routed_expert_up_proj(latent_states)


def latent_moe_forward(hidden_states, gate_fn, moe_fn, proj, shared_fn=None):
    """把官方 `forward`（`modeling_kimi_linear.py:815-838`）的**时序**写成可执行形式。

    三个可插拔点，对应框架侧要换掉的东西：

    | 参数 | 约定 | vLLM/SGLang 侧对应 |
    |---|---|---|
    | `gate_fn(x)` | 返回 `(topk_idx, topk_weight)`；**收到的 x 是投影前的原始 hidden** | 框架的 router/gate |
    | `moe_fn(latent, topk_idx, topk_weight)` | 输入输出都是 **latent 维** | vLLM `FusedMoE`（experts 的 hidden= mh） |
    | `shared_fn(x)` | 可选；**收到原始 hidden**，返回 `[..., H]` | 框架的 shared experts |

    `x` 支持 `[T,H]` 与 `[B,T,H]`（官方是 `view(-1, ...)` 再 `view(*orig_shape)` 还原，
    本函数照做）。
    """
    identity = hidden_states
    orig_shape = hidden_states.shape

    topk_idx, topk_weight = gate_fn(hidden_states)          # ① 投影前（H 维）

    x = hidden_states.view(-1, hidden_states.shape[-1])
    latent = proj.down(x)                                   # ② 进 latent 空间（mh 维）
    y = moe_fn(latent, topk_idx, topk_weight)               #    experts 在 latent 空间工作
    y = proj.up(y)                                          # ③ 回 H 维
    y = y.view(*orig_shape)

    if shared_fn is not None:
        y = y + shared_fn(identity)                         # ④ shared 走原始 hidden，不投影
    return y


def _nbad(a, b, atol=1e-2, rtol=1e-2):
    """标准判据 `±(atol + rtol·|ref|)`（别用纯相对误差，见 `docs/14` §10）。"""
    over = (a.float() - b.float()).abs().gt(atol + rtol * b.float().abs())
    return int(over.sum().item()), a.numel()


def _selfcheck():
    ok = True
    H, MH = 64, 32
    # 形状防线那两项测试靠"维数不等"才能检出错配
    assert H != MH, "自检前提：hidden_size 必须 != moe_hidden_size"

    print("=" * 70)
    print("段 1：不变量探针（纯 torch，不需要 NPU / transformers）")
    print("-" * 70)

    # --- 探针：用 recording 函数，检查「谁收到了什么形状」------------------------
    seen = {}

    def probe_gate(x):
        seen["gate_in"] = tuple(x.shape)
        return (torch.zeros(x.shape[0], 1, dtype=torch.long),
                torch.ones(x.shape[0], 1))

    def probe_moe(latent, _idx, _w):
        seen["moe_in"] = tuple(latent.shape)
        return latent

    def probe_shared(x):
        seen["shared_in"] = tuple(x.shape)
        return torch.zeros_like(x)

    proj = LatentMoeProjection(H, MH, use_norm=True)

    for shape in [(7, H), (2, 3, H)]:                       # 2D 与 3D 都要过
        x = torch.randn(*shape)
        seen.clear()
        y = latent_moe_forward(x, probe_gate, probe_moe, proj, shared_fn=probe_shared)

        checks = [
            ("① gate 收到的是投影前的 H 维", seen["gate_in"][-1] == H),
            ("② moe_fn 收到的是 latent 维 mh", seen["moe_in"][-1] == MH),
            ("③ shared_fn 收到的是原始 H 维", seen["shared_in"][-1] == H),
            (f"④ drop-in：输入输出形状一致 {shape}", tuple(y.shape) == tuple(shape)),
        ]
        for name, passed in checks:
            print(f"  [{'OK' if passed else 'FAIL'}] {name}")
            ok = ok and passed

    # --- 形状防线：错配必须 raise，不能静默算错 ---------------------------------
    bad_cases = [
        ("down 收到 latent 维", lambda: proj.down(torch.randn(4, MH))),
        ("up 收到 hidden 维", lambda: proj.up(torch.randn(4, H))),
    ]
    for name, fn in bad_cases:
        try:
            fn()
            print(f"  [FAIL] {name} —— 应当报错却静默通过了")
            ok = False
        except ValueError:
            print(f"  [OK] {name} —— 已按预期 raise ValueError")

    # --- norm 的位置：必须在 up 之前（官方行 829-832）---------------------------
    latent = torch.randn(4, MH)
    with torch.no_grad():
        proj.routed_expert_norm.weight.fill_(1.3)           # 非平凡权重，否则位置写错也看不出来
        got = proj.up(latent)
        want = proj.routed_expert_up_proj(proj.routed_expert_norm(latent))
        pos_ok = bool(torch.allclose(got, want, atol=0, rtol=0))
    print(f"  [{'OK' if pos_ok else 'FAIL'}] up() 确实应用了 norm"
          f"（漏掉会把 1.3 的缩放丢掉；而它该在 up 之前还是之后，由段 2 的官方对照来判）")
    ok = ok and pos_ok

    print()
    print("=" * 70)
    print("段 2：与官方 KimiSparseMoeBlock 的数值对照（灌同一份权重）")
    print("-" * 70)
    try:
        # ★ fla stub：`modeling_kimi_linear` 顶层 `from fla.* import ...` 会拉起 triton driver，
        #   而本机 triton-ascend 3.2.0 与 CANN 9.0 不配套（编译失败，见 docs/14 §15.3 更正）。
        #   KimiSparseMoeBlock 不使用 fla（只有 KDA 相关类才用）⇒ stub 掉是安全的。
        from unittest import mock
        for _m in ('fla', 'fla.modules', 'fla.ops', 'fla.ops.kda',
                   'fla.ops.utils', 'fla.ops.utils.index', 'fla.utils'):
            if _m not in sys.modules:
                sys.modules[_m] = mock.MagicMock()

        from reference.configuration_kimi_k3 import KimiLinearConfig
        from reference.modeling_kimi_linear import KimiSparseMoeBlock

        config = KimiLinearConfig(
            hidden_size=H,
            routed_expert_hidden_size=MH,       # ← 这一项非 None 才开 latent MoE（官方行 776）
            latent_moe_use_norm=True,
            num_experts=4,
            num_experts_per_token=2,
            moe_intermediate_size=16,
            num_shared_experts=0,               # 0 ⇒ shared 分支贡献恒为 0，两边一致
            num_hidden_layers=2,
        )
        official = KimiSparseMoeBlock(config).eval()

        mine = LatentMoeProjection(H, MH, use_norm=True, eps=config.rms_norm_eps)
        mine.routed_expert_down_proj.weight.data.copy_(official.routed_expert_down_proj.weight.data)
        mine.routed_expert_up_proj.weight.data.copy_(official.routed_expert_up_proj.weight.data)
        mine.routed_expert_norm.weight.data.copy_(official.routed_expert_norm.weight.data)

        torch.manual_seed(0)
        # ⚠️ 必须 3D：官方 gate 的 forward 会解包 3 维（`modeling_kimi_linear.py:704` 的
        #    `bsz, seq_len, h = hidden_states.shape`），传 2D 会直接 ValueError。
        #    （我们的 `latent_moe_forward` 2D/3D 都支持 —— 段 1 两种都测过。）
        x = torch.randn(2, 3, H)
        with torch.no_grad():
            theirs = official(x)
            ours = latent_moe_forward(x, official.gate, official.moe_infer, mine)
        bad, n = _nbad(ours, theirs)
        print(f"  shape {tuple(x.shape)} -> {tuple(ours.shape)}")
        print(f"  vs 官方 KimiSparseMoeBlock（同权重）: 超容差 {bad} / {n} 元素")
        if bad == 0:
            print("  => OK：接线时序与官方一致（gate/down/moe/norm/up 全部对上）")
        else:
            print("  => FAIL：接线与官方不一致，按上面三张表逐项排查")
            ok = False
    except (ImportError, ModuleNotFoundError) as exc:
        print(f"  [SKIP] 段 2 没跑（缺依赖）：{type(exc).__name__}: {exc}")
        print("         需要 transformers + 可导入的 reference/")
        print("         （装法见 ../scripts/l2_mlp_check.py 文件头）。**跳过不算通过**。")
    except Exception as exc:                                # noqa: BLE001
        # ★ 2026-09-19 修正：原先这里把**所有**异常都当 SKIP，结果把真 bug（官方 gate 要求 3D
        #   输入，传 2D 导致 ValueError）显示成"环境问题"，误导了一轮。
        #   现在只有缺依赖才 SKIP，其余一律 FAIL —— 自检不能假装通过。
        import traceback
        traceback.print_exc()
        print(f"  [FAIL] 段 2 出错（不是缺依赖，是接线/用例不对）：{type(exc).__name__}: {exc}")
        ok = False

    print("=" * 70)
    return ok


if __name__ == "__main__":
    sys.exit(0 if _selfcheck() else 1)
