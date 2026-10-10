#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# P0 一键跑：在 A3 机器上把「现在就能做」的全做一遍，输出汇总
#
#   bash kimi3_ops/scripts/run_p0.sh 2>&1 | tee /tmp/p0.log
#
# 设计原则：
#   · 每步先检查前置，不满足就跳过并说明缺什么（不猜、不硬跑）
#   · 只读 + 只在工程的 build_out 里写编译产物，**不改任何源文件**
#   · 失败的步骤继续往下走，最后统一汇总，方便一次性把 log 贴回来
# ---------------------------------------------------------------------------
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIMI3="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$KIMI3" || exit 1

OK="[ OK ]"; NG="[FAIL]"; SKIP="[SKIP]"
PASS_LIST=(); FAIL_LIST=(); SKIP_LIST=()

step()  { printf '\n\033[1m========== %s ==========\033[0m\n' "$1"; }
have()  { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- 0. 体检
step "0. 环境体检（preflight_check.sh）"
if [ -f "$SCRIPT_DIR/preflight_check.sh" ]; then
    bash "$SCRIPT_DIR/preflight_check.sh"
    PASS_LIST+=("preflight 已跑（看上面 SOC / 依赖结论）")
else
    FAIL_LIST+=("preflight_check.sh 不存在"); echo "$NG 找不到 preflight_check.sh"
fi

# ---------------------------------------------- 1. 纯 Python 工具（不依赖 NPU）
step "1. 纯 Python 工具（不依赖 NPU，立刻能跑）"
if have python3; then
    for s in benefit_total.py mxfp4_dequant_check.py quant_lab.py kv_cache_opt.py \
             fusion_tiling_calc.py interleave_weight.py; do
        if [ -f "$s" ]; then
            echo "--- $s ---"
            if python3 "$s" >/tmp/p0_$(basename "$s").log 2>&1; then
                head -25 /tmp/p0_$(basename "$s").log
                echo "  （完整输出: /tmp/p0_$(basename "$s").log）"
                PASS_LIST+=("$s")
            else
                echo "$NG $s 运行失败，末尾 20 行："
                tail -20 /tmp/p0_$(basename "$s").log
                FAIL_LIST+=("$s 运行失败")
            fi
        else
            SKIP_LIST+=("$s 不存在")
        fi
    done
else
    FAIL_LIST+=("python3 不可用"); echo "$NG python3 不可用"
fi

# ------------------------------------------------------- 2. 编译 SituAndMul
step "2. 编译 SituAndMul（会暴露所有笔误 —— 这是真正的验收点）"
SOC_HINT=""
if [ -d /usr/local/Ascend/driver/device ]; then
    fw=$(ls /usr/local/Ascend/driver/device 2>/dev/null | grep -m1 -E '^ascend_.*_device' || true)
    case "$fw" in
        ascend_910_93_*) SOC_HINT="ascend910_93" ;;
        ascend_910b*)    SOC_HINT="ascend910b" ;;
        *)               SOC_HINT="" ;;
    esac
fi
echo "本机 SOC 判定: ${SOC_HINT:-<未判定>}"

if [ -n "$SOC_HINT" ] && [ -f "kernels/SituAndMul/CMakePresets.json" ]; then
    if ! grep -q "\"$SOC_HINT\"" kernels/SituAndMul/CMakePresets.json; then
        echo "$NG CMakePresets 里的 SOC 与本机不一致 —— 先改这两处再编译："
        echo "    sed -i 's/ascend910_93/$SOC_HINT/g' kernels/*/CMakePresets.json"
        echo "    sed -i 's/ascend910_93/$SOC_HINT/g' kernels/*/op_host/*.cpp"
        SKIP_LIST+=("编译（SOC 不匹配，需先改）")
    else
        if [ -z "${ASCEND_HOME_PATH:-}" ]; then
            echo "提示: 未设 ASCEND_HOME_PATH，尝试 source set_env.sh"
            # shellcheck disable=SC1091
            source /usr/local/Ascend/cann-9.0.0/set_env.sh 2>/dev/null || true
        fi
        ( cd kernels/SituAndMul && bash build.sh ) >/tmp/p0_build.log 2>&1
        if [ $? -eq 0 ]; then
            echo "$OK 编译成功"
            ls -la kernels/SituAndMul/build_out/*.run 2>/dev/null | head -3
            PASS_LIST+=("SituAndMul 编译")
            # 安装（若用户想装，取消下面注释）
            # ( cd kernels/SituAndMul/build_out && ./custom_opp_*.run --install-path=$ASCEND_HOME_PATH/opp --quiet )
        else
            echo "$NG 编译失败，末尾 40 行："
            tail -40 /tmp/p0_build.log
            FAIL_LIST+=("SituAndMul 编译失败（见 /tmp/p0_build.log）")
        fi
    fi
else
    SKIP_LIST+=("编译（SOC 未判定或缺 CMakePresets）")
fi

# --------------------------------------------------------- 3. 对拍 + 达标率
step "3. 正确性对拍 + 带宽达标率"
TORCH_OK=0
if python3 - <<'PY' >/tmp/p0_torch.log 2>&1
import torch, torch_npu
assert torch.npu.device_count() > 0
print("torch_npu ok, devices =", torch.npu.device_count())
PY
then
    TORCH_OK=1
fi
if [ "$TORCH_OK" -eq 1 ]; then
    cat /tmp/p0_torch.log
    echo "--- test_situ_real.py ---"
    python3 test_situ_real.py 2>&1 | tail -15
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then
        PASS_LIST+=("test_situ_real 对拍")
    else
        FAIL_LIST+=("test_situ_real 失败（注意上面输出里的 FAIL 行）")
    fi
    echo "--- bench_situ_real.py --bench-bw-only ---"
    python3 bench_situ_real.py --bench-bw-only 2>&1 | tail -5
    if [ "${PIPESTATUS[0]}" -ne 0 ]; then
        FAIL_LIST+=("bench 峰值带宽失败")
    fi
else
    echo "$SKIP torch_npu 不可用（末 5 行）："
    tail -5 /tmp/p0_torch.log
    echo "  装法（版本要与 CANN 9.0.0 配套，查官方版本表）："
    echo "    pip install torch torch_npu"
    SKIP_LIST+=("对拍 + 达标率（缺 torch_npu）")
fi

# ---------------------------------------------------------------- 汇总
step "汇总"
echo "  ✅ 通过:"
for x in "${PASS_LIST[@]:-}"; do [ -n "$x" ] && echo "     - $x"; done
echo "  ⚠️  失败:"
for x in "${FAIL_LIST[@]:-}"; do [ -n "$x" ] && echo "     - $x"; done
echo "  ⏭️  跳过:"
for x in "${SKIP_LIST[@]:-}"; do [ -n "$x" ] && echo "     - $x"; done
echo
echo "把整份 log 贴回来，我按失败项给下一步。"
echo "（本脚本不修改任何源文件；编译产物只在 kernels/*/build_out/ 下）"
