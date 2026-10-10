#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 上机前置检查（A2 / A3 通用）—— 一条命令回答「这台机器能不能直接跑本项目」
#
# 用法：
#   bash preflight_check.sh              # 检查 + 给结论
#   bash preflight_check.sh | tee preflight.txt
#
# 它自动判定三件最容易出错的事：
#   ① 这台机器的 SOC 是什么 → 算子的 AddConfig 该写哪个（写错就是"编译过、跑不起来"）
#   ② CANN / 编译器 / torch_npu 是否齐 → 不齐就别急着 build.sh
#   ③ 项目依赖（aclnn_common.h）在哪 → 设 OPS_TRANSFORMER_PKG（路径解析统一在 aclnn_ops.py）
# ---------------------------------------------------------------------------
set -u

OK="[ OK ]"; NG="[FAIL]"; WARN="[WARN]"
section() { printf '\n===== %s =====\n' "$1"; }
have()    { command -v "$1" >/dev/null 2>&1; }

SOC_GUESS=""
CANN_DIR=""

# ---------------------------------------------------------------- 1. 设备与驱动
section "1. 设备与驱动"
if have npu-smi; then
    npu-smi info 2>&1 | head -25
else
    echo "$NG npu-smi 不存在（驱动未装或非昇腾机器）"
fi
echo "--- 设备节点 ---"
ls /dev/davinci* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc 2>/dev/null | head -20 \
    || echo "$NG 缺少 /dev/davinci* 等节点"

section "驱动版本"
if [ -f /usr/local/Ascend/driver/version.info ]; then
    cat /usr/local/Ascend/driver/version.info | head -6
else
    echo "$WARN 未找到 /usr/local/Ascend/driver/version.info"
fi

# ------------------------------------------------------- 2. SOC 自动判定（关键）
section "2. SOC 判定（决定 AddConfig 写什么）"
# 最可靠的办法：驱动自带的固件包名就是 SOC 名
if [ -d /usr/local/Ascend/driver/device ]; then
    fw=$(ls /usr/local/Ascend/driver/device 2>/dev/null | grep -m1 -E '^ascend_.*_device' || true)
    echo "固件样本: ${fw:-<未找到>}"
    case "$fw" in
        ascend_910_93_*) SOC_GUESS="ascend910_93" ;;
        ascend_910b*)    SOC_GUESS="ascend910b" ;;
        ascend_910a*)    SOC_GUESS="ascend910a" ;;
        ascend_310p*)    SOC_GUESS="ascend310p" ;;
        ascend_310b*)    SOC_GUESS="ascend310b" ;;
        ascend_950*)     SOC_GUESS="ascend950" ;;
        *)               SOC_GUESS="" ;;
    esac
fi
# 交叉验证：CANN 内置 tiling/config 的 SOC 目录
if [ -d /usr/local/Ascend/cann-*/opp/built-in/data/tiling ]; then
    echo "CANN 内置 tiling 的 SOC: $(ls /usr/local/Ascend/cann-*/opp/built-in/data/tiling 2>/dev/null | tr '\n' ' ')"
fi
if [ -n "$SOC_GUESS" ]; then
    echo "$OK 判定 SOC = $SOC_GUESS"
    echo "     → 本项目的 AddConfig / CMakePresets 应写: $SOC_GUESS"
    echo "       （kernels/*/op_host/*.cpp 与 kernels/*/CMakePresets.json 各一处）"
else
    echo "$WARN 未能自动判定 SOC，看上面 npu-smi 的 Name 列人工确认"
    echo "     对照: A2=ascend910b / A3=ascend910_93 / 310P=ascend310p"
fi

# -------------------------------------------------------------- 3. CANN 与编译器
section "3. CANN 与编译器"
for d in /usr/local/Ascend/cann-* /usr/local/Ascend/ascend-toolkit/latest; do
    [ -d "$d" ] && CANN_DIR="$d" && break
done
if [ -n "$CANN_DIR" ]; then
    echo "$OK CANN 目录: $CANN_DIR"
    [ -f "$CANN_DIR/version.cfg" ] && cat "$CANN_DIR/version.cfg" | head -3
    echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<未设置，需 source $CANN_DIR/set_env.sh>}"
else
    echo "$NG 未找到 CANN 安装目录"
fi
for c in ccec bisheng msopgen cmake; do
    if have "$c"; then printf '  %s -> %s\n' "$c" "$(command -v "$c")"; else printf '  %s 不可用\n' "$c"; fi
done
[ -f "$CANN_DIR/tools/tikcpp/ascendc_kernel_cmake/fwk_modules/util/preset_parse.py" ] \
    && echo "$OK build.sh 依赖的 preset_parse.py 存在" \
    || echo "$WARN preset_parse.py 未找到（build.sh 可能走 cmake --preset 分支）"

# -------------------------------------------------------------- 4. Python 栈
section "4. Python 栈"
python3 - <<'PY' 2>/dev/null || echo "$NG python3 不可用"
import sys
print("python", sys.version.split()[0])
for mod in ("torch", "torch_npu", "numpy"):
    try:
        m = __import__(mod)
        print(f"  {mod:10s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"  {mod:10s} 不可用: {type(e).__name__}")
try:
    import torch, torch_npu
    print(f"  npu 可见设备数 = {torch.npu.device_count()}")
except Exception:
    print("  （torch_npu 不可用，算子 harness / aclnn_ops 都跑不了）")
PY

# ---------------------------------------------------------- 5. 项目依赖路径
section "5. 项目依赖（aclnn_ops.py 解析的两个路径）"
echo "找 aclnn_common.h（来自 ops-transformer，不是 torch_npu 自带）:"
found=$(find / -name aclnn_common.h -not -path '*/proc/*' 2>/dev/null | head -3 || true)
if [ -n "$found" ]; then
    echo "$OK $found"
    echo "     → 设 OPS_TRANSFORMER_PKG=<上面那个文件的 common/inc 的上一级>（路径解析统一在 aclnn_ops.py）"
else
    echo "$NG 未找到 aclnn_common.h —— 需要 ops-transformer 仓库，否则 wrapper JIT 编译会失败"
    echo "     （这是 docs/01 记过的坑：ACLNN_CMD 宏由它提供）"
fi
echo "自定义算子安装目录:"
ls -d /usr/local/Ascend/*/opp/vendors/customize 2>/dev/null || echo "  （尚无，build.sh + 安装 .run 之后才会有）"

# ------------------------------------------- 6. 项目 SOC 设置 vs 本机（告诉你要改哪）
section "6. 项目 SOC 设置 vs 本机"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIMI3="$(cd "$SCRIPT_DIR/.." && pwd)"           # <repo>/kimi3_ops/scripts -> <repo>/kimi3_ops
PRESET="$KIMI3/kernels/SituAndMul/CMakePresets.json"
if [ -f "$PRESET" ]; then
    cur=$(grep -o '"value"[[:space:]]*:[[:space:]]*"ascend[^"]*"' "$PRESET" | head -1 \
          | sed 's/.*"\(ascend[^"]*\)"/\1/')
    echo "项目里 ASCEND_COMPUTE_UNIT = ${cur:-<未找到>}"
    echo "本机判定 SOC               = ${SOC_GUESS:-<未判定>}"
    if [ -n "$SOC_GUESS" ] && [ -n "$cur" ] && [ "$SOC_GUESS" != "$cur" ]; then
        echo ""
        echo "$WARN SOC 不一致，上机前必须先改，否则编译能过但产物跑不起来"
        echo "     （就是 docs/01 记过的 'does not has any binary' 那个坑）："
        echo ""
        echo "  sed -i 's/$cur/$SOC_GUESS/g' $KIMI3/kernels/*/CMakePresets.json"
        echo "  sed -i 's/$cur/$SOC_GUESS/g' $KIMI3/kernels/*/op_host/*.cpp"
        echo ""
        echo "  另需确认 CMakePresets.json 里的 ASCEND_CANN_PACKAGE_PATH 指向本机 CANN："
        grep -o '"value"[[:space:]]*:[[:space:]]*"/usr/local/Ascend[^"]*"' "$PRESET" | head -1
    elif [ -n "$cur" ]; then
        echo "$OK SOC 一致，不用改"
    fi
else
    echo "$WARN 未找到 $PRESET（脚本被拷到别处了？手动核对 kernels/*/CMakePresets.json）"
fi

# ---------------------------------------------------------------- 7. 结论
section "7. 结论"
cat <<EOF
  能不能直接往下走，按这三条对：

  ① SOC：${SOC_GUESS:-<未判定>}
     → 若与 kernels/*/CMakePresets.json 里的值不一致，先改（两处：CMakePresets + op_host 的 AddConfig）

  ② CANN + 编译器齐了？
     → 齐:  cd kimi3_ops/kernels/SituAndMul && bash build.sh
     → 不齐: 先装 CANN（ccec / bisheng / msopgen 缺一不可）

  ③ torch_npu + aclnn_common.h 齐了？
     → 齐:  cd kimi3_ops && python test_situ_real.py        # 先过正确性
     → 不齐: 只做 ②（编译），对拍等环境补齐再跑

  正确性过了之后才是性能：
      python bench_situ_real.py --bench-bw-only
      python bench_situ_real.py --peak-bw <上一步输出的 GB/s>
EOF
printf '\n（本脚本只读，不修改任何文件）\n'
