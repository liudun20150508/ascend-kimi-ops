#!/usr/bin/env bash
# =============================================================================
# 一键验收（把"验证五件套"变成可执行流程，而不是口头的"我验证过"）
#
#   五件套 → 本脚本里的落点
#   ① 判据先行      : 所有判据在 §2 集中声明（跑之前就写好），逐项对照
#   ② 独立第二实现  : C++ 直调（aclrt+aclnn）与 torch_npu 两条路线互证（--full）
#   ③ 负对照        : 内置「错误前提必须报警」的反向检查（scripts/negative_control.py）
#   ④ 证据分级      : 报告里每项标 [实测]/[可推]，并记录环境指纹
#   ⑤ 可复现+留证   : 固定 seed；每项独立日志；失败也保留；一条命令重跑
#
# 用法：
#   bash scripts/run_all_checks.sh              # 全量
#   bash scripts/run_all_checks.sh --quick      # 跳过 L2 与性能基准
#   bash scripts/run_all_checks.sh --full       # 额外编 C++ 直调做"第二实现"互证
#
# 产物：
#   verify_logs/<时间戳>/env.txt      环境指纹
#   verify_logs/<时间戳>/判据.md       判据先行（阈值清单）
#   verify_logs/<时间戳>/*.log        每项日志（失败也保留）
#   verify_logs/<时间戳>/REPORT.md    验收报告
#
# 退出码：0 = 全 PASS（SKIP 不算失败）；1 = 有 FAIL
# =============================================================================
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

QUICK=0; FULL=0
for a in "$@"; do
  case "$a" in
    --quick) QUICK=1 ;;
    --full)  FULL=1 ;;
    *) echo "未知参数: $a"; exit 2 ;;
  esac
done

TS="$(date +%Y%m%d-%H%M%S)"
LOGDIR="$ROOT/verify_logs/$TS"
mkdir -p "$LOGDIR"

PY=/usr/local/python3.11.15/bin/python3
[ -x "$PY" ] || PY=python3
L2PY=/workspace/venv_l2/bin/python
[ -x "$L2PY" ] || L2PY="$PY"
TIMEOUT="${CHECK_TIMEOUT:-1800}"

# SiTU-GLU 达标率判据阈值（只对访存受限的大 case 生效）。默认 80%。
# ⚠️ 实测两条路线存在差异：**C++ 直调 81.4%（达标）** vs **torch_npu 路线 77.6%（未达标）**
#    —— 差异来自调用路径开销与 shape 不同（450 MB vs 376 MB 搬运）。
#    判据**不为通过而下调**；确需放松就显式 `SITU_MIN_UTIL=75`，并在报告里写明原因。
SITU_MIN_UTIL="${SITU_MIN_UTIL:-80}"

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}"
for _c in /workspace/ops-transformer/torch_extension/cann_ops_transformer \
          /tmp/ops-transformer/torch_extension/cann_ops_transformer; do
  if [ -d "$_c" ]; then export OPS_TRANSFORMER_PKG="$_c"; break; fi
done
export KIMI3_VENDOR="${KIMI3_VENDOR:-$ASCEND_HOME_PATH/opp/vendors/customize}"
export ASCEND_CUSTOM_OPP_PATH="$KIMI3_VENDOR"
export LD_LIBRARY_PATH="$KIMI3_VENDOR/op_api/lib:${LD_LIBRARY_PATH:-}"

# ---- §1 环境指纹（数字离开环境就没有意义）----------------------------------
{
  echo "# 环境指纹"
  date -u
  echo; echo "## SOC / 卡"
  npu-smi info 2>&1 | head -18
  echo; echo "## 软件版本"
  "$PY" - <<'PYEOF' 2>&1 | tail -4
import torch, torch_npu
print("torch", torch.__version__, "| torch_npu", torch_npu.__version__,
      "| devices", torch.npu.device_count(), "|", torch.npu.get_device_name(0))
PYEOF
  echo; echo "## 自定义算子（vendor）"
  ls "$KIMI3_VENDOR/op_api/include/" 2>&1
} > "$LOGDIR/env.txt" 2>&1

# ---- §2 判据先行 -----------------------------------------------------------
cat > "$LOGDIR/判据.md" <<'EOF'
| 层 | 检查项 | 判据（跑之前就写好的阈值） | 证据 |
|---|---|---|---|
| 单元 | SiTU-GLU 对拍 | 全部 case `max_rel < 1e-2` | 实测 |
| 单元 | AttnRes 对拍 | 超容差 = 0，判据 `±(atol + rtol·\|ref\|)` | 实测 |
| 单元 | 融合算子对拍 | 超容差 = 0（4 组 shape） | 实测 |
| 单元 | 融合算子索引 + 反向自证 | PASS，且**反向必须 FAIL**（否则检查无分辨力） | 实测 |
| 单元 | **负对照** | 故意用错参数 ⇒ **必须超容差** | 实测 |
| 集成 | 真实模型层（官方 KimiMLP 同权重） | 超容差 = 0 | 实测 |
| 性能 | 本卡峰值带宽 | copy 口径，仅作**分母声明** | 实测 |
| 性能 | SiTU-GLU 达标率 | 中位达标率 ≥ 80%（**仅对访存受限成立**，见下注） | 实测 |
| 性能 | 相对华为（内置 SwiGLU） | **只报同口径时间/带宽比**；达标率 >100% 触发判据报警 | 实测 |
| 系统 | 端到端 TTFT/TPOT | 需推理框架（当前外部依赖阻塞）| —— |

> ★ **判据的前提也要被验证**：`≥80% 带宽达标率` 只对**访存受限**的算子成立。
> 本项目实测发现 SiTU-GLU 的 `aiv_vec_time` 占 **93.5%**（计算受限）——
> 也就是说这个阈值对它**不适用**，正确判据是 V 吞吐（见 `docs/04` §1 的更正）。
EOF

PASS_N=0; FAIL_N=0; SKIP_N=0
ROWS=""

run_step() {  # run_step <slug> <层> <名称> <判据> <通过标志正则|-> <命令...>
  local slug="$1" layer="$2" name="$3" crit="$4" flag="$5"; shift 5
  local log="$LOGDIR/$slug.log"
  printf '▶ %-52s ... ' "$name"
  local rc=0
  ( timeout "$TIMEOUT" "$@" ) > "$log" 2>&1 || rc=$?
  local res
  if [ $rc -eq 0 ]; then res="PASS"; else res="FAIL"; fi
  if [ "$res" = "PASS" ] && [ "$flag" != "-" ] && ! grep -qE "$flag" "$log"; then
    res="FAIL(无通过标志)"; echo "⚠️  "
  fi
  case "$res" in
    PASS) PASS_N=$((PASS_N+1)) ;;
    *)    FAIL_N=$((FAIL_N+1)) ;;
  esac
  printf '%s (rc=%d) → %s.log\n' "$res" "$rc" "$slug"
  ROWS="$ROWS| $layer | $name | $crit | **$res** | $slug.log |"$'\n'
}

skip_step() {  # skip_step <层> <名称> <原因>
  printf '⊘ %-52s ... SKIP (%s)\n' "$2" "$3"
  SKIP_N=$((SKIP_N+1))
  ROWS="$ROWS| $1 | $2 | — | **SKIP**：$3 | — |"$'\n'
}

echo "==== 一键验收 $(date -u +%H:%M:%S) ===="
echo "日志目录: $LOGDIR"; echo

# ---- §3 单元层 -------------------------------------------------------------
run_step "unit_situ"   "单元" "SiTU-GLU 对拍"      "max_rel < 1e-2"    "ALL PASS" "$PY" test_situ_real.py
run_step "unit_attn"   "单元" "AttnRes 对拍"       "超容差 = 0"         "ALL PASS" "$PY" test_attn_res_real.py
run_step "unit_fused"  "单元" "融合算子对拍"        "超容差 = 0 (4 shape)" "ALL PASS" "$PY" test_situ_glu_fused_real.py
run_step "unit_index"  "单元" "融合算子索引+反向自证" "PASS 且反向必须 FAIL" "✅ PASS"  "$PY" scripts/check_fused_index.py
run_step "unit_negctl" "单元" "★负对照（错误参数必须超容差）" "必须报警" "✅ PASS"  "$PY" scripts/negative_control.py

# ---- §4 集成层（真实模型层）------------------------------------------------
if [ "$QUICK" = "1" ]; then
  skip_step "集成" "L2 真实模型层（官方 KimiMLP 同权重）" "--quick 跳过"
else
  run_step "l2_mlp" "集成" "L2 真实模型层（官方 KimiMLP 同权重）" "超容差 = 0" "与原生一致" \
    "$L2PY" scripts/l2_mlp_check.py
fi

# ---- §5 性能（判据：访存型算子达标率 ≥80%）--------------------------------
if [ "$QUICK" = "1" ]; then
  skip_step "性能" "峰值带宽 + SiTU-GLU 达标率" "--quick 跳过"
  skip_step "性能" "相对华为：内置 SwiGLU 同口径对测" "--quick 跳过"
  PEAK=""
else
  "$PY" bench_situ_real.py --bench-bw-only > "$LOGDIR/peak.log" 2>&1
  PEAK="$(grep -oE '[0-9]+\.[0-9]+ GB/s' "$LOGDIR/peak.log" | head -1 | cut -d' ' -f1)"
  if [ -z "$PEAK" ]; then
    skip_step "性能" "峰值带宽" "未解析到（见 peak.log）"
    PEAK=""
  else
    echo "本卡峰值带宽（copy 口径）: $PEAK GB/s"
    ROWS="$ROWS| 性能 | 峰值带宽（copy 口径） | 仅作分母声明 | **[实测] $PEAK GB/s** | peak.log |"$'\n'
    run_step "bench_situ" "性能" "SiTU-GLU 达标率" "适用域（搬运≥100MB）达标率 ≥ ${SITU_MIN_UTIL}%" "-" "$PY" bench_situ_real.py --peak-bw "$PEAK" --assert-util "100:${SITU_MIN_UTIL}"
    run_step "bench_swiglu" "性能" "相对华为：内置 SwiGLU 同口径对测" "只报同口径比" "-" \
      "$PY" scripts/bench_swiglu_vs_situ.py --peak-bw "$PEAK"
  fi
fi

# ---- §5.5 系统层（mini 引擎 smoke：TTFT/TPOT 能真的跑出来）-----------------
if [ "$QUICK" = "1" ]; then
  skip_step "系统" "mini 推理引擎（TTFT/TPOT/吞吐）" "--quick 跳过"
else
  run_step "sys_mini" "系统" "mini 推理引擎（TTFT/TPOT/吞吐）" "跑通并出 TTFT/TPOT" "TPOT" \
    "$PY" scripts/mini_engine_bench.py --preset tiny --iters 3
fi

# ---- §6 独立第二实现（C++ 直调，需 --full）---------------------------------
if [ "$FULL" = "1" ]; then
  run_step "impl_cpp" "第二实现" "C++ 直调（aclrt+aclnn）编译并运行" "EXIT=0" "-" bash -c '
    g++ -O2 -std=c++17 scripts/bench_situ_aclnn.cpp \
      -I"$ASCEND_HOME_PATH/include" -I"$KIMI3_VENDOR/op_api/include" \
      -L"$ASCEND_HOME_PATH/lib64" -L"$KIMI3_VENDOR/op_api/lib" \
      -lnnopbase -lcust_opapi -lascendcl -o /tmp/bench_situ_aclnn && /tmp/bench_situ_aclnn'
else
  skip_step "第二实现" "C++ 直调（aclrt+aclnn）" "需 --full（会调用 g++）"
fi

# ---- §7 判据复核 + 报告 ----------------------------------------------------
EXTRA=""
if [ -f "$LOGDIR/bench_situ.log" ]; then
  # 判据由**被测脚本自己**断言（--assert-util），这里只转述结论，避免二次判定
  # （踩过的坑：拿跨全部 shape 的中位达标率当判据 ⇒ 被小 shape 拉成 0.9% 而误报 FAIL）
  CRIT="$(grep -E '^\[判据\]' "$LOGDIR/bench_situ.log" | head -1)"
  VERDICT="$(grep -E '全部满足|判据未满足' "$LOGDIR/bench_situ.log" | head -1)"
  if [ -n "$CRIT" ]; then EXTRA="${EXTRA}- $CRIT $VERDICT"$'\n'; fi
fi
if [ -f "$LOGDIR/bench_swiglu.log" ]; then
  SW="$(grep -oE '耗时比（自研 / 内置） +: [0-9.]+x' "$LOGDIR/bench_swiglu.log" | head -1)"
  RATIO="$(grep -oE '自研 / 内置） +: [0-9.]+x' "$LOGDIR/bench_swiglu.log" | head -1 | grep -oE '[0-9]+\.?[0-9]*x')"
  [ -n "$RATIO" ] && EXTRA="$EXTRA- 大 shape 下「自研 / 内置」耗时比 = **$RATIO**（>1 = 内置更快）"$'\n'
  grep -q '口径自检' "$LOGDIR/bench_swiglu.log" && \
    EXTRA="$EXTRA- ⚠️ **口径自检被触发**（有算子达标率 >100% ⇒ 分母口径不适用于该算子，只能同口径横比）"$'\n'
fi

{
  echo "# 验收报告（$TS）"
  echo
  echo "- 环境指纹：\`env.txt\`　判据先行：\`判据.md\`"
  echo
  echo "## 结果"
  echo
  echo "| 层 | 检查项 | 判据 | 结果 | 日志 |"
  echo "|---|---|---|---|---|"
  printf '%s' "$ROWS"
  echo
  if [ -n "$EXTRA" ]; then echo "## 判据复核"; echo; printf '%s' "$EXTRA"; echo; fi
  echo "## 汇总"
  echo
  echo "- **PASS**：$PASS_N　**FAIL**：$FAIL_N　**SKIP**：$SKIP_N"
  echo "- SKIP 不算失败，但**必须写明原因**；空缺就写空缺，别编"
  echo
  echo "## 这套流程在证明什么"
  echo
  echo "1. **判据先行**：阈值在跑之前就写死在 \`判据.md\`，不是跑完再挑一个好看的数"
  echo "2. **负对照（最重要）**：故意用错参数，判据**必须报警**；报不出来说明判据没分辨力，前面的 PASS 都不可信"
  echo "3. **独立第二实现**：C++ 直调与 torch_npu 两条互不依赖的路线互证（\`--full\`）"
  echo "4. **证据分级 + 溯源**：每项一个日志，失败也保留 ⇒ 可事后取证"
  echo "5. **可复现**：固定 seed + 一条命令重跑；环境指纹随报告存档"
} > "$LOGDIR/REPORT.md"

echo
echo "==== 汇总：PASS=$PASS_N FAIL=$FAIL_N SKIP=$SKIP_N ===="
echo "报告: $LOGDIR/REPORT.md"
[ "$FAIL_N" -eq 0 ] && exit 0 || exit 1
