#!/bin/bash
# ---------------------------------------------------------------------------
# SituGluFused「重建版」上机验证：编译 → 安装 → 对拍 →（失败则自动取证）
#
# 为什么要有这个脚本：
#   重建版（见 docs/14 §16）最早是在「宿主沙箱拒绝 bash」的容器里写的，当时一行都没编译过，
#   所以这份脚本的目标是「让上机的人一条命令跑完，并且**每一步都有明确的判定与下一步**」。
#   ★ 2026-09-19 已上机跑通（对拍 PASS，见 docs/14 §16.4）；脚本保留用于重跑与阶段 2 验证。
#
# 用法（不需要可执行位）：
#   bash kimi3_ops/scripts/verify_fused_rebuild.sh                # 小 shape M=8 H=32 d=64
#   bash kimi3_ops/scripts/verify_fused_rebuild.sh 64 256 4096    # 自定义 M H d
#
# 判定与失败分支的完整清单见 kernels/SituGluFused/README.md §6（10 条待验证项）。
# 本脚本只负责「把事实跑出来」，不做任何猜测性修改。
# ---------------------------------------------------------------------------

set -uo pipefail

M="${1:-8}"; H="${2:-32}"; D="${3:-64}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIMI3_ROOT="$(dirname "$SCRIPT_DIR")"
FUSED_DIR="$KIMI3_ROOT/kernels/SituGluFused"
LOG_DIR="${TMPDIR:-/tmp}/situ_glu_fused_verify.$$"     # 放临时目录，不污染仓库
mkdir -p "$LOG_DIR"

CKPT_COMPILE=fail; CKPT_INSTALL=fail; CKPT_VERIFY=fail

info() { printf '\033[36m%s\033[0m\n' "$*"; }
ok()   { printf '\033[32m[ OK ] %s\033[0m\n' "$*"; }
bad()  { printf '\033[31m[FAIL] %s\033[0m\n' "$*"; }
warn() { printf '\033[33m[WARN] %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------------------
info "[0/5] 环境检查"
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  if [ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann-9.0.0/set_env.sh
  fi
fi
if [ -z "${ASCEND_HOME_PATH:-}" ]; then
  bad "ASCEND_HOME_PATH 未设置（先 source \$ASCEND_HOME_PATH/set_env.sh）"
  exit 1
fi
echo "  ASCEND_HOME_PATH = $ASCEND_HOME_PATH"
command -v msopgen >/dev/null 2>&1 && ok "msopgen 在 PATH" || warn "msopgen 不在 PATH（msopgen 生成的工程已存在，编译通常仍可继续）"
echo "  对拍 shape: M=$M H=$H d=$D    日志目录: $LOG_DIR"
if [ $((D % 16)) -ne 0 ]; then
  bad "d=$D 不是 16 的倍数：host 侧会直接 GRAPH_FAILED（重建版的不变量，见 README §5）"
  exit 1
fi

# ---------------------------------------------------------------------------
info "[1/5] 编译融合算子（cd kernels/SituGluFused && bash build.sh）"
( cd "$FUSED_DIR" && bash build.sh ) >"$LOG_DIR/build.log" 2>&1
rc=$?
tail -25 "$LOG_DIR/build.log"
if [ $rc -eq 0 ]; then
  ok "编译 EXIT=0"; CKPT_COMPILE=ok
else
  bad "编译 EXIT=$rc —— 完整日志: $LOG_DIR/build.log"
  echo "     先看日志最后 40 行的第一条 error（模板参数不匹配 / 未定义符号）"
  echo "     对应 README §6 #1；本机可核对头文件: \$ASCEND_HOME_PATH/asc/include/adv_api/matmul/"
fi

# ---------------------------------------------------------------------------
info "[2/5] 安装到 vendor 目录"
RUNFILE="$(ls -t "$FUSED_DIR"/build_out/custom_opp_*.run 2>/dev/null | head -1)"
if [ -z "$RUNFILE" ]; then
  bad "找不到 build_out/custom_opp_*.run 产物（编译没成功？）"
else
  echo "  产物: $RUNFILE"
  "$RUNFILE" --install-path="$ASCEND_HOME_PATH/opp" --quiet >"$LOG_DIR/install.log" 2>&1
  rc=$?
  VENDOR_HDR="$(ls -d "$ASCEND_HOME_PATH"/opp/vendors/*/op_api/include/aclnn_situ_glu_fused.h 2>/dev/null | head -1)"
  if [ -n "$VENDOR_HDR" ]; then
    ok "安装完成：$VENDOR_HDR"; CKPT_INSTALL=ok
  else
    bad "安装后没找到 aclnn_situ_glu_fused.h（install EXIT=$rc，日志 $LOG_DIR/install.log）"
    warn "注意：多个独立算子包会互相覆盖 vendor 目录（本算子自己一个包，见 docs/14 §11）"
  fi
fi

# ---------------------------------------------------------------------------
info "[3/5] 编译对拍程序（纯 aclrt，不需要 torch）"
# ★ 实测修正(2026-09-19)：vendors/ 下可能同时有多个目录（如 custom_transformer + customize），
#   不能简单 head -1 —— 要选**含本算子头文件**的那个。
VENDOR_ROOT=""
for d in "$ASCEND_HOME_PATH"/opp/vendors/*/; do
  if [ -f "$d/op_api/include/aclnn_situ_glu_fused.h" ]; then VENDOR_ROOT="${d%/}"; break; fi
done
if [ -z "$VENDOR_ROOT" ]; then
  bad "找不到含 aclnn_situ_glu_fused.h 的 vendor 目录，跳过对拍"
else
  # ★ 实测修正：链接必须补 -lnnopbase（否则 `DSO missing from command line: libnnopbase.so`）
  g++ -O2 -std=c++17 "$SCRIPT_DIR/verify_situ_glu_fused_aclnn.cpp" -o "$LOG_DIR/verify_sgf" \
      -I"$ASCEND_HOME_PATH/include" \
      -I"$VENDOR_ROOT/op_api/include" \
      -L"$ASCEND_HOME_PATH/lib64" -lascendcl -lnnopbase \
      -L"$VENDOR_ROOT/op_api/lib" -lcust_opapi >"$LOG_DIR/gxx.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    ok "对拍程序编译通过"
  else
    bad "对拍程序编译失败（日志 $LOG_DIR/gxx.log）"
    tail -15 "$LOG_DIR/gxx.log"
  fi
fi

# ---------------------------------------------------------------------------
info "[4/5] 跑对拍（timeout ${TIMEOUT_SEC}s；先让 host 落盘真实 tiling，再按 subN 重排权重）"
rm -f /tmp/situ_glu_fused_tiling.txt
if [ ! -x "$LOG_DIR/verify_sgf" ]; then
  bad "没有对拍可执行文件，跳过"
else
  ASCEND_CUSTOM_OPP_PATH="$VENDOR_ROOT" \
  LD_LIBRARY_PATH="$VENDOR_ROOT/op_api/lib:${LD_LIBRARY_PATH:-}" \
  timeout "$TIMEOUT_SEC" "$LOG_DIR/verify_sgf" "$M" "$H" "$D" 2>&1 | tee "$LOG_DIR/verify.log"
  rc=${PIPESTATUS[0]}
  echo
  case $rc in
    0)  ok "对拍 PASS（nBad == 0，容差 ±(1e-2 + 1e-2·|ref|)）"; CKPT_VERIFY=ok ;;
    1)  if grep -q "判定.*PASS" "$LOG_DIR/verify.log" 2>/dev/null; then
          warn "数值判定 PASS，但后续 bench（重复执行）失败"
          echo "     ⚠️ 已知问题（2026-09-19 实测定性）：第 2 次起报 507057（MTE DDR address out of range）"
          echo "        空 kernel 也复现 ⇒ 与环境/框架层有关，**不影响数值结论**（见 docs/14 §16.4.1）"
          CKPT_VERIFY=ok
        else
          bad "对拍失败 EXIT=$rc（数值不过）"; fi ;;
    124) bad "挂住（timeout ${TIMEOUT_SEC}s 命中，无异常）"
         echo "     → README §6 #2（REGIST_MATMUL_OBJ 的 workspace 是不是症结）"
         echo "     → README §6 #3（GetBlockIdx() 在 AIV 上是否 ∈ [0,aivNum)）"
         echo "     → README §6 #5（SyncAll<false>() 的跨核可见性）"
         echo "     → README §6 #9（删掉 SetBufferSpace 后默认 UB 是否够）" ;;
    *)   bad "对拍失败 EXIT=$rc"
         echo "     → 若日志出现数值超容差: README §6 #4（DataCopyPad 单位/UB 行距）/ #6（C 的行距）"
         echo "     → 若日志出现 507015: 见下面的自动取证" ;;
  esac
fi

# ---------------------------------------------------------------------------
info "[5/5] 失败取证（只在失败时看）"
echo "  --- host 落盘的 tiling（确认 subN/baseM/baseN 是否符合预期）---"
if [ -f /tmp/situ_glu_fused_tiling.txt ]; then cat /tmp/situ_glu_fused_tiling.txt; else warn "没生成（host tiling 没跑到落盘那一步）"; fi

echo "  --- plog 里的设备侧错误（最近 40 行）---"
PLOG_DIR="$HOME/ascend/log/debug/plog"
if [ -d "$PLOG_DIR" ]; then
  grep -rn "fault kernel_name\|mte error info\|vec error info\|errorStr" "$PLOG_DIR" 2>/dev/null | tail -40
  echo "    读法：\"The DDR address of the MTE instruction is out of range\" → 仍是地址问题；"
  echo "          mte/vec error info 全 0 且纯挂死 → 查核配比 / 跨核同步"
else
  warn "没找到 $PLOG_DIR（换个 log 路径：\$HOME/ascend/log/ 或 /var/log/npu/）"
fi

echo "  --- 越界定位工具（按需跑，会生成 info.txt + 单算子复现脚本）---"
echo "     python3 \$ASCEND_HOME_PATH/tools/msaicerr/msaicerr.py --help"
echo "     mssanitizer（挂住类：看 CrossCore flag 配对）"

# ---------------------------------------------------------------------------
echo
info "===== 汇总 ====="
printf '  编译   : %s\n' "$CKPT_COMPILE"
printf '  安装   : %s\n' "$CKPT_INSTALL"
printf '  对拍   : %s\n' "$CKPT_VERIFY"
echo "  日志   : $LOG_DIR"
echo
if [ "$CKPT_VERIFY" = ok ]; then
  echo "  ★ 下一步（顺序不能反）："
  echo "    1) docs/14 §16 的状态已按本次实测同步（✅ 2026-09-19）；若本次 M/N/d 或耗时与 §16.4/§16.4.2 不同，一并更新"
  echo "    2) 再谈收益：README §2 说过，**阶段 1（C 落 workspace）本来就没有 HBM 收益**，"
  echo "       要省掉 [M,2d] 的往返得做阶段 2（照 ffn_nonquant_nz.h 的 GetTensorC 到 UB）"
  echo "    3) 把 scripts/verify_fused_rebuild.sh 的结果贴进 产出总览与索引.md §6 的 P2 ⑧"
else
  echo "  ★ 上机失败不要盲改：先按 README §6 的 10 条定位（每条都写了「如果不符怎么办」）"
fi
