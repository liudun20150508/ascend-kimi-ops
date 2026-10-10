#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 昇腾机器环境探测：一条命令固化「实际设备数 / 单卡显存 / CANN 版本 / 软件栈版本」
#
# 解决的问题：部署方案 §2「64 卡起」的显存账是按公开规格推算的，
#            必须用实机 npu-smi 输出把「单卡显存」这个乘数钉死。
#
# 用法： bash probe_ascend_env.sh                # 打印
#        bash probe_ascend_env.sh > env.txt      # 留档，贴进部署方案
# ---------------------------------------------------------------------------
set -u

section() { printf '\n===== %s =====\n' "$1"; }
have()    { command -v "$1" >/dev/null 2>&1; }

section "驱动与设备（npu-smi）"
if have npu-smi; then
    npu-smi info
    echo "--- 设备节点 ---"
    ls /dev/davinci* /dev/davinci_manager /dev/devmm_svm /dev/hisi_hdc 2>/dev/null || true
else
    echo "npu-smi 不存在（非昇腾机器或驱动未装）"
fi

section "驱动版本"
if have npu-smi; then
    npu-smi info -t board -i 0 2>/dev/null | head -20 || true
    npu-smi info -t health -i 0 2>/dev/null | head -10 || true
fi

section "CANN 版本"
if [ -f /usr/local/Ascend/ascend-toolkit/latest/version.cfg ]; then
    cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
fi
ls -d /usr/local/Ascend/*/ 2>/dev/null || echo "（未发现 /usr/local/Ascend/*）"
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<未设置>}"
echo "--- 编译器 ---"
for c in ccec bisheng msopgen; do
    have "$c" && printf '  %s -> %s\n' "$c" "$(command -v "$c")" || printf '  %s 不可用\n' "$c"
done

section "Python 栈（torch / torch_npu）"
python3 - <<'PY' 2>/dev/null || echo "（python3 不可用）"
for mod in ("torch", "torch_npu"):
    try:
        m = __import__(mod)
        print(f"{mod:10s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{mod:10s} 不可用: {e}")
try:
    import torch, torch_npu  # noqa: F401
    print("npu 设备数:", torch.npu.device_count())
    for i in range(torch.npu.device_count()):
        prop = torch.npu.get_device_properties(i)
        free, total = torch.npu.mem_get_info(i)
        print(f"  dev{i}: {torch.npu.get_device_name(i)}  总显存 {total/2**30:.1f} GiB  空闲 {free/2**30:.1f} GiB")
except Exception as e:
    print("设备查询失败:", e)
PY

section "HCCL / 通信环境"
env | grep -E "^(HCCL_|ASCEND_|RANK_|WORLD_SIZE|MASTER_)" | sort || echo "（无相关环境变量）"
have hccn_tool && { for i in 0 1; do echo "--- dev $i ---"; hccn_tool -i "$i" -link -g 2>&1 | head -6; done; } \
                || echo "hccn_tool 不存在"

section "RDMA（昇腾侧也是网卡的事）"
have ibv_devices && ibv_devices || echo "ibv_devices 不存在（装 rdma-core）"
have hccn_tool || true

section "结论：写回部署方案的三个数"
cat <<'EOF'
需要从上面输出里抄走的三个数（填进 KimiK3-昇腾部署方案 §2）：
  1. 单卡显存 GiB        ____________   （决定「~40 卡」这个乘数）
  2. 实际可见设备数      ____________   （决定最小可行规模）
  3. CANN 版本           ____________   （决定 MXFP4 融合算子要用的 recipe 分支）
EOF
