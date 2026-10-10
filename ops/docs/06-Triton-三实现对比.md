# 三实现对比：CANN 内置 / AscendC 自研 / Triton

> 待办 #6：「用 Triton 重写一个算子，产出三实现对比」。
> 选 SiTU-GLU 是因为它正好是三实现都能做的算子，而且**第一极是缺的**
> （华为只有 SiLU 融合，没有 SiTU 融合）—— 对比才有信息量。

## 三份实现分别在哪

| 实现 | 位置 | 能否跑 |
|---|---|---|
| ① CANN 内置 | 无 SiTU 融合算子（只有 `SiLU`/`SwiGLU` 系列） | 只能证明「缺」 |
| ② AscendC 自研 | `../kernels/SituAndMul/`（kernel + tiling + host） | 需昇腾 A3 + CANN |
| ③ Triton | `../triton_kernels/situ_glu_triton.py` | 需 GPU + triton（昇腾侧 `triton-ascend`） |

数学定义三者必须**逐行一致**（这也是验证方法：三份实现在同一组随机输入上互相做 golden）：

```
gate, up = x[..., :d], x[..., d:]
situ(gate) = beta * tanh(gate / beta) * sigmoid(gate)     # beta = 4.0
up'        = linear_beta * tanh(up / linear_beta)          # linear_beta = 25.0
y          = situ(gate) * up'
```

## 对比维度（问答真正会问的三件事）

| 维度 | ① CANN 内置 | ② AscendC | ③ Triton |
|---|---|---|---|
| **要不要写 kernel** | 不用，但**没有这个算子** | 要写：kernel + tiling + host + wrapper | 只写「逐元素程序」，tiling/流水全由编译器决定 |
| **能否控制 UB/流水** | 不可见 | 完全可控（`TQue` depth、`tileElem`、核间切分） | 不可控（靠 `BLOCK`/`num_warps` 间接影响） |
| **性能上限** | 无可比对象 | 上限最高（能到带宽墙的 80%+，见 `docs/04`） | 通常 80~95% 于手写 kernel，但开发成本 1/5 |
| **可移植性** | 锁 CANN | 锁昇腾（换 SOC 要调 tiling 常数） | 换后端只需换编译器（NVIDIA/AMD/**昇腾 triton-ascend**） |
| **首次开发时间** | —— | 天级（要理解三级存储 + 同步） | 小时级 |
| **适合什么时候用** | 生态已有的活别重造 | 补缺算子 / 要压到接近带宽上限 | 快速验证收益、算法还在迭代的阶段 |

**结论（一句话，可以直接背）：**

> **Triton 用来「先验证这条路值不值得做」，AscendC 用来「把收益榨干到最后一滴」。**
> 因为 Triton 把 tiling、流水、同步都藏进编译器，你能五分钟验证一个融合想法；
> 但当你要控制 `TQue` 深度、UB 分块、核间负载均衡时，只有 AscendC 这一层能给。
> 我选 SiTU-GLU 做这个对比，就是因为它既证明了「Triton 能快速复现数学」，
> 也证明了「要做到带宽达标率 80%+ 还是得回到 AscendC」。

## 对拍方法（三实现互证）

```bash
# 基准（golden）：纯 torch 参考实现
python3 tests/test_situ.py                     # 参考实现自测

# ② AscendC（昇腾 A3）
python3 bench_situ_real.py                     # 正确性 + 带宽达标率

# ③ Triton（GPU）
python3 triton_kernels/situ_glu_triton.py              # 正确性 + 带宽达标率
```

三者都用同一判据：`max_rel < 1e-2`（bf16 内部 fp32 计算），
`rel = |out - ref| / (|ref| + 1e-3)` —— 加绝对项是为了避免参考值跨零时的假失败（见 `docs/04`）。

## 昇腾侧跑 Triton 的注意点

昇腾用 `triton-ascend`（昇腾自研后端），差异主要在三处：

1. `device="cuda"` → `device="npu"`，计时用 `torch.npu.Event`；
2. `tl.math.tanh` 是否可用取决于后端实现的 libdevice 覆盖范围 —— 不可用时用
   `2*sigmoid(2x) - 1` 恒等式绕开（`tanh(x) = 2σ(2x) − 1`），数值上也稳定；
3. `BLOCK` 的最优值与 UB 容量相关：`BLOCK` 太大反而因寄存器/UB 压力掉速，
   需要用 `--block` 扫 256/512/1024/2048 找拐点（与 AscendC 的 `tileElem` 是同一件事）。

> 第 3 点正好可以回答「Triton 能不能替代手写 kernel」：
> **连一个纯 elementwise 算子都有 `BLOCK` 这个需要手调的参数**，
> 只是它把「UB 分块 + 流水 + 同步」压缩成了一个数字 —— 这就是它快的原因，也是它的天花板。
