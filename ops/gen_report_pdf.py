#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成《KIMI3 昇腾 A3 算子开发 设计·验证·测试报告》PDF。

用 reportlab 的 UnicodeCIDFont('STSong-Light')（CID 中文字体，无需 TTF 文件，
PDF 查看器自带该字体）。
"""
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
F = 'STSong-Light'

TITLE = ParagraphStyle('title', fontName=F, fontSize=19, leading=26, alignment=1, spaceAfter=4)
SUB = ParagraphStyle('sub', fontName=F, fontSize=9.5, leading=14, alignment=1, textColor=colors.HexColor('#555555'), spaceAfter=10)
H1 = ParagraphStyle('h1', fontName=F, fontSize=14, leading=19, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor('#1a1a1a'))
H2 = ParagraphStyle('h2', fontName=F, fontSize=11.5, leading=16, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor('#333333'))
BODY = ParagraphStyle('body', fontName=F, fontSize=10, leading=15.5, spaceAfter=3)
CODE = ParagraphStyle('code', fontName=F, fontSize=8.8, leading=13, leftIndent=10, backColor=colors.HexColor('#f4f4f4'), borderPadding=5, spaceAfter=4)
BULLET = ParagraphStyle('bullet', fontName=F, fontSize=10, leading=15, leftIndent=14, bulletIndent=4, spaceAfter=2)

def P(text, style=BODY):
    return Paragraph(text, style)

def bullets(items):
    return [Paragraph(f'• {x}', BULLET) for x in items]

doc = SimpleDocTemplate(
    '/workspace/kimi3_ops/设计验证测试报告.pdf',
    pagesize=A4, topMargin=18*mm, bottomMargin=16*mm, leftMargin=16*mm, rightMargin=16*mm,
    title='KIMI3 昇腾 A3 算子开发 设计·验证·测试报告')

story = []
story.append(P('KIMI3 昇腾 A3 算子开发<br/>设计 · 验证 · 测试报告', TITLE))
story.append(P('目标设备：昇腾 A3（Ascend 910_9382，SOC ascend910_93） · CANN 9.0.0 · torch_npu 2.10.0', SUB))
story.append(Spacer(1, 6))

# 1 概述
story.append(P('1. 概述', H1))
story.append(P('1.1 背景', H2))
story.append(P('KIMI3 是 MoonshotAI 的混合架构大模型（MLA / KDA 线性注意力 + MoE）。在昇腾 NPU 上部署时，需要把模型每一层映射到 CANN 算子。经与华为 CANN 现有算子库比对，KIMI3 特有算子存在两类缺口：① 华为完全没实现的算子（需从零写 AscendC kernel）；② 华为有但形状/接口不匹配、实为图接线问题的算子。'))
story.append(P('1.2 交付物清单', H2))
dl = [
    ['自研 AscendC 算子', 'SiTU-GLU（SituAndMul）、AttnRes 残差栈'],
    ['参考实现（golden）', 'reference/situ.py、attn_res.py、kda.py、kda_chunked.py'],
    ['torch 包装', 'situ_and_mul_wrapper.cpp、attn_res_wrapper.cpp'],
    ['验证 harness', 'test_situ_real.py、test_attn_res_real.py、compare_real_chunk_kda.py'],
    ['文档', '本报告 + README.md + docs/（环境、开发流程、KDA 对比）'],
]
t = Table(dl, colWidths=[38*mm, 132*mm])
t.setStyle(TableStyle([('FONT', (0,0), (-1,-1), F, 9.5), ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor('#bbbbbb')),
                       ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f0f0f0')), ('VALIGN', (0,0), (-1,-1), 'TOP'),
                       ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3)]))
story.append(t)

# 2 缺口分析
story.append(P('2. 缺口分析', H1))
gap = [
    ['算子', '华为覆盖', '结论'],
    ['MLA attention', 'ATB MultiLatentAttention', '✅ 已有'],
    ['MoE dispatch/combine', 'npu_ops_transformer', '✅ 已有'],
    ['sigmoid 路由 + topk', 'ATB FusedAddTopkDiv', '✅ 已有'],
    ['KDA（Gated Delta Rule）', 'aclnnChunkKdaFwd', '✅ 已有（补 wrapper + 验证）'],
    ['SiTU-GLU', '❌ 只有 SiLU', '自研 AscendC kernel'],
    ['AttnRes 残差栈', '❌ 无', '自研 AscendC kernel'],
    ['LatentMoE 投影', '❌ 形状不符', '图级组合（无需新 kernel）'],
]
t = Table(gap, colWidths=[40*mm, 62*mm, 68*mm])
t.setStyle(TableStyle([('FONT', (0,0), (-1,-1), F, 9.5), ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor('#bbbbbb')),
                       ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8e8e8')), ('VALIGN', (0,0), (-1,-1), 'TOP'),
                       ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3)]))
story.append(t)
story.append(P('<b>结论</b>：华为真正缺的算子只有 2 个（SiTU-GLU、AttnRes），本仓库均已实现并验证；LatentMoE 是模型图接线问题，不属算子开发。'))

# 3 算子设计
story.append(P('3. 算子设计', H1))

story.append(P('3.1 SiTU-GLU（SituAndMul）', H2))
story.append(P('<b>数学定义</b>（取自 modeling_kimi_linear.py 的 SituAndMul）：输入 x ∈ [..., 2d]（前 d 为 gate、后 d 为 up），输出 y ∈ [..., d]。'))
story.append(P('gate = x[..., :d]；up = x[..., d:]<br/>situ(gate) = beta·tanh(gate/beta)·sigmoid(gate)（beta=4.0）<br/>up\' = linear_beta·tanh(up/linear_beta)（linear_beta=25.0）<br/>y = situ(gate) · up\'', CODE))
story.append(P('与 SiLU（x·sigmoid(x)）不同：SiTU 是 tanh·sigmoid 乘积，且 up 分支带 tanh 压缩，是 KIMI3 特有激活变体，华为只有 SiLU 融合。'))
story.append(P('<b>接口</b>：aclnnSituAndMul(x, beta, linearBeta, out)，x/out 为 bf16，out 形状 [..., d]。'))
story.append(P('<b>Kernel 设计</b>：沿最后一维劈两半（gate 前 d、up 后 d）；d 维按 TILE_ELEM=4096 分块；内部 fp32 计算；输入 DataCopy(MTE2)→向量(V) 与 向量(V)→输出 DataCopy(MTE3) 用 TQue 的 EnQue/DeQue 同步；按 token 行分 8 core。'))

story.append(P('3.2 AttnRes 残差栈', H2))
story.append(P('<b>数学定义</b>（取自 _apply_attn_res）：本质是「mini-attention」，对 B+1 个残差向量做 RMS-norm + 学习 score 点积 + softmax + 加权求和。'))
story.append(P('v = cat([block_residual, prefix_sum[:,None]], dim=1)   (T, B+1, H)<br/>k = v · rsqrt(mean(v², -1) + eps)<br/>scores = sum(k · score_weight, -1)<br/>probs = softmax(scores, -1)<br/>out = sum(probs[:,:,None] · v, 1)', CODE))
story.append(P('其中 score_weight = norm.weight · proj.weight.squeeze(0) ∈ (H,) 为学到的常量。'))
story.append(P('<b>接口</b>：aclnnAttnRes(blockResidual, prefixSum, scoreWeight, eps, out)，blockResidual (T,B,H)、prefixSum (T,H)、scoreWeight (H,)、out (T,H)，均 bf16。'))
story.append(P('<b>Kernel 设计</b>：两趟——Pass 1 逐残差向量沿 H 归约 sum(v²) 与 sum(v·score_weight)，得 score 后向量 Exp 在 32 对齐 buffer 上做 softmax；Pass 2 加权求和 out = Σ_b probs[b]·v[b]。归约用 ReduceSum(dst, src, workBuf, H) + GetValue(0)。按 token 分 8 core。'))

# 4 验证方法论
story.append(P('4. 验证方法论', H1))
story.append(P('<b>无权重验证</b>：只靠 shape + 随机数据 + 参考实现验证算子正确性，无需下载模型权重。'))
story += bullets([
    '从 KIMI3 config 拿 shape/dtype；',
    '固定 seed 生成随机输入；',
    '参考实现（纯 torch，fp32 内部 → bf16 输出）跑出 golden；',
    '待验算子（AscendC kernel，NPU 上）跑候选输出；',
    '对比：shape/dtype 一致 + 数值误差在容差内（bf16 用 max_rel < 1e-2）。',
])
story.append(P('<b>为什么 golden 可信</b>：golden 是纯 torch 的直白实现，与 NPU kernel 完全解耦、不共享任何实现，作为正确性金标准。'))

# 5 测试结果
story.append(P('5. 测试结果', H1))
story.append(P('5.1 SiTU-GLU', H2))
r1 = [
    ['shape', 'd', 'max_abs', 'max_rel', '结果'],
    ['(2, 3, 128)', '64', '0.0', '0.0', '✅'],
    ['(4, 5, 256)', '128', '0.0', '0.0', '✅'],
    ['(2, 7, 4096)', '2048', '9.5e-7', '8.3e-4', '✅'],
    ['(2, 7, 22016)', '11008', '1.56e-2', '7.5e-3', '✅'],
    ['(3, 5, 256)', '128', '0.0', '0.0', '✅'],
]
t = Table(r1, colWidths=[40*mm, 25*mm, 33*mm, 33*mm, 20*mm])
t.setStyle(TableStyle([('FONT', (0,0), (-1,-1), F, 9.5), ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor('#bbbbbb')),
                       ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8e8e8')), ('ALIGN', (1,1), (4,-1), 'CENTER'),
                       ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3)]))
story.append(t)
story.append(P('（d=11008 为 KIMI3 真实 intermediate_size；末行用不同 beta/linear_beta 验证参数化。）'))

story.append(P('5.2 AttnRes', H2))
r2 = [
    ['T', 'B', 'H', 'max_abs', 'max_rel', '结果'],
    ['4', '2', '512', '9.5e-7', '8.8e-4', '✅'],
    ['8', '4', '4096', '9.8e-4', '4.7e-3', '✅'],
    ['16', '8', '4096', '1.56e-2', '7.7e-3', '✅'],
    ['32', '4', '1024', '3.9e-3', '7.4e-3', '✅'],
    ['2', '1', '512', '0.0', '0.0', '✅'],
]
t = Table(r2, colWidths=[24*mm, 24*mm, 30*mm, 33*mm, 33*mm, 20*mm])
t.setStyle(TableStyle([('FONT', (0,0), (-1,-1), F, 9.5), ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor('#bbbbbb')),
                       ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8e8e8')), ('ALIGN', (0,1), (5,-1), 'CENTER'),
                       ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3)]))
story.append(t)
story.append(P('（H=4096 为 KIMI3 真实 hidden_size。）'))
story.append(P('5.3 KDA（华为算子，仅验证）：真实华为 chunk_kda_fwd vs 参考实现，attn_out / final_state 相对误差 ~4e-3（详见 docs/03）。'))

# 6 关键实现要点
story.append(P('6. 关键实现要点与坑', H1))
story += bullets([
    'bf16 类型是 bfloat16_t（DT_BF16 是宏=27，不是类型）；',
    'REGISTER_TILING_DEFAULT 须在 GET_TILING_DATA 之前（它声明 tilingData）；',
    '跨引擎同步必须 EnQue/DeQue（漏了 → 输出全零，最易踩的坑）；',
    '标量 exp 在 aicore 上不存在，softmax 用向量 AscendC::Exp；',
    'uint→float cast 不允许，倒数类量移到 host 端算好传进 tiling；',
    '归约用 ReduceSum(dst, src, workBuf, count) + GetValue(0)。',
])
story.append(P('完整流程见 docs/02-AscendC算子开发流程与坑.md。'))

# 7 结论
story.append(P('7. 结论与后续工作', H1))
story.append(P('<b>结论</b>：华为缺的 2 个真算子（SiTU-GLU、AttnRes）均已实现 AscendC kernel 并在 A3 上验证通过，误差在 bf16 精度内。LatentMoE 不需要新 kernel——它是图级组合：down_proj(Linear H→moe_hidden) → MoE → [RMSNorm] → up_proj(Linear moe_hidden→H)，全用现有算子拼，属 vllm/sglang 适配层的图接线。'))
story.append(P('<b>后续工作</b>（建议由昇腾团队或部署侧完成）：① LatentMoE 图接线（vllm-ascend/sglang 插入 down/up 投影）；② 性能 benchmark（本报告只验证正确性，未测吞吐/显存收益）；③ 算子合入 CANN（按 docs/02 工程结构对齐规范）。'))

# 8 附录
story.append(P('8. 附录', H1))
story.append(P('8.1 环境信息', H2))
env = [
    ['设备', 'Ascend 910_9382 × 2（SOC ascend910_93）'],
    ['CANN', '9.0.0（/usr/local/Ascend/cann-9.0.0）'],
    ['torch / torch_npu', '2.10.0+cpu / 2.10.0'],
    ['AscendC 编译器', 'ccec / bisheng'],
    ['算子脚手架', 'msopgen'],
]
t = Table(env, colWidths=[38*mm, 132*mm])
t.setStyle(TableStyle([('FONT', (0,0), (-1,-1), F, 9.5), ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor('#bbbbbb')),
                       ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#f0f0f0')), ('VALIGN', (0,0), (-1,-1), 'TOP'),
                       ('TOPPADDING', (0,0), (-1,-1), 3), ('BOTTOMPADDING', (0,0), (-1,-1), 3)]))
story.append(t)
story.append(P('8.2 复现步骤', H2))
story.append(P('# 1. 编译算子（以 SiTU-GLU 为例）\ncd kernels/SituAndMul && bash build.sh\n# 产出 build_out/custom_opp_ubuntu_aarch64.run\n\n# 2. 安装到 vendor\ncd build_out && ./custom_opp_ubuntu_aarch64.run --install-path=/usr/local/Ascend/cann-9.0.0/opp --quiet\n\n# 3. 跑对拍（自动 JIT 编译 wrapper + golden 对比）\ncd /workspace/kimi3_ops && python test_situ_real.py\n# AttnRes 同理：python test_attn_res_real.py', CODE))

doc.build(story)
print('PDF 生成完成: /workspace/kimi3_ops/设计验证测试报告.pdf')
