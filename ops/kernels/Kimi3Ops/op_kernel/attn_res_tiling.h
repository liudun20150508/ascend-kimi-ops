/* -------------------------------------------------------------------------
 * This file is part of the MindStudio project.
 * Copyright (c) 2025 Huawei Technologies Co.,Ltd.
 *
 * MindStudio is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * ------------------------------------------------------------------------- */


#ifndef ATTN_RES_TILING_H
#define ATTN_RES_TILING_H
#include <cstdint>

struct AttnResTilingData {
    uint32_t T;          // num_tokens
    uint32_t B;          // num_blocks
    uint32_t H;          // hidden_size
    uint32_t blockDim;   // 参与计算的核心数
    float eps;           // rms_norm_eps
    float invH;          // 1.0f / H（避免 kernel 内 uint->float cast）
};

#endif // ATTN_RES_TILING_H
