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


#ifndef SITU_AND_MUL_TILING_H
#define SITU_AND_MUL_TILING_H
#include <cstdint>

struct SituAndMulTilingData {
    uint32_t d;          // 切分维：输出最后一维 = 输入最后一维 / 2
    uint32_t numRows;    // 行数 = 输入除最后一维外各维乘积
    uint32_t blockDim;   // 参与计算的核心数
    float beta;          // activation_situ_beta
    float linear_beta;   // activation_situ_linear_beta
};

#endif // SITU_AND_MUL_TILING_H
