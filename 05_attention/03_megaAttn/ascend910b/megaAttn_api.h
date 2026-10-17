#pragma once
#include "megaAttn_config.h"

namespace mega_ascend {
// 设备输入：Q 沿用 [N,32,64,16] 布局；两路 KV 均保持原来的量化分页布局。
struct Stage1Input {
    const uint16_t *q;
    const uint8_t *kv_fp8, *kv_fp4;
    const int32_t *indices, *extra_indices;
    const float *cos_sin; // [position,64]，前 32 个 cos，后 32 个 sin
};

// 阶段 1 的观测点，同时作为 Matmul 的 GM 暂存；不是最终算子的输出接口。
struct Stage1Output {
    uint16_t *q_loaded; // [16,512]，还没有做 RoPE
    uint16_t *q_rope;   // [16,512]，做完 RoPE，作为 Matmul A
    uint16_t *kv;       // [64,512]，无效行填 0，作为 Matmul B（转置）
    int32_t *valid;     // [64]，0/1
    float *scores;     // [16,64]，最终包含 1/sqrt(512) 缩放与 -inf mask
    int32_t *status;    // 分配至少 8 个 int32，即一个 32 B 搬运块
};
} // namespace mega_ascend

// config、cube_tiling、workspace 都是设备地址；stream 是 aclrtStream。
extern "C" void launch_mega_attn_stage1(
    const mega_ascend::Stage1Input &in,
    const mega_ascend::Stage1Output &out,
    void *config, void *cube_tiling, void *workspace, void *stream);
