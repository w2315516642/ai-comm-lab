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
// 阶段 2：P/PV 是复用的单块暂存，output 是所有 KV 的 attention 结果。
struct Stage2Output {
    uint16_t *p;       // [16,64]，未归一化的 exp(score-m)，BF16
    float *pv;         // [16,512]，当前块的 P*V，FP32
    float *output;     // [16,512]，最终归一化结果；不含 sink、逆 RoPE、量化
    float *row_max;    // [16]，最终 m；全空时 -inf
    float *row_sum;    // [16]，最终 l=sum(exp(score-m))；全空时 0
};
// 同一多核调度器下的三条路径：标量基线、同步向量化、向量化加 KV 预取。
enum class Stage4Mode : int32_t { Reference = 0, Vector = 1, Pipeline = 2 };

// 阶段 3 的完整输出，布局沿用 SM100；一个 kernel 只写 cfg 指定的 16 个 head。
struct Stage3Output {
    uint8_t *fp8;      // [N,8,16,8,32]，每个元素一个 E4M3 字节
    uint32_t *scales;  // [256,sf_stride]，每个 word 打包四个 UE8M0 指数
    float *lse;        // [N,64]，不含 sink；全空行约定为 +inf
    int32_t sf_stride; // >= N，且为 4 的倍数
};
} // namespace mega_ascend

// config、cube_tiling、workspace 都是设备地址；stream 是 aclrtStream。
extern "C" void launch_mega_attn_stage1(
    const mega_ascend::Stage1Input &in,
    const mega_ascend::Stage1Output &out,
    void *config, void *cube_tiling, void *workspace, void *stream);

// cfg.kv_begin 在阶段 2 忽略，从第 0 块遍历全部 KV；scratch 复用阶段 1 的缓冲。
extern "C" void launch_mega_attn_stage2(
    const mega_ascend::Stage1Input &in,
    const mega_ascend::Stage1Output &scratch,
    const mega_ascend::Stage2Output &out,
    void *config, void *qk_tiling, void *pv_tiling, void *workspace, void *stream);

// sink 为 [64] FP32 或 nullptr；online.output 在此导出当前步骤的 FP32 输出观测点。
extern "C" void launch_mega_attn_stage3(
    const mega_ascend::Stage1Input &in,
    const mega_ascend::Stage1Output &scratch,
    const mega_ascend::Stage2Output &online,
    const float *sink, const mega_ascend::Stage3Output &out,
    void *config, void *qk_tiling, void *pv_tiling, void *workspace, void *stream);

// configs 是 task_count 个 Stage1Config；每项指定一个 query 和 16 个 head。
// scratch/online 的各字段分配 block_num 份；唯独 status 分配 task_count*8 个 int32。
// out 仍是完整输出；1 <= block_num <= min(task_count, 可用 AIC 核数)。
// Vector/Pipeline 不写 q_loaded、valid、output、row_max、row_sum；scores 暂存原始 QK。
extern "C" void launch_mega_attn_stage4(
    const mega_ascend::Stage1Input &in,
    const mega_ascend::Stage1Output &scratch,
    const mega_ascend::Stage2Output &online,
    const float *sink, const mega_ascend::Stage3Output &out,
    uint32_t task_count, uint32_t block_num,
    void *configs, void *qk_tiling, void *pv_tiling, void *workspace, void *stream,
    mega_ascend::Stage4Mode mode = mega_ascend::Stage4Mode::Pipeline);
