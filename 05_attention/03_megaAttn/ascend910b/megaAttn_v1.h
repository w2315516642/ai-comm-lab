#pragma once
#include <cstdint>

namespace mega_ascend_v1 {
constexpr int HEADS = 64, DIM = 512, ROPE_DIM = 64;
constexpr int HEAD_TILE = 16, KV_TILE = 64;
constexpr float SCORE_SCALE = 0.04419417382415922f; // 1 / sqrt(512)
constexpr int Q_ELEMS = HEAD_TILE * DIM;
constexpr int KV_ELEMS = KV_TILE * DIM;
constexpr int SCORE_ELEMS = HEAD_TILE * KV_TILE;
constexpr int RAW_ROW_BYTES = 544; // 512 B 数据 + 最多 32 B scale

// 一个任务计算一个 query 的 16 个 head；有效长度内的物理槽号由 indexer 保证合法。
struct TaskConfig {
    int32_t query = 0, head_begin = 0, position = 0;
    int32_t topk = 0, extra_topk = 0, length = 0, extra_length = 0;
    int32_t page_size = 0, page_stride = 0;
    int32_t extra_page_size = 0, extra_page_stride = 0;
    int32_t indices_stride = 0, extra_indices_stride = 0;
};
static_assert(sizeof(TaskConfig) == 13 * sizeof(int32_t));

// 输入均为设备地址；分页 KV 保持每页先 data、后 scale 的原有布局。
struct Input {
    const uint16_t *q; // BF16 [N,32,64,16]
    const uint8_t *kv_fp8, *kv_fp4;
    const int32_t *indices, *extra_indices;
    const float *cos_sin; // [position,64]，前 32 个 cos，后 32 个 sin
    const float *sink;    // [64]，nullptr 表示不使用 sink
};

// 最终输出沿用 SM100 布局；LSE 不含 sink，空行约定为 +inf。
struct Output {
    uint8_t *fp8;     // [N,8,16,8,32]，E4M3
    uint32_t *scales; // [256,sf_stride]，每个 word 打包四个 UE8M0
    float *lse;      // [N,64]
    int32_t sf_stride; // >= N，且为 4 的倍数；padding 不写
};

// 每核 GM 暂存：Q/KV/P 为 BF16，QK/PV 为 FP32，各段均按 32 B 对齐。
// 这些数据只连接 Vector 与 Matmul，不是算子的输出。
struct ScratchLayout {
    static constexpr uint64_t Q = 0;
    static constexpr uint64_t KV = Q + Q_ELEMS * 2;
    static constexpr uint64_t Scores = KV + KV_ELEMS * 2;
    static constexpr uint64_t P = Scores + SCORE_ELEMS * 4;
    static constexpr uint64_t PV = P + SCORE_ELEMS * 2;
    static constexpr uint64_t BytesPerCore = PV + Q_ELEMS * 4; // 118 KiB
};
constexpr uint64_t ScratchBytes(uint32_t block_num) {
    return uint64_t(block_num) * ScratchLayout::BytesPerCore;
}
} // namespace mega_ascend_v1

// 一次 launch 完成整个 batch。tasks 为设备端 TaskConfig 数组；各任务的输出区域不能重叠。
// 1 <= block_num <= min(task_count, 可用 AIC 核数)；task_count=0 时直接返回。
// scratch 至少 ScratchBytes(block_num) 字节；system_workspace 另按平台要求分配。
// 两个 tiling 均为设备地址：QK=[16,512]x[512,64]，PV=[16,64]x[64,512]。
// Matmul 的 UB 预算各限 16 KiB；所有设备缓冲必须保持有效，直到 stream 执行完成。
extern "C" void launch_mega_attn_v1(
    const mega_ascend_v1::Input &in, const mega_ascend_v1::Output &out,
    const mega_ascend_v1::TaskConfig *tasks, uint32_t task_count, uint32_t block_num,
    void *scratch, void *qk_tiling, void *pv_tiling, void *system_workspace, void *stream);
