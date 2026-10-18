#pragma once
#include <cstdint>

namespace mega_ascend {
// 阶段 1~4 已在 910B2 / CANN 9.0.0 实卡验收：数值、布局、多核和预取对拍通过。
// B2C 先前验证了阶段 1/2；完整新版的 B2C 实卡结果尚未补测。
// run.sh 自动识别芯片，回归阶段 1~4，再输出完整 batch 延迟和 query/s。
constexpr int HEADS = 64, DIM = 512, ROPE_DIM = 64;
constexpr int HEAD_TILE = 16, KV_TILE = 64;
constexpr float SCORE_SCALE = 0.04419417382415922f; // 1 / sqrt(512)
constexpr int Q_ELEMS = HEAD_TILE * DIM;           // 8192 个 BF16 = 16 KiB
constexpr int KV_ELEMS = KV_TILE * DIM;            // 32768 个 BF16 = 64 KiB
constexpr int SCORE_ELEMS = HEAD_TILE * KV_TILE;   // 1024 个 FP32 = 4 KiB
constexpr int RAW_ROW_BYTES = 544; // 最大 512 B 数据 + 32 B scale，32 B 对齐

// 一次阶段测试的坐标和分页参数；所有字段为 int32，便于在设备侧逐字读取。
struct Stage1Config {
    int32_t query = 0, head_begin = 0, kv_begin = 0, position = 0;
    int32_t topk = 0, extra_topk = 0, length = 0, extra_length = 0;
    int32_t page_size = 0, pages = 0, page_stride = 0;
    int32_t extra_page_size = 0, extra_pages = 0, extra_page_stride = 0;
    int32_t indices_stride = 0, extra_indices_stride = 0;
};
static_assert(sizeof(Stage1Config) == 16 * sizeof(int32_t));

// 状态表示第一个尚未实现的步骤；0 才允许进入数值验收，-1 表示设备未回写。
enum class Step : int32_t {
    Done = 0, LoadQuery = 1, ApplyQueryRope = 2, LoadKvTile = 3,
    DecodeKvTile = 4, ComputeQk = 5, MaskScores = 6,
    InitOnlineState = 10, OnlineSoftmax = 11, ComputePv = 12,
    AccumulateOutput = 13, NormalizeOutput = 14,
    ApplySinkAndLse = 20, ApplyOutputRope = 21, QuantizeOutput = 22,
    ProcessTasks = 30
};
inline const char *StepName(int32_t step) {
    switch (static_cast<Step>(step)) {
        case Step::Done: return "Done";
        case Step::LoadQuery: return "LoadQuery";
        case Step::ApplyQueryRope: return "ApplyQueryRope";
        case Step::LoadKvTile: return "LoadKvTile";
        case Step::DecodeKvTile: return "DecodeKvTile";
        case Step::ComputeQk: return "ComputeQk";
        case Step::MaskScores: return "MaskScores";
        case Step::InitOnlineState: return "InitOnlineState";
        case Step::OnlineSoftmax: return "OnlineSoftmax";
        case Step::ComputePv: return "ComputePv";
        case Step::AccumulateOutput: return "AccumulateOutput";
        case Step::NormalizeOutput: return "NormalizeOutput";
        case Step::ApplySinkAndLse: return "ApplySinkAndLse";
        case Step::ApplyOutputRope: return "ApplyOutputRope";
        case Step::QuantizeOutput: return "QuantizeOutput";
        case Step::ProcessTasks: return "ProcessTasks/BindTask";
        default: return "invalid/missing device status";
    }
}
} // namespace mega_ascend
