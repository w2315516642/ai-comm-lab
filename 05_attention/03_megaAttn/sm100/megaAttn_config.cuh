// 改编自 FlashMLA；Copyright (c) 2025 DeepSeek，MIT 许可。
#pragma once
#include "kernels/kv_cache_format.h"
#include "kernels/sm100/prefill/sparse/fused_norm_rope_attn_rope_cast_fwd/core_attn/kernel.h"
#include <cute/tensor.hpp>
#include <cutlass/fast_math.h>
#include <kerutils/kerutils.cuh>

#include "kernels/defines.h"

namespace mega_attn {
using namespace cute;
using fp8_e4m3 = cutlass::float_e4m3_t;
// 沿用上游的核函数参数结构，传递设备指针、尺寸和步长。
using Params =
    sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::ParamT<
        SparseAttnFwdMode::Decode>;

// 固定配置：64 个 Q head，共享一组 KV；每次处理 64 个稀疏位置。
constexpr uint32_t H_Q = 64, H_Q_PER_CTA = 64, D_QK = 512, D_VO = 512;
// D_ROPE 是 Q/O 最后 64 个维度；WV_GROUP_SIZE 描述输出每组 8 个 head，量化每 32 个值一组。
constexpr uint32_t D_ROPE = 64, WV_GROUP_SIZE = 8, O_QUANT_TILE_SIZE = 32;
// B_TOPK 是一次流水线处理的索引数量，不是整行 topk；NUM_THREADS 是一个 CTA 的线程数。
constexpr uint32_t B_TOPK = 64, NUM_THREADS = 512;
// CLC 确认计数：三个完整 WG，加 MMA 1 线程、调度 1 线程、索引 32 线程，共 418。
constexpr uint32_t NUM_WORKING_THREADS = 128 + 128 + 1 + 1 + 32 + 128;
// 64 个 head 映射到 128 个线程，故 FOLD_FACTOR=2；QK 的 512 维则拆成两条 256 维部分点积。
constexpr uint32_t FOLD_FACTOR = 2, NUM_MRGEMM_RAILS = 2;
// 完整分数个数=64 head×64 token=4096；分给128线程，得到4096/128=32个/线程。
constexpr uint32_t NUM_P_ELEMS_PER_THREAD = 32;
// KV 三槽承接搬运/QK/SV 的不同进度；索引四槽允许提前准备，P 两槽容纳相邻块分数。
// 3/4/2属于当前流水线的缓冲深度选择，不是由head数直接算出的唯一值；改动需同时验证同步时序。
constexpr uint32_t NUM_KV_SLOTS = 3, NUM_INDICES_BUFS = 4, NUM_P_BUFS = 2;

// 描述一种 KV 量化格式的行宽、scale 数量和搬运布局。
template <ModelType MT> struct KVFormat {

    using Base = KVCacheFormat<MT>;
    static constexpr bool IS_FP4 = Base::IS_FP4;
    static constexpr uint32_t QUANT_TILE_SIZE = Base::QUANT_TILE_SIZE;
    // FP8量化粒度32，故512/32=16个scale；FP4粒度16，故512/16=32个scale，均每个1 B。
    static constexpr uint32_t NUM_SCALES_EACH_TOKEN =
        Base::NUM_SCALES_EACH_TOKEN;
    // 压缩数据行距：FP8 为 512 字节、FP4 为 256 字节；scale 存在每页的数据区之后。
    static constexpr uint32_t TMA_K_STRIDE = Base::TMA_K_STRIDE;

    static constexpr uint32_t RAW_TOKEN_DATA_BYTES = Base::QUANT_BYTES;
    // 搬入 shared memory 后每行多留 16 字节，打散相邻行在 shared memory bank 上的映射。
    static constexpr uint32_t RAW_TOKEN_SMEM_STRIDE = RAW_TOKEN_DATA_BYTES + 16;
    // 16 B chunk内可放128/8=16个FP8或128/4=32个FP4；每行分别有512/16=32或256/16=16个chunk。
    static constexpr uint32_t NUM_CHUNKS_PER_ROW = RAW_TOKEN_DATA_BYTES / 16;
    static constexpr uint32_t CHUNK_ELEMS = Base::IS_FP4 ? 32 : 16;
};
using OrigKVFormat = KVFormat<ModelType::V41>;
using ExtraKVFormat = KVFormat<ModelType::V41_FP4>;
constexpr uint32_t NUM_SCALES_EACH_TOKEN_PER_CTA = 16;
// gather4 目标按 128 字节对齐；每个原始数据行额外留 16 字节。
// 按较大的 FP8 行距计算：ceil(4*528/128)*128=2176，FP4 也复用同样的四行组间距。
// 四行FP8有效搬运4×528=2112 B，补64 B使下一组从2176 B起对齐；FP4四行仅搬4×272=1088 B。
// 一个64行块占16个组，原始区预留16×2176=34816 B，小于解码后槽容量64×512×2=65536 B。
constexpr uint32_t RAW_KV_GROUP_BYTES = 2176;

// 定义一个 CTA 内共享的缓冲区、同步屏障和调度结果。
struct SharedMemoryPlan {

    // 三份 KV 双用途缓冲：先放压缩数据，再原地展开成 BF16。
    // 每KV槽=64×512×2=65536 B=64 KiB，三槽共192 KiB；这里alignas是起点对齐，不是额外的数据元素。
    alignas(1024) bf16 kv_slots[NUM_KV_SLOTS][B_TOPK * D_QK];

    // s 保存本块指数权重；p_exchange_buf 用于两个对应 warp 交换 QK 的部分点积。
    // is_k_valid 每个 token 只占一位，供 softmax 屏蔽无效位置。
    // S=64×64×2=8192 B=8 KiB；交换区=4warp×32lane×32float×4 B=16384 B=16 KiB。
    alignas(1024) bf16 s[H_Q_PER_CTA * B_TOPK];
    alignas(1024) float p_exchange_buf[4][32 * NUM_P_ELEMS_PER_THREAD];
    // 四份位图=4×(64/8)=32 B；注意后续字段的alignas还可能引入额外填充。
    alignas(1024) uint8_t is_k_valid[NUM_INDICES_BUFS][B_TOPK / 8];

    // 索引线程统一生成 TMA 坐标；16 字节对齐便于一次读取 gather4 所需的四个 int。
    // 坐标数组=4槽×64行×4 B=1024 B；FP8 scale数组=4×64×16 B=4096 B。
    alignas(16) int decode_tma_coords[NUM_INDICES_BUFS][B_TOPK];
    uint8_t decode_scales[NUM_INDICES_BUFS]
                         [B_TOPK * NUM_SCALES_EACH_TOKEN_PER_CTA];

    // FP8 scale 直接共享；FP4 scale 较多，仅共享全局地址，由反量化线程按需加载。
    // 64-bit指针数组=4×64×8=2048 B；若直接存FP4 scale则要4×64×32=8192 B。
    const uint8_t *decode_scale_ptrs[NUM_INDICES_BUFS][B_TOPK];

    // max_buf 用于半行最大值交换；mi_buf 保存每 head 的基准，li_buf 保留两半各自的分母。
    // max_buf和li_buf各128×4=512 B；mi_buf只存64个head，共64×4=256 B。
    float rowwise_max_buf[128];
    float rowwise_mi_buf[H_Q_PER_CTA];
    float rowwise_li_buf[128];

    // full 通知消费者可读，empty 通知生产者可复用；phase 随环形槽翻转。
    // wait(phase) 等本轮完成；生产者复用槽前等 empty(phase^1)，区别新旧两轮状态。
    transac_bar_t bar_kv_slot_full[NUM_KV_SLOTS],
        bar_kv_slot_empty[NUM_KV_SLOTS];
    transac_bar_t bar_indices_full[NUM_INDICES_BUFS],
        bar_indices_empty[NUM_INDICES_BUFS];
    // tQ/tO/tP 的屏障保护 TMEM；SO 保护 shared memory 的 S 和需要原地缩放的 TMEM O。
    transac_bar_t bar_tQ_empty, bar_tQ_full;

    transac_bar_t bar_tO_empty, bar_tO_full;
    transac_bar_t bar_tP_full[NUM_P_BUFS];
    transac_bar_t bar_SO_full, bar_SO_empty;
    // CLC 屏障保护任务响应；li_mi 保护 softmax 给输出线程的统计量；raw_kv 等待 TMA 搬完。
    transac_bar_t bar_clc_full, bar_clc_empty;
    transac_bar_t bar_li_mi_full, bar_li_mi_empty;
    transac_bar_t bar_raw_kv_full;

    ku::CLCResponseObj clc_response_obj;
    array_aligned<uint32_t, 1> tmem_start_addr;
};

// 规定 O、Q 和两份 QK 分数在 512 列 TMEM 中的起始位置。
struct tmem_cols {
    // O 占 256 列；Q 的 BF16 两两打包后占 128 列；两个 P 各占 64 列，正好用满。
    // 按128行、每行每列32 bit计，一列提供128×4=512 B；这是TMEM列，不是逻辑矩阵列。
    // O：64×512×4/512=256列；Q：64×512×2/512=128列。
    // 每份P含两路部分点积：64×64×2×4/512=64列。
    // 因此起点依次是0、256、256+128=384、384+64=448；末端448+64=512。
    static constexpr uint32_t O = 0, Q = 256, P_0 = 384, P_1 = 448;
    static __device__ constexpr uint32_t get_p(uint32_t i) {
        return i ? P_1 : P_0;
    }
};
// QK：TMEM×shared 的 BF16 输入、FP32 累加；128 列输出包含两路 64-token 的部分点积。
using TiledMMA_QK = decltype(make_tiled_mma(SM100_MMA_F16BF16_WS_TS_NOELECT<
                                            bf16,
                                            bf16,
                                            float,
                                            64,
                                            128,
                                            UMMA::Major::K,
                                            UMMA::Major::K>{}));
// SV：shared×shared，每个 atom 覆盖 256 个输出维度；完整的 512 维由两个 atom 组成。
using TiledMMA_SV = decltype(make_tiled_mma(SM100_MMA_F16BF16_WS_SS_NOELECT<
                                            bf16,
                                            bf16,
                                            float,
                                            64,
                                            256,
                                            UMMA::Major::K,
                                            UMMA::Major::MN>{}));
using AllocatorT = cute::TMEM::Allocator1Sm;

// 仅KV/S/交换区已用192+8+16=216 KiB；还需元数据、屏障和对齐填充，最终以sizeof为准。
// 227×1024=232448 B是这里采用的总大小校验上限，不能把所有数组裸字节数相加当作sizeof。
static_assert(sizeof(SharedMemoryPlan) <= 227 * 1024);

// 保存 FP8 和 FP4 缓存的 TMA 描述符，供 gather4 搬运使用。
struct TMAParams {
    CUtensorMap fp8, fp4;
};
// 预计算页大小的快速除法参数，用于定位 token 所在页及页内位置。
struct AuxParams {
    cutlass::FastDivmod fast_divmod_page_block_size;
    cutlass::FastDivmod fast_divmod_extra_page_block_size;
};
// 分配线程组同步屏障的编号，避免不同同步操作互相干扰。
struct barrier_ids {
    static constexpr int WG0_SYNC = 0, WG3_SYNC = 1;
    static constexpr int WG3_WARP02_SYNC = 2, WG3_WARP13_SYNC = 3;
};
} // namespace mega_attn
