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
using Params =
    sm100::prefill::fused_norm_rope_attn_rope_cast_fwd::core_attn::ParamT<
        SparseAttnFwdMode::Decode>;

// 固定配置 64 个 Q head，共享一组 KV；每次处理 64 个稀疏位置
constexpr uint32_t H_Q = 64, H_Q_PER_CTA = 64, D_QK = 512, D_VO = 512;
constexpr uint32_t D_ROPE = 64, WV_GROUP_SIZE = 8, O_QUANT_TILE_SIZE = 32;
constexpr uint32_t B_TOPK = 64, NUM_THREADS = 512;
constexpr uint32_t NUM_WORKING_THREADS = 128 + 128 + 1 + 1 + 32 + 128;
constexpr uint32_t FOLD_FACTOR = 2, NUM_MRGEMM_RAILS = 2;
constexpr uint32_t NUM_P_ELEMS_PER_THREAD = 32;
constexpr uint32_t NUM_KV_SLOTS = 3, NUM_INDICES_BUFS = 4, NUM_P_BUFS = 2;

template <ModelType MT> struct KVFormat {
    using Base = KVCacheFormat<MT>;
    static constexpr bool IS_FP4 = Base::IS_FP4;
    static constexpr uint32_t QUANT_TILE_SIZE = Base::QUANT_TILE_SIZE;
    static constexpr uint32_t NUM_SCALES_EACH_TOKEN =
        Base::NUM_SCALES_EACH_TOKEN;
    static constexpr uint32_t TMA_K_STRIDE = Base::TMA_K_STRIDE;

    static constexpr uint32_t RAW_TOKEN_DATA_BYTES = Base::QUANT_BYTES;
    static constexpr uint32_t RAW_TOKEN_SMEM_STRIDE = RAW_TOKEN_DATA_BYTES + 16;
    static constexpr uint32_t NUM_CHUNKS_PER_ROW = RAW_TOKEN_DATA_BYTES / 16;
    static constexpr uint32_t CHUNK_ELEMS = Base::IS_FP4 ? 32 : 16;
};

using QrigKVFormat = KVFormat<ModelType::V41>;
using ExtraKVFormat = KVFormat<ModelType::V41_FP4>;
constexpr uint32_t NUM_SCALES_EACH_TOKEN_PER_CTA = 16;
// fp8 最大搬运 512B，加上 16 为 528B，用 gather4 搬运一次 2112B
// 向上对齐到 128 为 2176B
constexpr uint32_t RAW_KV_GROUP_BYTES = 2176;

struct SharedMemoryPlan {
    alignas(1024) bf16 kv_slots[NUM_KV_SLOTS][B_TOPK * D_QK];
    alignas(1024) bf16 s[H_Q_PER_CTA * B_TOPK];
    alignas(1024) float p_exchange_buf[4][32 * NUM_P_ELEMS_PER_THREAD];
    alignas(1024) uint8_t is_k_valid[NUM_INDICES_BUFS][B_TOPK / 8];

    alignas(16) int decode_tma_coords[NUM_INDICES_BUFS][B_TOPK];
    uint8_t decode_scales[NUM_INDICES_BUFS]
                         [B_TOPK * NUM_SCALES_EACH_TOKEN_PER_CTA];

    const uint8_t *decode_scale_ptrs[NUM_INDICES_BUFS][B_TOPK];

    float rowwise_max_buf[128];
    float rowwise_mi_buf[H_Q_PER_CTA];
    float rowwise_li_buf[128];

    // full 通知消费者可读，empty 通知生产者可复用；
    transac_bar_t bat_kv_slot_full[NUM_KV_SLOTS];
    transac_bar_t bar_kv_slot_empty[NUM_KV_SLOTS];

    transac_bar_t bar_indices_full[NUM_INDICES_BUFS];
    transac_bar_t bar_indices_empty[NUM_INDICES_BUFS];

    transac_bar_t bar_tQ_full, bar_tQ_empty;
    transac_bar_t bar_tO_full, bar_tO_empty;
    transac_bar_t bar_tP_full[NUM_P_BUFS];
    transac_bar_t bar_SO_full, bar_SO_empty;
    transac_bar_t bar_clc_full, bar_clc_empty;
    transac_bar_t bar_li_mi_full, bar_li_mi_empty;
    transac_bar_t bar_raw_kv_full;

    ku::CLCResponseObj clc_response_obj;
    array_aligned<uint32_t, 1> tmem_start_addr;
};

struct tmem_cols {
    static constexpr uint32_t O = 0, Q = 256, P_0 = 384, P_1 = 448;
    static __device__ constexpr uint32_t get_p(uint32_t i) {
        return i ? P_1 : P_0;
    }
};
using TiledMMA_QK = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_TS_NOELECT<
        bf16,
        bf16,
        float,
        64,
        128,
        UMMA::Major::K,
        UMMA::Major::K>{}
));
using TiledMMA_SV = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_WS_SS_NOELECT<
        bf16,
        bf16,
        float,
        64,
        256,
        UMMA::Major::K,
        UMMA::Major::MN>{}
));
using AllocatorT = cute::TMEM::Allocator1Sm;

static_assert(sizeof(SharedMemoryPlan) <= 227 * 1024);

struct TMAParams {
    CUtensorMap fp8, fp4;
};
struct AuxParams {
    cutlass::FastDivmod fast_divmod_page_block_size;
    cutlass::FastDivmod fast_divmod_extra_page_block_size;
};
struct barrier_ids {
    static constexpr int WG0_SYNC = 0;
    static constexpr int WG3_SYNC = 1;
    static constexpr int WG3_WARP02_SYNC = 2;
    static constexpr int WG3_WARP13_SYNC = 3;
};
} // namespace mega_attn
