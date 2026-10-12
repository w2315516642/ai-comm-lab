// 学习实现，改编自 FlashMLA；Copyright (c) 2025 DeepSeek，MIT 许可。
#include <cuda_fp8.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>
#include <math_constants.h>
#include <stdexcept>

#include "kernels/sm100/common_subroutine.h"
#include "kernels/sm100/helpers.h"
#include "kernels/utils.h"
#include "megaAttn_api.h"
#include "megaAttn_config.cuh"

namespace mega_attn_v1 {
using namespace mega_attn;
using namespace sm100;
constexpr float MAX_INIT_VAL = -1e30f, O_QUANT_CLAMP_MIN_VALUE = 1e-4f;

__device__ __forceinline__ float2
apply_rope(const float2 &x, float c, float s) {
    return ku::float2_fma({x.x, x.x}, {c, s}, {-x.y * s, x.y * c});
}

enum class KVLocation { ORIG, ORIG_AND_EXTRA, EXTRA };
// 记录一个 Q 的处理任务，包括索引长度，KV 块数和同步阶段
struct Job {
    bool is_valid;
    uint32_t s_q_idx, job_idx_mod_2, topk_length, num_kv_blocks;
    uint32_t extra_topk_length, num_orig_slots;
};

class Kernel {
    const Params &params;
    const TMAParams &tma_params;
    const AuxParams &aux_params;
    SharedMemoryPlan &smem;

    uint32_t lane_idx, warp_idx, idx_in_warpgroup;
    // 仅 WG0 使用：保留当前输出和下个 Q 的 RoPE 系数。
    float cached_cos[2]{}, cached_sin[2]{};
    // 仅 WG1 使用：每行由 4 个 lane 协作，每线程处理两行
    static constexpr uint32_t NUM_LANES_PER_ROW = 4, NUM_ROWS_PER_WAVEFRONT = 8;
    static constexpr uint32_t NUM_TOKENS_PER_THREAD = 2, NUM_ROWS_PER_WARP = 16;
    static constexpr uint32_t MAX_CHUNKS_PER_LANE = 8;
    static constexpr uint32_t STS_ATOM_COL_STRIDE_BYTES = B_TOPK * 128;
    static constexpr uint32_t STS_TOKEN_STRIDE_BYTES = 8 * 128;

    uint32_t local_warp_idx, row_in_wavefront, idx_in_row, chunk_rotation;
    uint32_t sts_offsets[8];

  public:
    __device__ Kernel(
        const Params &p,
        const TMAParams &t,
        const AuxParams &a,
        SharedMemoryPlan &s
    )
        : params(p), tma_params(t), aux_params(a), smem(s),
          warp_idx(cutlass::canonical_warp_idx_sync()),
          idx_in_warpgroup(threadIdx.x % 128), lane_idx(threadIdx.x % 32),
          local_warp_idx(warp_idx - 4), row_in_wavefront(lane_idx % 8),
          idx_in_row(lane_idx / 8), chunk_rotation(row_in_wavefront / 4 * 4) {}

    __device__ __forceinline__ void run() {
        initialize();
        uint32_t wg = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        if (wg == 0) {
            cutlass::arch::warpgroup_reg_alloc<184>();
            run_q_output();
        } else if (wg == 1) {
            cutlass::arch::warpgroup_reg_alloc<128>();
            run_kv();
        } else if (wg == 3) {
            cutlass::arch::warpgroup_reg_alloc<128>();
            run_softmax();
        } else {
            // WG2 要让出寄存器给 WG0，使用 dealloc 而不是 alloc。
            cutlass::arch::warpgroup_reg_dealloc<72>();
            if (warp_idx == 8 && elect_one_sync())
                run_mma();
            else if (warp_idx == 9 && elect_one_sync())
                run_scheduler();
            else if (warp_idx == 10)
                run_indices();
        }
    }

  private:
    __device__ __forceinline__ void initialize() {
        if (warp_idx == 1 && elect_one_sync()) {
#pragma unroll
            for (uint32_t i = 0; i < NUM_KV_SLOTS; ++i) {
                smem.bar_kv_slot_full[i].init(129);
                smem.bar_kv_slot_empty[i].init(1);
            }
#pragma unroll
            for (uint32_t i = 0; i < NUM_INDICES_BUFS; ++i) {
                smem.bar_indices_full[i].init(32);
                smem.bar_indices_empty[i].init(256);
            }
#pragma unroll
            for (uint32_t i = 0; i < NUM_P_BUFS; ++i) {
                smem.bar_tP_full[i].init(1);
            }

            smem.bar_clc_full.init(1);
            smem.bar_clc_empty.init(NUM_WORKING_THREADS);

            smem.bar_tQ_full.init(128);
            smem.bar_tQ_empty.init(1 + 128);

            smem.bar_tO_full.init(1);
            smem.bar_tO_empty.init(128);

            smem.bar_SO_full.init(128);
            smem.bar_SO_empty.init(1);

            smem.bar_li_mi_full.init(128);
            // WG0 的 128 个线程都读完统计量后，WG3 才能覆盖。
            smem.bar_li_mi_empty.init(128);
            smem.bar_raw_kv_full.init(1);

            fence_barrier_init();
        } else if (warp_idx == 3) {
            AllocatorT().allocate(512, smem.tmem_start_addr.data());
            AllocatorT().release_allocation_lock();
        }

        __syncthreads();
    }

    __device__ __forceinline__ Job make_job(uint32_t phase, uint32_t query) {
        uint32_t main_length = params.topk_length
                                   ? __ldg(params.topk_length + query)
                                   : params.topk;
        uint32_t extra_length = params.extra_topk_length
                                    ? __ldg(params.extra_topk_length + query)
                                    : params.extra_topk;
        return {
            true,
            query,
            phase,
            main_length,
            ku::ceil_div(
                uint32_t(params.topk) + extra_length, uint32_t(B_TOPK)
            ),
            extra_length,
            uint32_t(params.topk)
        };
    }

    // 等 CLC 返回下个任务
    __device__ __forceinline__ Job get_next_job(const Job &cur_args) {
        smem.bar_clc_full.wait(cur_args.job_idx_mod_2);
        ku::CLCResult next_cta0_idx =
            ku::get_clc_query_response<true>(smem.clc_response_obj);
        smem.bar_clc_empty.arrive();

        if (!next_cta0_idx.is_valid) {
            return Job{};
        } else {
            return make_job(cur_args.job_idx_mod_2 ^ 1, next_cta0_idx.x);
        }
    }

    __device__ __forceinline__ void load_q(const Job &cur_job) {
        static constexpr uint32_t NUM_CACHED_BF16_PER_THREAD =
            H_Q_PER_CTA * D_QK / 128;
        static constexpr uint32_t NUM_BF16_PER_LOAD = 256 / 16;
        bf16 cached_q[NUM_CACHED_BF16_PER_THREAD];

        uint32_t cur_q_position =
            __ldg(params.token_positions + cur_job.s_q_idx);
        cached_cos[1] =
            __ldg(params.cos_sin_cache + cur_q_position * D_ROPE + lane_idx);
        cached_sin[1] = __ldg(
            params.cos_sin_cache + cur_q_position * D_ROPE + D_ROPE / 2 +
            lane_idx
        );

        bf16 *q_token_base =
            params.q + (uint64_t)cur_job.s_q_idx * params.stride_q_s_q;

        // 默认 D_ROPE == TILE_SIZE
        static constexpr uint32_t TILE_SIZE = 64;
        static constexpr uint32_t TILES_PER_THREAD = D_QK / (2 * TILE_SIZE);
        // 4 个 tile 的编号是 3..0；用有符号数让减到 -1 时正常退出。
#pragma unroll
        for (int32_t local_tile_idx = int32_t(TILES_PER_THREAD) - 1;
             local_tile_idx >= 0;
             --local_tile_idx) {
            uint32_t tile_idx = local_tile_idx * 2 + warp_idx / 2;

#pragma unroll
            for (uint32_t i = 0; i < TILE_SIZE / NUM_BF16_PER_LOAD; ++i) {
                uint32_t h_q_idx = idx_in_warpgroup % H_Q_PER_CTA;
                uint32_t d_q_idx = tile_idx * TILE_SIZE + i * NUM_BF16_PER_LOAD;

                KU_LDG_256(
                    q_token_base + h_q_idx * NUM_BF16_PER_LOAD + d_q_idx * H_Q,
                    cached_q + local_tile_idx * TILE_SIZE +
                        i * NUM_BF16_PER_LOAD,
                    ".nc",
                    "no_allocate",
                    "evict_first",
                    "256B"
                );
            }

            // 最后一个 TILE 要做 RoPE
            if (tile_idx == (D_QK / TILE_SIZE - 1)) {
#pragma unroll
                for (uint32_t j = 0; j < TILE_SIZE; j += 2) {
                    float2 x = __bfloat1622float2(
                        *(nv_bfloat162 *)(cached_q +
                                          local_tile_idx * TILE_SIZE + j)
                    );
                    // 32 个 lane 里面已经取好了 RoPE 系数
                    // 这里去对应的 lane拿就好，不用再从全局内存拿
                    float cur_cos =
                        __shfl_sync(0xffffffff, cached_cos[1], j / 2);
                    float cur_sin =
                        __shfl_sync(0xffffffff, cached_sin[1], j / 2);
                    float2 y = apply_rope(x, cur_cos, cur_sin);
                    *(nv_bfloat162 *)(cached_q + local_tile_idx * TILE_SIZE + j
                    ) = nv_bfloat162{
                        __float2bfloat16_rn(y.x), __float2bfloat16_rn(y.y)
                    };
                }
            }
        }

        // 等待 TMEM 的旧 Q 被消费完毕再搬进去
        smem.bar_tQ_empty.wait(cur_job.job_idx_mod_2 ^ 1);
        ku::tcgen05_after_thread_sync();

        static constexpr uint32_t NUM_CACHED_UINT32 =
            NUM_CACHED_BF16_PER_THREAD / 2;
        ku::tmem_st_32dp32bNx<NUM_CACHED_UINT32 / 2>(tmem_cols::Q, cached_q);
        ku::tmem_st_32dp32bNx<NUM_CACHED_UINT32 / 2>(
            tmem_cols::Q + NUM_CACHED_UINT32 / 2,
            cached_q + NUM_CACHED_UINT32 / 2 * 2
        );

        cutlass::arch::fence_view_async_tmem_store();

        ku::tcgen05_before_thread_sync();
        smem.bar_tQ_full.arrive();
    }

    __device__ __forceinline__ uint8_t
    quantize_output(float *output, float output_scale, fp8_e4m3 *packed) {
        float output_abs_max =
            get_max<O_QUANT_TILE_SIZE, true>(output) * output_scale;
        output_abs_max = max(O_QUANT_CLAMP_MIN_VALUE, output_abs_max);

        float sf = output_abs_max / 448.0f;
        uint32_t sf_as_uint32 = *reinterpret_cast<uint32_t *>(&sf);

        uint32_t exp_sf = (int32_t)((sf_as_uint32 - 1) >> 23) + (1 - 127);
        uint32_t sf_inv_as_uint32 = (127 - exp_sf) << 23;
        float sf_inv = *reinterpret_cast<float *>(&sf_inv_as_uint32);

        float cur_multiplier = output_scale * sf_inv;
        float2 cur_multiplier2 = float2(cur_multiplier, cur_multiplier);
#pragma unroll
        for (uint32_t j = 0; j < O_QUANT_TILE_SIZE; j += 2) {
            float2 x = *(float2 *)(output + j);
            x = ku::float2_mul(x, cur_multiplier2);
            *(__nv_fp8x2_storage_t *)(packed + j) = __nv_cvt_float2_to_fp8x2(
                x, __NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3
            );
        }
        return exp_sf + 127;
    }

    __device__ __forceinline__ void write_output(
        const Job &cur_job,
        const fp8_e4m3 (&output_fp8)[2][128],
        const uint8_t (&output_sf)[2][4]
    ) {
        constexpr uint32_t NUM_MMA_ATOMS = 2, MMA_ATOM_N = 256,
                           NUM_O_TMEM_COLS_PER_ATOM = 128;
        uint32_t head_idx = idx_in_warpgroup % H_Q_PER_CTA;
        uint32_t wv_group_idx = head_idx / WV_GROUP_SIZE;
        uint32_t head_idx_in_wv_group = head_idx % WV_GROUP_SIZE;

#pragma unroll
        for (uint32_t mma_atom_idx = 0; mma_atom_idx < NUM_MMA_ATOMS;
             ++mma_atom_idx) {
            constexpr uint32_t OUTPUT_SAVE_AS_SCALE_GRAN = 32;
            uint32_t head_dim_idx_base =
                mma_atom_idx * MMA_ATOM_N +
                (warp_idx / 2) * (MMA_ATOM_N / FOLD_FACTOR);

#pragma unroll
            for (uint32_t i = 0;
                 i < NUM_O_TMEM_COLS_PER_ATOM / OUTPUT_SAVE_AS_SCALE_GRAN;
                 ++i) {
                uint32_t sf_block_idx =
                    head_idx_in_wv_group +
                    (head_dim_idx_base / OUTPUT_SAVE_AS_SCALE_GRAN + i) *
                        WV_GROUP_SIZE;

                *((uint8_t *)(params.out_sf + cur_job.s_q_idx +
                              wv_group_idx * params.stride_out_sf_wv_group +
                              (sf_block_idx / 4) * params.stride_out_sf_head_dim
                  ) +
                  sf_block_idx % 4) = output_sf[mma_atom_idx][i];
            }

#pragma unroll
            for (uint32_t i = 0; i < NUM_O_TMEM_COLS_PER_ATOM; i += 32) {
                uint32_t head_dim_idx = head_dim_idx_base + i;
                KU_STG_256(
                    params.out_fp8 + (uint64_t)cur_job.s_q_idx * (H_Q * D_VO) +
                        wv_group_idx * (WV_GROUP_SIZE * D_VO) +
                        head_idx_in_wv_group * 32 +
                        head_dim_idx * WV_GROUP_SIZE,
                    output_fp8[mma_atom_idx] + i,
                    "no_allocate",
                    "evict_first"
                );
            }
        }
    }

    __device__ __forceinline__ void store_o(
        const Job &cur_job, bool is_last_job
    ) {
        smem.bar_li_mi_full.wait(cur_job.job_idx_mod_2);
        float li = 0.0f;
        float mi = smem.rowwise_mi_buf[idx_in_warpgroup % H_Q_PER_CTA];

        li = smem.rowwise_li_buf[idx_in_warpgroup] +
             smem.rowwise_li_buf[idx_in_warpgroup ^ 64];

        smem.bar_li_mi_empty.arrive();

        if (idx_in_warpgroup < H_Q_PER_CTA) {
            // LSE 是 [Q, head]，64 个 head 各写自己的位置。
            uint32_t global_index = cur_job.s_q_idx * H_Q + idx_in_warpgroup;
            // 内部用 exp2，转回自然对数需要 mi*ln(2)，不是乘无穷大。
            float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
            cur_lse = cur_lse == -CUDART_INF_F ? CUDART_INF_F : cur_lse;
            params.lse[global_index] = cur_lse;
        }

        float attn_sink =
            params.attn_sink == nullptr
                ? -CUDART_INF_F
                : __ldg(params.attn_sink + idx_in_warpgroup % H_Q_PER_CTA) *
                      CUDART_L2E_F;
        float output_scale =
            li == 0.0f ? 0.0f : __fdividef(1.0f, li + exp2f(attn_sink - mi));

        smem.bar_tO_full.wait(cur_job.job_idx_mod_2);
        ku::tcgen05_after_thread_sync();

        static constexpr uint32_t MMA_ATOM_N = 256;
        static constexpr uint32_t NUM_MMA_ATOMS = D_VO / MMA_ATOM_N;
        static constexpr uint32_t NUM_O_TMEM_COLS_PER_ATOM =
            MMA_ATOM_N / FOLD_FACTOR;
        static constexpr uint32_t EPILOGUE_TILE_SIZE = O_QUANT_TILE_SIZE;
        static constexpr uint32_t NUM_EPILOGUE_TILES_PER_ATOM =
            NUM_O_TMEM_COLS_PER_ATOM / EPILOGUE_TILE_SIZE;

        fp8_e4m3 output_fp8[NUM_MMA_ATOMS][NUM_O_TMEM_COLS_PER_ATOM];
        uint8_t output_sf[NUM_MMA_ATOMS]
                         [NUM_O_TMEM_COLS_PER_ATOM / O_QUANT_TILE_SIZE];
#pragma unroll
        for (uint32_t mma_atom_idx = 0; mma_atom_idx < NUM_MMA_ATOMS;
             ++mma_atom_idx) {
#pragma unroll
            for (uint32_t epilogue_tile_idx_in_atom = 0;
                 epilogue_tile_idx_in_atom < NUM_EPILOGUE_TILES_PER_ATOM;
                 ++epilogue_tile_idx_in_atom) {
                uint32_t tmem_col_base =
                    tmem_cols::O + mma_atom_idx * NUM_O_TMEM_COLS_PER_ATOM +
                    epilogue_tile_idx_in_atom * EPILOGUE_TILE_SIZE;
                float output[EPILOGUE_TILE_SIZE];

                ku::tmem_ld_32dp32bNx<EPILOGUE_TILE_SIZE>(
                    tmem_col_base, output
                );
                cutlass::arch::fence_view_async_tmem_load();

                if (mma_atom_idx + 1 == NUM_MMA_ATOMS &&
                    epilogue_tile_idx_in_atom + 1 ==
                        NUM_EPILOGUE_TILES_PER_ATOM) {
                    ku::tcgen05_before_thread_sync();
                    if (!is_last_job) {
                        smem.bar_tO_empty.arrive();
                    }
                }

                bool should_perform_rope;
                {
                    should_perform_rope = mma_atom_idx + 1 == NUM_MMA_ATOMS &&
                                          epilogue_tile_idx_in_atom >=
                                              NUM_EPILOGUE_TILES_PER_ATOM -
                                                  D_ROPE / EPILOGUE_TILE_SIZE &&
                                          warp_idx >= 2;
                    if (should_perform_rope) {
#pragma unroll
                        for (uint32_t j = 0; j < EPILOGUE_TILE_SIZE; j += 2) {
                            float2 x = *(float2 *)(output + j);
                            uint32_t src_lane =
                                j / 2 + (epilogue_tile_idx_in_atom + 1 ==
                                                 NUM_EPILOGUE_TILES_PER_ATOM
                                             ? EPILOGUE_TILE_SIZE / 2
                                             : 0);
                            float cur_cos = __shfl_sync(
                                0xffffffff, cached_cos[0], src_lane
                            );
                            float cur_sin = __shfl_sync(
                                0xffffffff, cached_sin[0], src_lane
                            );
                            float2 y = apply_rope(x, cur_cos, cur_sin);
                            *(float2 *)(output + j) = y;
                        }
                    }
                }

                output_sf[mma_atom_idx][epilogue_tile_idx_in_atom] =
                    quantize_output(
                        output,
                        output_scale,
                        output_fp8[mma_atom_idx] +
                            epilogue_tile_idx_in_atom * EPILOGUE_TILE_SIZE
                    );
            }
        }

        write_output(cur_job, output_fp8, output_sf);
    }

    // WG0：Q0 -> Q1 -> O0 -> Q2 -> O1，预取下个 Q 与当前输出重叠
    __device__ __forceinline__ void run_q_output() {
        Job cur_job = make_job(0, blockIdx.x);
        // 读取 Q 和 RoPE 系数，对 Q 做 RoPE
        // 然后等待 Q 的 TMEM 区域可写，再写进去
        load_q(cur_job);
        do {
            Job next_job = get_next_job(cur_job);
            cached_cos[0] = cached_cos[1];
            cached_sin[0] = -cached_sin[1];
            if (next_job.is_valid) {
                load_q(next_job);
            }
            store_o(cur_job, !next_job.is_valid);
            cur_job = next_job;
        } while (cur_job.is_valid);

        NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
        if (warp_idx == 0) {
            AllocatorT().free(0, 512);
        }
    }

    __device__ __forceinline__ void run_scheduler() {
        bool phase = 0;
        while (true) {
            smem.bar_clc_empty.wait(phase ^ 1);
            ku::issue_clc_query_multicast_cluster_all(
                smem.bar_clc_full, smem.clc_response_obj
            );

            smem.bar_clc_full.arrive_and_expect_tx(sizeof(smem.clc_response_obj)
            );

            smem.bar_clc_full.wait(phase & 1);
            ku::CLCResult clc_result =
                ku::get_clc_query_response<true>(smem.clc_response_obj);
            smem.bar_clc_empty.arrive();
            if (!clc_result.is_valid) {
                break;
            }
            phase ^= 1;
        }
    }

    template <KVLocation LOC>
    __device__ __forceinline__ void load_indices_tile(
        const Job &cur_job, uint32_t kv_block_idx, RingBufferState &rs
    ) {
        int *indices_base = params.indices + (int64_t)cur_job.s_q_idx *
                                                 params.stride_indices_s_q;
        int *extra_indices_base =
            params.extra_indices +
            (int64_t)cur_job.s_q_idx * params.stride_extra_indices_s_q;
        constexpr uint32_t NUM_INDICES_PER_THREAD = 2;

        auto [indices_buf_idx, indices_buf_phase] = rs.get<NUM_INDICES_BUFS>();
        smem.bar_indices_empty[indices_buf_idx].wait(indices_buf_phase ^ 1);
#pragma unroll
        for (uint32_t i = 0; i < NUM_INDICES_PER_THREAD; ++i) {
            uint32_t pos = kv_block_idx * B_TOPK + i * 32 + lane_idx;
            bool in_extra;
            if constexpr (LOC == KVLocation::ORIG) {
                in_extra = false;
            } else if constexpr (LOC == KVLocation::EXTRA) {
                in_extra = true;
            } else {
                in_extra = pos >= cur_job.num_orig_slots;
            }

            uint32_t local_pos = in_extra ? pos - cur_job.num_orig_slots : pos;
            uint32_t valid_len =
                in_extra ? cur_job.extra_topk_length : cur_job.topk_length;

            int cur_index =
                local_pos < valid_len
                    ? __ldg(
                          (in_extra ? extra_indices_base : indices_base) +
                          local_pos
                      )
                    : -1;

            int slot_count =
                in_extra
                    ? params.extra_num_blocks * params.extra_page_block_size
                    : params.num_blocks * params.page_block_size;
            bool is_index_valid = cur_index >= 0 && cur_index < slot_count;
            int64_t src_block_stride = in_extra ? params.stride_extra_kv_block
                                                : params.stride_kv_block;
            const auto &fast_divmod =
                in_extra ? aux_params.fast_divmod_extra_page_block_size
                         : aux_params.fast_divmod_page_block_size;

            uint32_t token_idx = is_index_valid ? (uint32_t)cur_index : 0;
            int idx_in_block;
            int block_idx = fast_divmod.divmod(idx_in_block, (int)token_idx);
            uint32_t row = i * 32 + lane_idx;

            uint32_t tma_k_stride = in_extra ? ExtraKVFormat::TMA_K_STRIDE
                                             : OrigKVFormat::TMA_K_STRIDE;
            uint32_t num_scales_each_token =
                in_extra ? ExtraKVFormat::NUM_SCALES_EACH_TOKEN
                         : OrigKVFormat::NUM_SCALES_EACH_TOKEN;

            smem.decode_tma_coords[indices_buf_idx][row] =
                is_index_valid ? (src_block_stride / tma_k_stride) * block_idx +
                                     idx_in_block
                               : -1;
            uint32_t page_block_size = in_extra ? params.extra_page_block_size
                                                : params.page_block_size;
            uint8_t *kv_base =
                (uint8_t *)(in_extra ? params.extra_kv : params.kv);

            uint8_t *scale_ptr = kv_base + page_block_size * tma_k_stride +
                                 (int64_t)block_idx * src_block_stride +
                                 idx_in_block * num_scales_each_token;
            bool is_fp4_token = in_extra;

            if (!is_fp4_token) {
                uint8_t *scale_dst = smem.decode_scales[indices_buf_idx] +
                                     row * NUM_SCALES_EACH_TOKEN_PER_CTA;

                __int128_t scales;
                asm volatile("ld.global.nc.L1::no_"
                             "allocate.b128 %0, [%1];"
                             : "=q"(scales)
                             : "l"((uint64_t)scale_ptr));
                *(__int128_t *)scale_dst = is_index_valid ? scales : 0;
            } else {
                smem.decode_scale_ptrs[indices_buf_idx][row] =
                    is_index_valid ? scale_ptr : nullptr;
            }

            uint32_t mask = __ballot_sync(0xffffffff, is_index_valid);
            if (lane_idx == 0) {
                *((uint32_t *)smem.is_k_valid[indices_buf_idx] + i) = mask;
            }
        }

        smem.bar_indices_full[indices_buf_idx].arrive();
        rs.update();
    }

    __device__ __forceinline__ void run_indices() {
        Job cur_job = make_job(0, blockIdx.x);
        RingBufferState rs;
        do {
            for_each_kv_tile(cur_job, [&]<KVLocation LOC>(uint32_t tile) {
                load_indices_tile<LOC>(cur_job, tile, rs);
            });
            cur_job = get_next_job(cur_job);
        } while (cur_job.is_valid);
    }

    __device__ __forceinline__ uint32_t get_row_idx(uint32_t token_idx) {
        return local_warp_idx * NUM_ROWS_PER_WARP +
               token_idx * NUM_ROWS_PER_WAVEFRONT + row_in_wavefront;
    }

    template <typename F>
    __device__ __forceinline__ uint32_t get_chunk_base(uint32_t half) {
        constexpr uint32_t NUM_CHUNKS_PER_LANE =
            F::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW;

        if constexpr (NUM_CHUNKS_PER_LANE <= 4) {
            return (idx_in_row * NUM_CHUNKS_PER_LANE + chunk_rotation) %
                   F::NUM_CHUNKS_PER_ROW;
        } else {
            return idx_in_row * NUM_CHUNKS_PER_LANE +
                   (half * 4 ^ chunk_rotation);
        }
    }

    __device__ __forceinline__ uint32_t get_raw_row_offset(uint32_t row) {
        return row / 4 * RAW_KV_GROUP_BYTES;
    }

    template <uint32_t NUM_FP8_TOKENS>
    __device__ __forceinline__ void for_each_token(auto callable) {
        cute::for_each(
            cute::make_int_sequence<NUM_TOKENS_PER_THREAD>{},
            [&](auto token_idx) {
                if constexpr (token_idx < NUM_FP8_TOKENS) {
                    callable.template operator()<OrigKVFormat>(token_idx);
                } else {
                    callable.template operator()<ExtraKVFormat>(token_idx);
                }
            }
        );
    }

    template <typename F>
    __device__ __forceinline__ void load_raw_token(
        uint32_t token_idx,
        uint8_t *slot_base,
        uint32_t (&cached_input)[NUM_TOKENS_PER_THREAD][MAX_CHUNKS_PER_LANE][4]
    ) {
        uint32_t row = get_row_idx(token_idx);
        uint8_t *row_base = slot_base + get_raw_row_offset(row) +
                            row % 4 * F::RAW_TOKEN_SMEM_STRIDE;

#pragma unroll
        for (uint32_t g = 0; g < F::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW;
             ++g) {
            *(__int128_t *)(cached_input[token_idx][g]) = ku::ld_shared(
                row_base + get_chunk_base<F>(g / 4) * 16 + g % 4 * 16
            );
        }
    }

    template <typename F>
    __device__ __forceinline__ void dequantize_token(
        uint32_t token_idx,
        uint8_t *slot_base,
        const uint32_t (&cached_input
        )[NUM_TOKENS_PER_THREAD][MAX_CHUNKS_PER_LANE][4],
        const uint32_t (&cached_scales)[NUM_TOKENS_PER_THREAD][2]
    ) {
        static constexpr uint32_t C_LANE =
            F::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW;
        static constexpr uint32_t NUM_STS_PER_CHUNK =
            F::CHUNK_ELEMS * sizeof(bf16) / 16;
        ku::nvbf16x2 fp4_scales[2];
#pragma unroll
        for (uint32_t g = 0; g < C_LANE; ++g) {
            ku::nvbf16x2 data_bf16x2[F::CHUNK_ELEMS / 2];
            if constexpr (F::IS_FP4) {
                if (g % 2 == 0) {
                    fp8x4_to_bf16x2x2(
                        cached_scales[token_idx][g / 2], fp4_scales
                    );
                }

#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    fp4x8_to_bf16x2x4(
                        cached_input[token_idx][g][j], data_bf16x2 + j * 4
                    );
                }

                ku::nvbf16x2 scale_lo = __low2bfloat162(fp4_scales[g % 2]),
                             scale_hi = __high2bfloat162(fp4_scales[g % 2]);

                // 数组元素是 BF16 二元组，元素数为标量数的一半。
#pragma unroll
                for (uint32_t k = 0; k < F::CHUNK_ELEMS / 2; ++k) {
                    data_bf16x2[k] = __hmul2(
                        data_bf16x2[k],
                        k < F::QUANT_TILE_SIZE / 2 ? scale_lo : scale_hi
                    );
                }
            } else {
                __nv_fp8_e8m0 scale = ((__nv_fp8_e8m0 *)cached_scales[token_idx]
                )[g * F::CHUNK_ELEMS / F::QUANT_TILE_SIZE];

                // 数组元素是 BF16 二元组，元素数为标量数的一半。
#pragma unroll
                for (uint32_t k = 0; k < F::CHUNK_ELEMS / 2; ++k) {
                    data_bf16x2[k] = fp8x2_to_bf16x2_with_scale(
                        ((ku::nve4m3x2 *)cached_input[token_idx][g])[k], scale
                    );
                }
            }

            static constexpr uint32_t CHUNKS_PER_ATOM_ROW =
                128 / (F::CHUNK_ELEMS * 2);
            uint32_t atom_col = get_chunk_base<F>(g / 4) / CHUNKS_PER_ATOM_ROW +
                                g % 4 / CHUNKS_PER_ATOM_ROW;

#pragma unroll
            for (uint32_t j = 0; j < NUM_STS_PER_CHUNK; ++j) {
                ku::st_shared(
                    slot_base +
                        sts_offsets
                            [g % CHUNKS_PER_ATOM_ROW * NUM_STS_PER_CHUNK + j] +
                        atom_col * STS_ATOM_COL_STRIDE_BYTES +
                        token_idx * STS_TOKEN_STRIDE_BYTES,
                    *(__int128_t *)(data_bf16x2 + j * 4)
                );
            }
        }
    }

    template <uint32_t NUM_FP8_TOKENS>
    __device__ __forceinline__ void process_block(
        uint32_t num_orig_rows, RingBufferState &rs
    ) {
        auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
        smem.bar_indices_full[indices_buf_idx].wait(indices_bar_phase);

        static constexpr uint32_t FP4_SCALE_BYTES_PER_LANE =
            ExtraKVFormat::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW * 2;

        uint32_t cached_scales[NUM_TOKENS_PER_THREAD]
                              [FP4_SCALE_BYTES_PER_LANE / 4];
        const uint8_t *scale_ptrs[NUM_TOKENS_PER_THREAD];
        for_each_token<NUM_FP8_TOKENS>([&]<typename F>(uint32_t token_idx) {
            uint32_t row = get_row_idx(token_idx);
            if constexpr (F::IS_FP4) {
                scale_ptrs[token_idx] =
                    smem.decode_scale_ptrs[indices_buf_idx][row];
            } else {
                const uint8_t *scale_src = smem.decode_scales[indices_buf_idx] +
                                           row * NUM_SCALES_EACH_TOKEN_PER_CTA;
                uint32_t scales = *(uint32_t *)(scale_src + idx_in_row * 4);
                cached_scales[token_idx][0] =
                    chunk_rotation ? __byte_perm(scales, scales, 0x1032)
                                   : scales;
            }
        });

        auto [kv_slot_idx, kv_bar_phase] = rs.get<NUM_KV_SLOTS>();
        // 生产者等上一轮 empty；第一轮 phase=0 时等 phase=1，允许首次填槽。
        smem.bar_kv_slot_empty[kv_slot_idx].wait(kv_bar_phase ^ 1);
        uint8_t *slot_base = (uint8_t *)smem.kv_slots[kv_slot_idx];

        cute::for_each(
            cute::make_int_sequence<NUM_ROWS_PER_WARP / 4>{},
            [&](auto i) {
                uint32_t row_start = local_warp_idx * NUM_ROWS_PER_WARP + i * 4;
                constexpr bool is_fp4 =
                    i / (NUM_ROWS_PER_WAVEFRONT / 4) >= NUM_FP8_TOKENS;
                auto tensor_map = is_fp4 ? &tma_params.fp4 : &tma_params.fp8;
                if (elect_one_sync()) {
                    int4 coords =
                        *(int4 *)(smem.decode_tma_coords[indices_buf_idx] +
                                  row_start);
                    ku::tma_gather4(
                        tensor_map,
                        smem.bar_raw_kv_full,
                        slot_base + get_raw_row_offset(row_start),
                        0,
                        coords,
                        (int64_t)TMA::CacheHintSm90::EVICT_FIRST
                    );
                }
            }
        );

        smem.bar_indices_empty[indices_buf_idx].arrive();

        for_each_token<NUM_FP8_TOKENS>([&]<typename F>(uint32_t token_idx) {
            if constexpr (F::IS_FP4) {
                uint64_t scale_addr =
                    ((uint64_t)scale_ptrs[token_idx] + get_chunk_base<F>(0) * 2
                    );

                uint64_t scales = 0;
                if (scale_ptrs[token_idx] != nullptr) {
                    asm volatile("ld.global.nc.L1::no_"
                                 "allocate.b64 %0, [%1];"
                                 : "=l"(scales)
                                 : "l"(scale_addr)
                                 : "memory");
                }
                *(uint64_t *)cached_scales[token_idx] = scales;
            }
        });

        if (idx_in_warpgroup == 0) {
            smem.bar_raw_kv_full.arrive_and_expect_tx(
                num_orig_rows * OrigKVFormat::RAW_TOKEN_SMEM_STRIDE +
                (B_TOPK - num_orig_rows) * ExtraKVFormat::RAW_TOKEN_SMEM_STRIDE
            );
        }
        smem.bar_raw_kv_full.wait(rs.get<1>().second);

        uint32_t cached_input[NUM_TOKENS_PER_THREAD][MAX_CHUNKS_PER_LANE][4];

        for_each_token<NUM_FP8_TOKENS>([&]<typename F>(uint32_t token_idx) {
            load_raw_token<F>(token_idx, slot_base, cached_input);
        });

        NamedBarrier::arrive_and_wait(128, 7);

        for_each_token<NUM_FP8_TOKENS>([&]<typename F>(uint32_t token_idx) {
            dequantize_token<F>(
                token_idx, slot_base, cached_input, cached_scales
            );
        });

        fence_view_async_shared();
        smem.bar_kv_slot_full[kv_slot_idx].arrive();
        rs.update();
    }

    __device__ __forceinline__ void for_each_kv_tile(
        const Job &cur_args, auto callable
    ) {
        uint32_t num_full_orig_blocks =
            std::min(cur_args.num_kv_blocks, cur_args.num_orig_slots / B_TOPK);
        bool has_mixed_block = num_full_orig_blocks < cur_args.num_kv_blocks &&
                               cur_args.num_orig_slots % B_TOPK != 0;
#pragma unroll 1
        for (uint32_t kv_block_idx = 0; kv_block_idx < num_full_orig_blocks;
             ++kv_block_idx) {
            callable.template operator()<KVLocation::ORIG>(kv_block_idx);
        }
        if (has_mixed_block) {
            callable.template operator(
            )<KVLocation::ORIG_AND_EXTRA>(num_full_orig_blocks);
        }

#pragma unroll 1
        for (uint32_t kv_block_idx = num_full_orig_blocks + has_mixed_block;
             kv_block_idx < cur_args.num_kv_blocks;
             ++kv_block_idx) {
            callable.template operator()<KVLocation::EXTRA>(kv_block_idx);
        }
    }

    __device__ __forceinline__ void run_kv() {
        {
            Tensor sKV = make_tensor(
                make_smem_ptr(smem.kv_slots[0]),
                ku::make_umma_canonical_k_major_layout<B_TOPK, D_QK, 128>()
            );
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                sts_offsets[i] = (uint32_t)((&sKV(get_row_idx(0), i * 8) -
                                             smem.kv_slots[0]) *
                                            sizeof(bf16));
            }
        }

        Job cur_job = make_job(0, blockIdx.x);
        RingBufferState rs;
        do {
            for_each_kv_tile(cur_job, [&]<KVLocation LOC>(uint32_t tile) {
                if constexpr (LOC == KVLocation::ORIG) {
                    process_block<2>(64, rs);
                } else if constexpr (LOC == KVLocation::EXTRA) {
                    process_block<0>(0, rs);
                } else {
                    uint32_t rows = cur_job.num_orig_slots - tile * B_TOPK;
                    uint32_t count =
                        std::clamp(
                            int(rows) - int(local_warp_idx * 16), 0, 16
                        ) /
                        8;
                    switch (count) {
                    case 0:
                        process_block<0>(rows, rs);
                        break;
                    case 1:
                        process_block<1>(rows, rs);
                        break;
                    case 2:
                        process_block<2>(rows, rs);
                        break;
                    }
                }
            });
            cur_job = get_next_job(cur_job);
        } while (cur_job.is_valid);
    }

    __device__ __forceinline__ void issue_qk(
        const Job &job,
        uint32_t kv_block_idx,
        RingBufferState &rs,
        auto &tQ,
        auto &tP
    ) {
        if (kv_block_idx == 0) {
            smem.bar_tQ_full.wait(job.job_idx_mod_2);
        }
        auto [kv_slot_idx, kv_bar_phase] = rs.get<NUM_KV_SLOTS>();
        Tensor sK = make_tensor(
            make_smem_ptr(smem.kv_slots[kv_slot_idx]),
            ku::make_umma_canonical_k_major_layout<
                B_TOPK * NUM_MRGEMM_RAILS,
                D_QK / NUM_MRGEMM_RAILS,
                128>()
        );

        smem.bar_kv_slot_full[kv_slot_idx].arrive();

        smem.bar_kv_slot_full[kv_slot_idx].wait(kv_bar_phase);
        auto [p_buf_idx, p_bar_phase] = rs.get<NUM_P_BUFS>();

        tP.data().get() = tmem_cols::get_p(p_buf_idx);

        ku::tcgen05_after_thread_sync();
        auto mma = TiledMMA_QK{};
        ku::utcmma_ts(mma, tQ, sK, tP, true);
        ku::umma_arrive_noelect(smem.bar_tP_full[p_buf_idx]);

        if (kv_block_idx == job.num_kv_blocks - 1) {
            ku::umma_arrive_noelect(smem.bar_tQ_empty);
        }
        rs.update();
    }

    __device__ __forceinline__ void issue_sv(
        const Job &job,
        uint32_t kv_block_idx,
        RingBufferState &rs,
        auto &sS,
        auto &tO
    ) {
        if (kv_block_idx == 0) {
            smem.bar_tO_empty.wait(job.job_idx_mod_2 ^ 1);
        }
        auto [kv_slot_idx, _] = rs.get<NUM_KV_SLOTS>();
        smem.bar_SO_full.wait(rs.get<1>().second);
        Tensor sV = make_tensor(
            // 逻辑块号可超过 2；实际只分配了 3 个循环使用的槽。
            make_smem_ptr(smem.kv_slots[kv_slot_idx]),
            ku::make_umma_canonical_mn_major_layout<D_VO, B_TOPK, 128>()
        );

        ku::tcgen05_after_thread_sync();
        auto mma = TiledMMA_SV{};

        ku::utcmma_ss(mma, sS, sV, tO, kv_block_idx == 0);

        ku::umma_arrive_noelect(smem.bar_kv_slot_empty[kv_slot_idx]);
        ku::umma_arrive_noelect(smem.bar_SO_empty);
        if (kv_block_idx == job.num_kv_blocks - 1) {
            ku::umma_arrive_noelect(smem.bar_tO_full);
        }
        rs.update();
    }

    // warp 8 的一个线程发射 MMA；QK 和前一块 SV 交错执行。
    __device__ __forceinline__ void run_mma() {
        auto tiled_mma_qk = TiledMMA_QK{};
        auto tiled_mma_sv = TiledMMA_SV{};
        Tensor tQ =
            tiled_mma_qk.get_slice(_0{}).make_fragment_A(partition_shape_A(
                tiled_mma_qk,
                Shape<Int<H_Q_PER_CTA>, Int<D_QK / NUM_MRGEMM_RAILS>>{}
            ));
        Tensor tP = partition_fragment_C(
            tiled_mma_qk,
            Shape<Int<H_Q_PER_CTA>, Int<B_TOPK * NUM_MRGEMM_RAILS>>{}
        );
        Tensor sS = make_tensor(
            make_smem_ptr(smem.s),
            ku::make_umma_canonical_k_major_layout<H_Q_PER_CTA, B_TOPK, 0>()
        );
        Tensor tO = partition_fragment_C(
            tiled_mma_sv, Shape<Int<H_Q_PER_CTA>, Int<D_VO>>{}
        );
        tQ.data().get() = tmem_cols::Q;
        tO.data().get() = tmem_cols::O;

        RingBufferState rs_qk, rs_sv;
        Job cur_job = make_job(0, blockIdx.x);
        issue_qk(cur_job, 0, rs_qk, tQ, tP);
        do {
#pragma unroll 1
            for (uint32_t kv_block_idx = 1;
                 kv_block_idx < cur_job.num_kv_blocks;
                 ++kv_block_idx) {
                issue_qk(cur_job, kv_block_idx, rs_qk, tQ, tP);
                issue_sv(cur_job, kv_block_idx - 1, rs_sv, sS, tO);
            }

            Job next_job = get_next_job(cur_job);
            if (next_job.is_valid) {
                issue_qk(next_job, 0, rs_qk, tQ, tP);
            }
            issue_sv(cur_job, cur_job.num_kv_blocks - 1, rs_sv, sS, tO);

            cur_job = next_job;
        } while (cur_job.is_valid);
    }

    __device__ __forceinline__ void run_softmax() {
        Job cur_job = make_job(0, blockIdx.x);
        uint32_t local_warp_idx = warp_idx - 12;

        bf16 *sS_base = smem.s +
                        (local_warp_idx >= 2 ? H_Q_PER_CTA * (B_TOPK / 2) : 0) +
                        (idx_in_warpgroup % H_Q_PER_CTA) * 8;

        RingBufferState rs;
        do {
            static constexpr uint32_t NUM_ELEMS_PER_THREAD =
                B_TOPK * H_Q_PER_CTA / 128;
            float mi = MAX_INIT_VAL;
            float li = 0.0f;
            float real_mi = -CUDART_INF_F;

            float score_multiplier = params.sm_scale_div_log2;

            smem.bar_tQ_empty.arrive();
#pragma unroll 1
            for (uint32_t kv_block_idx = 0;
                 kv_block_idx < cur_job.num_kv_blocks;
                 ++kv_block_idx) {
                auto [indices_buf_idx, indices_bar_phase] =
                    rs.get<NUM_INDICES_BUFS>();
                auto [p_buf_idx, p_buf_phase] = rs.get<NUM_P_BUFS>();
                smem.bar_tP_full[p_buf_idx].wait(p_buf_phase);
                smem.bar_indices_full[indices_buf_idx].wait(indices_bar_phase);
                ku::tcgen05_after_thread_sync();

                float p[NUM_ELEMS_PER_THREAD];
                // QK 分成两条 256 维部分点积，算完再相加
                retrieve_mask_and_reduce_p<
                    NUM_ELEMS_PER_THREAD,
                    barrier_ids::WG3_WARP02_SYNC,
                    barrier_ids::WG3_WARP13_SYNC,
                    false>(
                    tmem_cols::get_p(p_buf_idx),
                    (char *)&smem.is_k_valid[indices_buf_idx],
                    local_warp_idx,
                    lane_idx,
                    []() {},
                    smem.p_exchange_buf,
                    p
                );

                float cur_pi_max = get_max<NUM_ELEMS_PER_THREAD>(p);
                cur_pi_max *= score_multiplier;

                smem.rowwise_max_buf[idx_in_warpgroup] = cur_pi_max;
                NamedBarrier::arrive_and_wait(
                    64, barrier_ids::WG3_WARP02_SYNC + (local_warp_idx & 1)
                );
                smem.bar_indices_empty[indices_buf_idx].arrive();

                cur_pi_max =
                    max(cur_pi_max,
                        smem.rowwise_max_buf[idx_in_warpgroup ^ 64]);
                real_mi = max(real_mi, cur_pi_max);

                bool should_scale_o =
                    __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

                float new_max, scale_for_old;
                if (!should_scale_o) {
                    scale_for_old = 1.0f;
                    new_max = mi;
                } else {
                    new_max = max(cur_pi_max, mi);
                    scale_for_old = exp2f(mi - new_max);
                }
                mi = new_max;

                nv_bfloat16 s[NUM_ELEMS_PER_THREAD];
                float cur_sum = get_s_from_p<NUM_ELEMS_PER_THREAD>(
                    (nv_bfloat162 *)s, p, score_multiplier, new_max
                );
                li = fmaf(li, scale_for_old, cur_sum);

                smem.bar_SO_empty.wait(rs.get<1>().second ^ 1);

#pragma unroll
                for (int i = 0; i < NUM_ELEMS_PER_THREAD / 8; ++i) {
                    ku::st_shared(
                        sS_base + i * 8 * H_Q_PER_CTA,
                        *(__int128_t *)(s + i * 8)
                    );
                }

                if (kv_block_idx > 0 && should_scale_o) {
                    ku::tcgen05_after_thread_sync();
                    rescale_O<D_VO / FOLD_FACTOR, 32, tmem_cols::O>(
                        scale_for_old
                    );
                    ku::tcgen05_before_thread_sync();
                }

                fence_view_async_shared();
                ku::tcgen05_before_thread_sync();
                smem.bar_SO_full.arrive();
                rs.update();
            }

            if (real_mi == -CUDART_INF_F) {
                li = 0.0f;
                mi = -CUDART_INF_F;
            }

            smem.bar_li_mi_empty.wait(cur_job.job_idx_mod_2 ^ 1);
            if (local_warp_idx < H_Q_PER_CTA / 32) {
                smem.rowwise_mi_buf[idx_in_warpgroup] = mi;
            }
            smem.rowwise_li_buf[idx_in_warpgroup] = li;
            smem.bar_li_mi_full.arrive();

            cur_job = get_next_job(cur_job);
        } while (cur_job.is_valid);
    }
};

__global__ void __launch_bounds__(512, 1, 1) attention_kernel(
    __grid_constant__ const Params params,
    __grid_constant__ const TMAParams tma_params,
    __grid_constant__ const AuxParams aux_params
) {
    extern __shared__ char buffer[];
    Kernel(
        params,
        tma_params,
        aux_params,
        *reinterpret_cast<SharedMemoryPlan *>(buffer)
    )
        .run();
}

void validate_layout(const MegaAttnParams &a) {
    if (a.n <= 0 || a.topk <= 0 || a.topk % 8 || a.extra_topk <= 0)
        throw std::invalid_argument(
            "N/topk/extra_topk must be positive; topk must be a multiple of 8"
        );
    if (a.page_size <= 0 || a.num_pages <= 0 || a.extra_page_size <= 0 ||
        a.extra_num_pages <= 0 || a.page_stride < int64_t(a.page_size) * 528 ||
        a.page_stride % 512 ||
        a.extra_page_stride < int64_t(a.extra_page_size) * 288 ||
        a.extra_page_stride % 512)
        throw std::invalid_argument(
            "Invalid KV page dimensions or 512-byte page stride"
        );
    if (a.indices_stride < a.topk || a.extra_indices_stride < a.extra_topk ||
        a.indices_stride % 8 || a.extra_indices_stride % 8 ||
        a.sf_stride < a.n || a.sf_stride % 4)
        throw std::invalid_argument("Invalid index/scale strides");
}

Params make_params(const MegaAttnParams &a, cudaStream_t stream) {
    Params p{};
    p.s_q = a.n;

    // 将 exp(x) 改写为 exp2(x*log2(e))，把 log2(e) 提前并入缩放系数。
    p.sm_scale_div_log2 = a.sm_scale * 1.4426950408889634f;

    p.num_blocks = a.num_pages;
    p.page_block_size = a.page_size;
    p.topk = a.topk;
    p.extra_num_blocks = a.extra_num_pages;
    p.extra_page_block_size = a.extra_page_size;
    p.extra_topk = a.extra_topk;
    p.q = (bf16 *)a.q;
    // 上游参数沿用 bf16* 类型，但这里传的是压缩字节地址；真正读取时按 KVFormat
    // 和字节步长解释。
    p.kv = (bf16 *)a.kv_fp8;
    p.extra_kv = (bf16 *)a.kv_fp4;
    p.indices = (int *)a.indices;
    p.extra_indices = (int *)a.extra_indices;
    p.topk_length = (int *)a.topk_length;
    p.extra_topk_length = (int *)a.extra_topk_length;
    p.attn_sink = (float *)a.attn_sink;
    p.lse = a.lse;
    p.stride_q_s_q = 64 * 512;

    p.stride_kv_block = a.page_stride;

    p.stride_extra_kv_block = a.extra_page_stride;

    p.stride_indices_s_q = a.indices_stride;
    p.stride_extra_indices_s_q = a.extra_indices_stride;

    p.stream = stream;

    p.token_positions = (uint32_t *)a.positions;

    p.cos_sin_cache = (float *)a.cos_sin;

    p.out_fp8 = (fp8_e4m3 *)a.out_fp8;
    p.out_sf = a.out_sf;
    // 每组 8 个 head×每 head 16 个 scale=128 字节=32 个 uint32，所以组间跨 32
    // 个 scale 行。
    // 接head11、维度160的例子：若Q编号2、sf_stride=4，字下标=2+1×(32×4)+10×4=170。
    // 字节3在总缓冲区的偏移为170×4+3=683
    // B；sf_stride单位是uint32个数，不是字节。
    p.stride_out_sf_wv_group = 32 * a.sf_stride;
    p.stride_out_sf_head_dim = a.sf_stride;

    return p;
}

// 两种缓存只差行宽和步长，TMA 描述符构造共用这一份代码。
template <typename F>
CUtensorMap make_kv_map(void *data, int pages, int64_t stride) {
    // TMA 按 UINT32 描述原始字节，行宽/box 宽度都除以 4，满足 box 维度上限。
    // box 比实际数据行宽 16 字节，越界部分补零；奇数个 16-byte chunk
    // 的行距用于减少 bank 冲突。
    // FP8：全局行宽512/4=128个uint32，box宽(512+16)/4=132；FP4则是64和68。
    // 例：FP8有5页、页步长3072
    // B，TMA二维尺寸为[128,5×3072/512]=[128,30]，行距512 B。
    return ku::make_tensor_map(
        {uint64_t(F::RAW_TOKEN_DATA_BYTES) / 4,
         uint64_t(pages) * (stride / F::TMA_K_STRIDE)},
        {uint64_t(F::TMA_K_STRIDE)},
        {F::RAW_TOKEN_SMEM_STRIDE / 4, 1},
        data,
        CU_TENSOR_MAP_DATA_TYPE_UINT32,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B
    );
}
void check_cuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}

void launch(const Params &params) {
    TMAParams maps{
        make_kv_map<OrigKVFormat>(
            params.kv, params.num_blocks, params.stride_kv_block
        ),
        make_kv_map<ExtraKVFormat>(
            params.extra_kv,
            params.extra_num_blocks,
            params.stride_extra_kv_block
        )
    };
    AuxParams divisors{
        cutlass::FastDivmod(params.page_block_size),
        cutlass::FastDivmod(params.extra_page_block_size)
    };
    constexpr size_t bytes = sizeof(SharedMemoryPlan);
    check_cuda(cudaFuncSetAttribute(
        attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes
    ));

    cutlass::ClusterLaunchParams config{
        dim3(params.s_q), dim3(NUM_THREADS), dim3(1), bytes, params.stream
    };
    auto status = cutlass::launch_kernel_on_cluster(
        config, (void *)attention_kernel, params, maps, divisors
    );
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("SM100 kernel launch failed");
    }
    check_cuda(cudaGetLastError());
}
} // namespace mega_attn_v1

extern "C" void solve_v1(const MegaAttnParams &a, cudaStream_t stream) {
    mega_attn_v1::validate_layout(a);
    mega_attn_v1::launch(mega_attn_v1::make_params(a, stream));
}
