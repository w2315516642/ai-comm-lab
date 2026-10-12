#include <cuda_fp8.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/cluster_launch.hpp>
#include <math_constants.h>

#include <kernels/utils.h>

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
            cutlass::arch::warpgroup_reg_alloc<72>();
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
            for (uint32_t i = 0; i < NUM_KV_SLOTS; ++i) {
                smem.bar_kv_slot_full[i].init(129);
                smem.bar_kv_slot_empty[i].init(1);
            }
        }
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
        for (uint32_t local_tile_idx = TILES_PER_THREAD; local_tile_idx >= 0;
             --local_tile_idx) {
            uint32_t tile_idx = local_tile_idx * 2 + warp_idx / 2;

            for (uint32_t i = 0; i < TILE_SIZE / NUM_BF16_PER_LOAD; ++i) {
                uint32_t h_q_idx = idx_in_warpgroup % H_Q_PER_CTA;
                uint32_t d_q_idx = tile_idx * TILE_SIZE + i * NUM_BF16_PER_LOAD;

                KU_LDG_256(
                    q_token_base + h_q_idx * NUM_BF16_PER_LOAD,
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
                    *(nv_bfloat162 *)(cached_q + local_tile_idx * TILE_SIZE +
                                      j) = nv_bfloat162{
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

    __device__ __forceinline__ void store_o(
        const Job &cur_job, bool is_last_job
    ) {
        smem.bar_li_mi_full.wait(cur_job.job_idx_mod_2);
        float li = 0.0f;
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
                D_QK * NUM_MRGEMM_RAILS,
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
            make_smem_ptr(smem.kv_slots[kv_block_idx]),
            ku::make_umma_canonical_mn_major_layout<D_VO, B_TOPK, 128>()
        );

        ku::tcgen05_after_thread_sync();
        auto mma = TiledMMA_SV{};

        ku::utcmma_ss(mma, sS, sV, tO, kv_block_idx == 0);

        ku::umma_arrive_noelect(smem.bar_kv_slot_empty[kv_block_idx]);
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
            // TODO:
        } while (cur_job.is_valid);
    }
};

} // namespace mega_attn_v1
