// 改编自 FlashMLA；Copyright (c) 2025 DeepSeek，MIT 许可。
// 上游版本 0eee43b12f034b657133cf2afca6a72ebb6efccf，许可见
// third_party/flashmla/LICENSE。
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

namespace mega_attn_v0 {
using namespace mega_attn;
using namespace sm100;
// 符号约定：P=QK 的分数，S=exp2(缩放后的 P-mi)，O=累积的 S×V。
// tQ/tP/tO 是 TMEM 视图，sK/sV/sS 是 shared memory 视图；sf 表示量化 scale。
// MAX_INIT_VAL 用有限负数作为初始归一化基准，避免第一块出现 -inf-(-inf)。
// 量化下限让全零输出也能得到非零 scale，从而安全计算它的倒数。
constexpr float MAX_INIT_VAL = -1e30f, O_QUANT_CLAMP_MIN_VALUE = 1e-4f;

__device__ __forceinline__ float2
apply_rope(const float2 &x, float c, float s) {
    // 相邻两维旋转为 (x0*cos-x1*sin, x0*sin+x1*cos)；输出时传入负 sin 即逆旋转。
    return ku::float2_fma({x.x, x.x}, {c, s}, {-x.y * s, x.y * c});
}

// 标记当前 KV 块来自主缓存、额外缓存，还是两者交界处。
enum class KVLocation { ORIG, ORIG_AND_EXTRA, EXTRA };
// 记录一个 Q 的处理任务，包括索引长度、KV 块数和同步阶段。
struct Job {
    // 没有后续 CLC 任务时为 false，各条流水线据此结束。
    bool is_valid;
    // s_q_idx：当前 Q 的序号；job_idx_mod_2：本 CTA 处理任务的奇偶阶段。
    // topk_length：主索引行的有效前缀长度；num_kv_blocks：合并后要处理的 64-token 块数。
    uint32_t s_q_idx, job_idx_mod_2, topk_length, num_kv_blocks;
    // extra_topk_length 是额外索引的有效长度；num_orig_slots 固定为主 topk 容量。
    // 额外索引从 num_orig_slots 开始拼接，不会因某个 Q 的主有效长度较短而前移。
    uint32_t extra_topk_length, num_orig_slots;
};

// 保存当前线程的执行状态，并组织各 warp 的搬运、计算和输出。
class Kernel {
    // Kernel 对象属于当前线程；smem 引用的内存由整个 CTA 共享，普通成员则是线程私有状态。
    const Params &params;
    const TMAParams &tma_params;
    const AuxParams &aux_params;
    SharedMemoryPlan &smem;
    // 全 CTA 有 16 个 warp；三个编号分别是 warp [0,15]、组内线程 [0,127]、lane [0,31]。
    uint32_t warp_idx, idx_in_warpgroup, lane_idx;
    // 仅 WG0 使用：保留当前输出和下个 Q 的 RoPE 系数。
    // 下标 1 暂存最新加载的 Q 的系数；下标 0 留给当前 O，避免预取下个 Q 时覆盖。
    float cached_cos[2]{}, cached_sin[2]{};
    // 仅 WG1 使用：每行由 4 个 lane 协作，每线程处理两行。
    // 这里 token 指本线程参与的行，不是全局 token ID；chunk 指 16 字节压缩数据。
    // lane 0/8/16/24 合作处理同一行，lane 1/9/17/25 处理下一行，依此类推。
    // 一个 warp 每轮覆盖 8 行、共两轮；四个 warp 合计覆盖一个 64 行 KV 块。
    // 数值对应：32 lane/4 lane每行=8 行每轮；64 行/(4 warp×8 行每轮)=每线程参与 2 行。
    // 所以每 warp 覆盖 2×8=16 行，四个 warp 刚好覆盖 64 行。
    static constexpr uint32_t NUM_LANES_PER_ROW = 4, NUM_ROWS_PER_WAVEFRONT = 8;
    static constexpr uint32_t NUM_TOKENS_PER_THREAD = 2, NUM_ROWS_PER_WARP = 16;
    // 每 lane 分到 FP8 行的 8 个 chunk 或 FP4 行的 4 个 chunk，数组按较大的格式分配。
    // FP8：512 B/16 B每chunk/4 lane=8 chunk/lane；FP4：256/16/4=4 chunk/lane。
    static constexpr uint32_t MAX_CHUNKS_PER_LANE = 8;
    // MMA 的 shared memory 布局按每行 128 字节的块组织。
    // 横移一个块跨过 B_TOPK 行；同线程的第二行相对第一行下移 8 行。
    // 字节步长：横移一个 128 B 列块跨 64×128=8192 B；第二行下移 8×128=1024 B。
    static constexpr uint32_t STS_ATOM_COL_STRIDE_BYTES = B_TOPK * 128;
    static constexpr uint32_t STS_TOKEN_STRIDE_BYTES = 8 * 128;
    // 仅 WG1 使用：local_warp_idx 为组内 warp；row_in_wavefront 为当前 8 行中的行号。
    // idx_in_row 为协作 lane 的编号 [0,3]；chunk_rotation 为读取 chunk 时的轮换偏移。
    // run_softmax 中同名 local_warp_idx 是 WG3 的局部变量，与这里的 WG1 编号不同。
    uint32_t local_warp_idx, row_in_wavefront, idx_in_row, chunk_rotation;
    // 缓存首行 8 个 16 字节片段的 swizzle 后地址偏移，后续写回只需加行/列偏移。
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

    // 512 个线程分工；以下入口各自维护跨任务的环形缓冲区 phase。
    __device__ __forceinline__ void run() {
        initialize();
        // 一个 warpgroup=4 个 warp=128 个线程；用 warp 内广播让分工条件保持一致。
        // WG0 搬 Q/写 O，WG1 搬 KV，WG2 发射指令/调度/索引，WG3 做 softmax。
        uint32_t wg = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        if (wg == 0) {
            // Q/输出线程需要保存较多数据；WG2 让出寄存器，给 WG0 更大的寄存器预算。
            // 184/128/72 是沿用原版的调优预算，并非维度公式的唯一解。
            // 四组预算合计：(184+128+72+128)×128=65536 个 32-bit 寄存器。
            cutlass::arch::warpgroup_reg_alloc<184>();
            run_q_output();
        } else if (wg == 1) {
            cutlass::arch::warpgroup_reg_alloc<128>();
            run_kv();
        } else if (wg == 3) {
            cutlass::arch::warpgroup_reg_alloc<128>();
            run_softmax();
        } else {
            cutlass::arch::warpgroup_reg_dealloc<72>();
            // MMA 和 CLC 每次只需一个线程发射指令；索引则由 warp 10 的全部 32 个线程生成。
            // warp 11 不参与后续任务，所以也不计入 CLC 的确认线程数。
            if (warp_idx == 8 && elect_one_sync())
                run_mma();
            else if (warp_idx == 9 && elect_one_sync())
                run_scheduler();
            else if (warp_idx == 10)
                run_indices();
        }
    }

  private:
    // 初始化所有流水线屏障；warp 3 分配 TMEM。
    __device__ __forceinline__ void initialize() {
        if (warp_idx == 1 && elect_one_sync()) {
#pragma unroll

            for (uint32_t i = 0; i < NUM_KV_SLOTS; ++i) {
                // KV full 等 128 个反量化线程和 1 个 MMA 发射线程；empty 由 SV 完成后通知。
                smem.bar_kv_slot_full[i].init(129);
                smem.bar_kv_slot_empty[i].init(1);
            }
#pragma unroll
            for (uint32_t i = 0; i < NUM_INDICES_BUFS; ++i) {
                // 32 个索引线程生产；128 个 KV 线程和 128 个 softmax 线程都读完后才可复用。
                smem.bar_indices_full[i].init(32);
                smem.bar_indices_empty[i].init(256);
            }
#pragma unroll
            for (uint32_t i = 0; i < NUM_P_BUFS; ++i) {
                // 计数 1 对应 MMA 完成通知；发射指令本身并不代表结果已写入 TMEM。
                smem.bar_tP_full[i].init(1);
            }

            smem.bar_clc_full.init(1);
            // 所有 418 个参与线程都收到本次调度结果后，CLC 才能覆盖同一块响应缓冲区。
            smem.bar_clc_empty.init(NUM_WORKING_THREADS);
            smem.bar_tQ_full.init(128);
            // 除最后一次 QK 完成外，还需 softmax 的 128 次确认；两方都到齐后才能覆盖旧 Q。
            smem.bar_tQ_empty.init(1 + 128);

            smem.bar_tO_full.init(1);
            smem.bar_tO_empty.init(128);
            // SO 共同保护 S 和 O：softmax 要写 S，也可能缩放旧 O；SV 会读 S 并累加 O。
            smem.bar_SO_full.init(128);
            smem.bar_SO_empty.init(1);
            smem.bar_li_mi_full.init(128);
            smem.bar_li_mi_empty.init(128);
            smem.bar_raw_kv_full.init(1);
            // 让异步硬件看到已初始化的屏障，再通过 CTA 同步让其它线程开始使用。
            fence_barrier_init();
        } else if (warp_idx == 3) {
            // 本核独占这份 512 列 TMEM 分配，后续按 tmem_cols 的固定列偏移访问。
            AllocatorT().allocate(512, smem.tmem_start_addr.data());
            AllocatorT().release_allocation_lock();
        }

        __syncthreads();
    }

    __device__ __forceinline__ Job make_job(uint32_t phase, uint32_t query) {
        // 没有逐 Q 长度数组时使用统一 topk；有数组时只屏蔽无效前缀之外的位置。
        // 例如 topk=72、extra_length=121，总跨度 193，块数=(193+63)/64=4。
        // 四块分别是：64 主；8 主+56 额外；64 额外；1 额外+63 无效。
        // 即使 main_length 只有 60，额外部分仍从位置 72 开始，中间 12 个位置被屏蔽。
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
            // 总跨度是主容量加额外有效长度，向上取整让不足 64 行的尾块也执行并受掩码保护。
            ku::ceil_div(
                uint32_t(params.topk) + extra_length, uint32_t(B_TOPK)
            ),
            extra_length,
            uint32_t(params.topk)
        };
    }

    // 等 CLC 返回下个任务，所有参与线程都要确认收到。
    __device__ __forceinline__ Job get_next_job(const Job &cur_args) {
        smem.bar_clc_full.wait(cur_args.job_idx_mod_2);
        ku::CLCResult next_cta0_idx =
            ku::get_clc_query_response<true>(smem.clc_response_obj);
        smem.bar_clc_empty.arrive();

        if (!next_cta0_idx.is_valid) {
            return Job{};
        } else {
            // 每完成一个任务翻转 phase，避免把同一屏障上一轮的完成状态误认成这一轮。
            return make_job(cur_args.job_idx_mod_2 ^ 1, next_cta0_idx.x);
        }
    }

    // 主缓存和额外缓存的交界处可能落在同一个 tile 内。
    __device__ __forceinline__ void for_each_kv_tile(
        const Job &cur_args, auto callable
    ) {
        // 先走纯 FP8 块，再走至多一个混合块，最后走纯 FP4 块。
        // LOC 是模板参数，纯格式路径可在编译期去掉另一种格式的分支。
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

    // Q 搬入寄存器，最后 64 维做 RoPE，再写入 TMEM。
    __device__ __forceinline__ void load_q(const Job &cur_job) {
        static constexpr uint32_t NUM_CACHED_BF16_PER_THREAD =
            H_Q_PER_CTA * D_QK / 128;
        static constexpr uint32_t NUM_BF16_PER_LOAD = 256 / 16;
        // 64×512 个 BF16 分给 128 个线程，每线程持有 256 个值；一次 256-bit 加载取 16 个。
        bf16 cached_q[NUM_CACHED_BF16_PER_THREAD];

        // Q 序号用于定位输入，position 用于查 RoPE 表，两者不必相同；每 lane 缓存一对系数。
        // 64 个 RoPE 维度两两成对，需 64/2=32 个 cos 和 32 个 sin；恰好每 lane 缓存一对。
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

        static constexpr uint32_t TILE_SIZE = 64;
#pragma unroll
        // 倒序先处理含 RoPE 的最后一块；unsigned 从 0 减一变为 0xFFFFFFFF，循环因此退出。
        for (uint32_t local_tile_idx = 3; local_tile_idx != 0xFFFFFFFF;
             --local_tile_idx) {
            // 同一 head 由相隔 64 个线程的两条 lane 分担：warp 0/1 取偶数块，warp 2/3 取奇数块。
            // 每块 64 维，各自取 4 块，形成两条 256 维的部分点积。
            // 例如同为 head 5 的组内线程 5 和 69：前者取维度块 0/2/4/6，后者取 1/3/5/7。
            // 每线程 4 块×64 值=256 BF16；每块 64/16=4 次加载，共 16 次 32 B 加载。
            uint32_t tile_idx = local_tile_idx * 2 + warp_idx / 2;
#pragma unroll
            for (uint32_t i = 0; i < TILE_SIZE / NUM_BF16_PER_LOAD; ++i) {
                uint32_t h_q_idx = idx_in_warpgroup % H_Q_PER_CTA;
                uint32_t d_q_idx = tile_idx * TILE_SIZE + i * NUM_BF16_PER_LOAD;
                // Q 的物理布局是 [维度块, head, 16]，不是普通 [head,512]；因此维度偏移要乘 H_Q。
                // 线性下标=(维度块号×64+head)×16+块内维度；此处 d_q_idx 已按 16 对齐，
                // 可化为 d_q_idx×64+head×16。例：Q0、head=11、维度160，偏移=160×64+11×16=10416 个 BF16。
                // 宏名中的 256 是本次加载的 bit 数，即 32 B；参数 "256B" 是独立的缓存提示。
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

            if (local_tile_idx == 3 && tile_idx == D_VO / TILE_SIZE - 1) {
#pragma unroll

                for (uint32_t j = 0; j < TILE_SIZE; j += 2) {
                    float2 x = __bfloat1622float2(
                        *(nv_bfloat162 *)(cached_q +
                                          local_tile_idx * TILE_SIZE + j)
                    );

                    // RoPE 系数已分散在 32 个 lane 中；shuffle 取得当前维度对的系数，避免重复访存。
                    float cur_cos =
                        __shfl_sync(0xFFFFFFFF, cached_cos[1], j / 2);
                    float cur_sin =
                        __shfl_sync(0xFFFFFFFF, cached_sin[1], j / 2);
                    float2 y = apply_rope(x, cur_cos, cur_sin);
                    *(nv_bfloat162 *)(cached_q + local_tile_idx * TILE_SIZE + j
                    ) = nv_bfloat162{
                        __float2bfloat16_rn(y.x), __float2bfloat16_rn(y.y)
                    };
                }
            }
        }

        // 先把下个 Q 读入寄存器，直到旧 Q 可释放才写 TMEM，从而隐藏全局内存加载延迟。
        // 生产者等待 empty 的上一 phase；消费者等待 full 的当前 phase。
        smem.bar_tQ_empty.wait(cur_job.job_idx_mod_2 ^ 1);
        ku::tcgen05_after_thread_sync();

        // 每线程 256 BF16×2 B=512 B=128 个 uint32；两次各写 64 列，起始列分别是 256 和 320。
        static constexpr uint32_t NUM_CACHED_UINT32 =
            NUM_CACHED_BF16_PER_THREAD / 2;
        // 一个 32-bit TMEM 单元装两个 BF16；分两次写入，限制单条写指令需要的寄存器数量。
        ku::tmem_st_32dp32bNx<NUM_CACHED_UINT32 / 2>(tmem_cols::Q, cached_q);
        ku::tmem_st_32dp32bNx<NUM_CACHED_UINT32 / 2>(
            tmem_cols::Q + NUM_CACHED_UINT32 / 2,
            cached_q + NUM_CACHED_UINT32 / 2 * 2
        );

        // 先等本线程的 TMEM 写完成，再通过屏障交给异步 MMA，不能只靠普通线程同步。
        cutlass::arch::fence_view_async_tmem_store();

        ku::tcgen05_before_thread_sync();
        smem.bar_tQ_full.arrive();
    }

    // scale 向上取整为 2 的幂，输出 E4M3 数值与 UE8M0 指数。
    __device__ __forceinline__ uint8_t
    quantize_output(float *output, float output_scale, fp8_e4m3 *packed) {
        float output_abs_max =
            get_max<O_QUANT_TILE_SIZE, true>(output) * output_scale;

        output_abs_max = max(O_QUANT_CLAMP_MIN_VALUE, output_abs_max);
        // E4M3 最大有限值是 448；先求覆盖本组幅值所需的 scale，再向上取整到 2 的幂。
        float sf = output_abs_max / 448.0f;
        uint32_t sf_as_uint32 = *reinterpret_cast<uint32_t *>(&sf);
        // 利用浮点指数位求 ceil(log2(sf))；减一使 sf 恰好为 2 的幂时不会多升一级。
        // float 有 23 位尾数、指数偏置 127；这里求得 e=ceil(log2(sf))，逆 scale 为 2^(-e)。
        // 例如 amax=100：sf=100/448≈0.2232，e=-2，实际 scale=0.25、逆 scale=4，存储指数 e+127=125。
        uint32_t exp_sf = (int32_t)((sf_as_uint32 - 1) >> 23) + (1 - 127);
        uint32_t sf_inv_as_uint32 = (127 - exp_sf) << 23;
        float sf_inv = *reinterpret_cast<float *>(&sf_inv_as_uint32);
        // 将 softmax 归一化和量化除 scale 合成一次乘法，避免多遍处理这 32 个输出。
        float cur_multiplier = output_scale * sf_inv;
        float2 cur_multiplier_float2 = float2(cur_multiplier, cur_multiplier);
#pragma unroll
        for (uint32_t j = 0; j < O_QUANT_TILE_SIZE; j += 2) {
            float2 x = *(float2 *)(output + j);
            x = ku::float2_mul(x, cur_multiplier_float2);
            *(__nv_fp8x2_storage_t *)(packed + j) = __nv_cvt_float2_to_fp8x2(
                x, __NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3
            );
        }
        return exp_sf + 127;
    }

    // 输出按 [分组, 维度块, 组内 head, 32] 排列，scale 每四个打包成 uint32。
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
             mma_atom_idx += 1) {
            static constexpr uint32_t OUTPUT_SAVE_AS_SCALE_GRAN = 32;

            // 每个 MMA atom 覆盖 256 维，两组对应同一 head 的线程各负责其中 128 维。
            // head 的两组线程分别取每个 atom 的前/后 128 维：atom0 起点为 0/128，atom1 为 256/384。
            uint32_t head_dim_idx_base =
                mma_atom_idx * MMA_ATOM_N +
                (warp_idx / 2) * (MMA_ATOM_N / FOLD_FACTOR);
#pragma unroll
            for (uint32_t i = 0;
                 i < NUM_O_TMEM_COLS_PER_ATOM / OUTPUT_SAVE_AS_SCALE_GRAN;
                 ++i) {
                // 每 head 有 512/32=16 个 scale；每组 8 head，因此一个维度块对应连续 8 个 scale 字节。
                // 例：head11 属于组1、组内3；维度160 属于块5，sf_block_idx=5×8+3=43，位于第10个字的字节3。
                uint32_t sf_block_idx =
                    head_idx_in_wv_group +
                    (head_dim_idx_base / OUTPUT_SAVE_AS_SCALE_GRAN + i) *
                        WV_GROUP_SIZE;
                // sf_block_idx/4 选择 uint32 字，%4 选择其中一个字节；不同线程写各自的指数。
                *((uint8_t *)(params.out_sf + cur_job.s_q_idx +
                              wv_group_idx * params.stride_out_sf_wv_group +
                              (sf_block_idx / 4) * params.stride_out_sf_head_dim
                  ) +
                  sf_block_idx % 4) = output_sf[mma_atom_idx][i];
            }
#pragma unroll
            for (uint32_t i = 0; i < NUM_O_TMEM_COLS_PER_ATOM; i += 32) {
                uint32_t head_dim_idx = head_dim_idx_base + i;
                // 连续写 32 个 FP8；地址与 [N,8组,16维度块,8个head,32值] 的物理布局对应。
                // 以 Q2、head11、维度160 为例：2×32768+1×4096+3×32+160×8=71008 字节。
                // 各项依次跨过 Q、8-head 分组、组内 head、维度块；一次写入该块的 32 个 FP8。
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

    // 合并 softmax 分母，做逆 RoPE，并按每 32 个值量化输出。
    __device__ __forceinline__ void store_o(
        const Job &cur_job, bool is_last_job
    ) {
        smem.bar_li_mi_full.wait(cur_job.job_idx_mod_2);
        float li = 0.0f;
        float mi = smem.rowwise_mi_buf[idx_in_warpgroup % H_Q_PER_CTA];

        // 同一 head 的 64 个分数分成两半；线程号 xor 64 找到另一半，将两个分母部分和相加。
        // 线程5和69都对应 head5，各保存 32 个 token 的权重和；5 xor 64=69，合并后覆盖全部64个。
        li = smem.rowwise_li_buf[idx_in_warpgroup] +
             smem.rowwise_li_buf[idx_in_warpgroup ^ 64];

        smem.bar_li_mi_empty.arrive();

        if (idx_in_warpgroup < H_Q_PER_CTA) {
            uint32_t global_index = cur_job.s_q_idx * H_Q + idx_in_warpgroup;
            // 内部用 exp2，故自然对数 LSE=mi*ln(2)+ln(li)；空行按接口约定写 +inf。
            float cur_lse = fmaf(mi, CUDART_LN2_F, logf(li));
            cur_lse = cur_lse == -CUDART_INF_F ? +CUDART_INF_F : cur_lse;
            params.lse[global_index] = cur_lse;
        }

        // sink 相当于只给分母增加一个权重、不贡献 V；当前接口的 LSE 不包含它。
        float attn_sink =
            params.attn_sink == nullptr
                ? -CUDART_INF_F
                : __ldg(params.attn_sink + idx_in_warpgroup % H_Q_PER_CTA) *
                      CUDART_L2E_F;
        float output_scale =
            li == 0.0f ? 0.0f : __fdividef(1.0f, li + exp2f(attn_sink - mi));

        smem.bar_tO_full.wait(cur_job.job_idx_mod_2);
        ku::tcgen05_after_thread_sync();

        // 512 输出维度/256 维每atom=2 atom；每atom 的 64×256 个 float 分给128线程，每线程128个。
        // 按32个值量化一次，故每线程每atom有128/32=4个输出片段，共2×4=8个scale。
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
             mma_atom_idx += 1) {
#pragma unroll

            for (uint32_t epilogue_tile_idx_in_atom = 0;
                 epilogue_tile_idx_in_atom < NUM_EPILOGUE_TILES_PER_ATOM;
                 ++epilogue_tile_idx_in_atom) {
                // 每线程按列读：atom0 对应列0..127，atom1 对应128..255；片段起点依次加0/32/64/96。
                uint32_t tmem_col_base =
                    tmem_cols::O + mma_atom_idx * NUM_O_TMEM_COLS_PER_ATOM +
                    epilogue_tile_idx_in_atom * EPILOGUE_TILE_SIZE;
                float output[EPILOGUE_TILE_SIZE];

                ku::tmem_ld_32dp32bNx<EPILOGUE_TILE_SIZE>(
                    tmem_col_base, output
                );
                cutlass::arch::fence_view_async_tmem_load();

                // 最后一片 O 已读到寄存器，即可让下个任务重用 TMEM；量化和全局写回仍可继续。
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
                    // 只有最后 64 维需要逆 RoPE：它们位于第二个 MMA atom 的末尾，由 warp 2/3 持有。
                    // 第二个atom、后半线程的维度起点为256+128=384；片段2/3起点是448/480，正好覆盖末尾64维。
                    should_perform_rope = mma_atom_idx + 1 == NUM_MMA_ATOMS &&
                                          epilogue_tile_idx_in_atom >=
                                              NUM_EPILOGUE_TILES_PER_ATOM -
                                                  D_ROPE / EPILOGUE_TILE_SIZE &&
                                          warp_idx >= 2;
                    if (should_perform_rope) {
#pragma unroll

                        for (uint32_t j = 0; j < EPILOGUE_TILE_SIZE; j += 2) {
                            float2 x = *(float2 *)(output + j);
                            // 每片段32个值=16对：片段2取系数lane0..15，片段3取lane16..31，所以后者额外加32/2=16。
                            uint32_t src_lane =
                                j / 2 + (epilogue_tile_idx_in_atom + 1 ==
                                                 NUM_EPILOGUE_TILES_PER_ATOM
                                             ? EPILOGUE_TILE_SIZE / 2
                                             : 0);
                            float cur_cos = __shfl_sync(
                                0xFFFFFFFF, cached_cos[0], src_lane
                            );
                            float cur_sin = __shfl_sync(
                                0xFFFFFFFF, cached_sin[0], src_lane
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

    // WG0：Q0 → Q1 → O0 → Q2 → O1，预取下个 Q 与当前输出重叠。
    __device__ __forceinline__ void run_q_output() {
        Job cur_job = make_job(0, blockIdx.x);
        load_q(cur_job);
        do {
            Job next_job = get_next_job(cur_job);
            cached_cos[0] = cached_cos[1];
            // 先保存旧 Q 的系数并反转 sin；即使下面加载新 Q，当前 O 仍能使用正确的逆旋转。
            cached_sin[0] = -cached_sin[1];
            if (next_job.is_valid) {
                load_q(next_job);
            }
            store_o(cur_job, !next_job.is_valid);
            cur_job = next_job;
        } while (cur_job.is_valid);

        // WG0 全部读完最后一个 O 后，才由 warp 0 释放 TMEM 分配。
        NamedBarrier::arrive_and_wait(128, barrier_ids::WG0_SYNC);
        if (warp_idx == 0) {
            AllocatorT().free(0, 512);
        }
    }

    // WG3：在线 softmax；最大值变化时同步缩放已经累积的 O。
    __device__ __forceinline__ void run_softmax() {
        Job cur_job = make_job(0, blockIdx.x);
        uint32_t local_warp_idx = warp_idx - 12;

        // 每线程负责一个 head 的 32 个 token 权重；组内 warp 0/1 和 2/3 分担前后半。
        // S 按 MMA 可直接读取的 K-major 布局写入，地址不是简单的 row*64+col。
        // 同一head的两半在S中相隔64×(64/2)=2048个BF16；每个head的首段占8个BF16。
        // 例如组内线程5的基址偏移=5×8=40，线程69为2048+40=2088；后续每段再跨8×64=512个BF16。
        bf16 *sS_base = smem.s +
                        (local_warp_idx >= 2 ? H_Q_PER_CTA * (B_TOPK / 2) : 0) +
                        (idx_in_warpgroup % H_Q_PER_CTA) * 8;
        // rs 记录已处理块数：槽号=块数%槽数，phase=(块数/槽数)&1。
        // 它放在任务循环外，跨 Q 持续推进；屏障并未重置，因此不能每个 Q 从零计数。
        // 以3个KV槽为例：块0/1/2用槽0/1/2、phase0；块3/4/5复用同样槽、phase1；块6回到槽0、phase0。
        RingBufferState rs;
        do {
            // 合并两路部分点积后，64 head×64 token/128线程=每线程32个完整分数。
            static constexpr uint32_t NUM_ELEMS_PER_THREAD =
                B_TOPK * H_Q_PER_CTA / 128;
            // mi 是当前指数归一化基准；li 是本线程负责的半行权重和；O 保存对应加权分子。
            // real_mi 单独记录实际最大分数，用来识别整行都无有效索引的情况。
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
                auto [p_buf_idx, p_bar_phase] = rs.get<NUM_P_BUFS>();
                smem.bar_tP_full[p_buf_idx].wait(p_bar_phase);
                smem.bar_indices_full[indices_buf_idx].wait(indices_bar_phase);
                ku::tcgen05_after_thread_sync();

                float p[NUM_ELEMS_PER_THREAD];
                // QK 分成两条 256 维部分点积；这里交换并相加得到完整 512 维分数。
                // 无效索引置为 -inf，使其指数权重为零，不能仅靠把 KV 清零来忽略它。
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
                // 两个对应 warp 共 64 线程，交换同一 head 前后半行的最大值，不必同步整个 CTA。
                NamedBarrier::arrive_and_wait(
                    64, barrier_ids::WG3_WARP02_SYNC + (local_warp_idx & 1)
                );
                smem.bar_indices_empty[indices_buf_idx].arrive();

                cur_pi_max =
                    max(cur_pi_max,
                        smem.rowwise_max_buf[idx_in_warpgroup ^ 64]);
                real_mi = max(real_mi, cur_pi_max);
                // 沿用原版的延迟重缩放：最大值超过旧基准 6 个 log2 单位时才调整，减少 TMEM 操作。
                // 所以 mi 不一定等于 real_mi；__any_sync 让同一 warp 一致决定是否缩放。
                // 这里的6以log2计量，对应指数权重比2^6=64；它是延迟缩放阈值，不是6个token。
                bool should_scale_o =
                    __any_sync(0xffffffff, cur_pi_max - mi > 6.0f);

                float new_max, scale_for_old;
                if (!should_scale_o) {
                    scale_for_old = 1.0f;
                    new_max = mi;
                } else {
                    new_max = max(cur_pi_max, mi);
                    // 基准从 mi 改为 new_max 时，旧的分子 O 和分母 li 都必须乘 2^(mi-new_max)。
                    // 例：基准10升到17，旧项乘2^(10-17)=1/128；旧li=128、本块和=3，则新li=128/128+3=4。
                    // 旧O也同比缩放，例如256变2，保证分子与分母始终使用同一归一化基准。
                    scale_for_old = exp2f(mi - new_max);
                }
                mi = new_max;

                nv_bfloat16 s[NUM_ELEMS_PER_THREAD];
                float cur_sum = get_s_from_p<NUM_ELEMS_PER_THREAD>(
                    (nv_bfloat162 *)s, p, score_multiplier, new_max
                );
                // online softmax 递推：旧分母换到新基准，再加本块权重和；不需要保存整行分数。
                li = fmaf(li, scale_for_old, cur_sum);

                // 等上一块 SV 用完 S 和 O，才能写新 S、缩放旧 O，防止与异步 MMA 同时访问。
                smem.bar_SO_empty.wait(rs.get<1>().second ^ 1);
#pragma unroll
                // 一条128-bit写入容纳16 B/2 B=8个BF16；每线程32个权重，需32/8=4次写入。
                for (int i = 0; i < NUM_ELEMS_PER_THREAD / 8; ++i) {
                    ku::st_shared(
                        sS_base + i * 8 * H_Q_PER_CTA,
                        *(__int128_t *)(s + i * 8)
                    );
                }

                // 第一块 SV 会清空累加器，所以第一块无需缩放旧 O；后续块必须与 li 同步缩放。
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

            // 整行无效时强制分母为零；输出阶段据此写零输出，并采用接口约定的空行 LSE。
            if (real_mi == -CUDART_INF_F) {
                li = 0.0f;
                mi = -CUDART_INF_F;
            }

            // 一份 mi 足够，但两半的 li 都要保留；等 WG0 读完上一任务后才能覆盖。
            smem.bar_li_mi_empty.wait(cur_job.job_idx_mod_2 ^ 1);

            if (local_warp_idx < H_Q_PER_CTA / 32) {
                smem.rowwise_mi_buf[idx_in_warpgroup] = mi;
            }
            smem.rowwise_li_buf[idx_in_warpgroup] = li;
            smem.bar_li_mi_full.arrive();

            cur_job = get_next_job(cur_job);
        } while (cur_job.is_valid);
    }

    // QK 使用 TMEM 中的 Q 和 shared memory 中的 K。
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
        // CuTe Tensor 在这里是地址+布局视图，不是重新分配内存；K 被解释成两条部分点积的布局。
        // K原本为64 token×512维，重解释为(64×2)×(512/2)=128×256，元素总数仍是32768。
        // 所以QK产生64 head×128个部分分数；两路相加后才得到64×64个完整分数。
        Tensor sK = make_tensor(
            make_smem_ptr(smem.kv_slots[kv_slot_idx]),
            ku::make_umma_canonical_k_major_layout<
                B_TOPK * NUM_MRGEMM_RAILS,
                D_QK / NUM_MRGEMM_RAILS,
                128>()
        );

        // 补足初始化时的第 129 次 arrive；此配置没有另行搬运的 BF16 KV 部分。
        smem.bar_kv_slot_full[kv_slot_idx].arrive();

        smem.bar_kv_slot_full[kv_slot_idx].wait(kv_bar_phase);
        auto [p_buf_idx, p_bar_phase] = rs.get<NUM_P_BUFS>();

        // 两份 P 交替使用。QK[i] 能发射时，SV[i-2] 已被发射，旧 P[i-2] 已被 softmax 消费，
        // 所以当前调度下不需要额外的 P-empty 屏障；改动发射顺序时必须重新检查此关系。
        tP.data().get() = tmem_cols::get_p(p_buf_idx);

        ku::tcgen05_after_thread_sync();
        auto mma = TiledMMA_QK{};
        // true 表示本次 QK 从零开始；TS 表示 Q 在 TMEM、K 在 shared memory。
        ku::utcmma_ts(mma, tQ, sK, tP, true);
        // 这是异步 MMA 完成后才触发的通知，不能改成发射线程立即调用普通 arrive。
        ku::umma_arrive_noelect(smem.bar_tP_full[p_buf_idx]);

        if (kv_block_idx == job.num_kv_blocks - 1) {
            ku::umma_arrive_noelect(smem.bar_tQ_empty);
        }
        rs.update();
    }

    // SV 完成后释放 KV 槽和 S 缓冲区。
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
        // K 和 V 复用同一份解码数据；这里换成 MN-major 视图，让 S×V 按需要的方向读取。
        Tensor sV = make_tensor(
            make_smem_ptr(smem.kv_slots[kv_slot_idx]),
            ku::make_umma_canonical_mn_major_layout<D_VO, B_TOPK, 128>()
        );
        ku::tcgen05_after_thread_sync();
        auto mma = TiledMMA_SV{};
        // SS 表示两输入都在 shared memory；第一块清零 O，其余块在同一个 TMEM 累加器上累加。
        ku::utcmma_ss(mma, sS, sV, tO, kv_block_idx == 0);
        // QK 用完还不能释放 KV，因为后面的 SV 还要读同一块 V；须等 SV 完成。
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

        // QK 比 SV 提前一块，因此各自保存槽号/phase，不能用一个同步更新的计数器。
        RingBufferState rs_qk, rs_sv;

        Job cur_job = make_job(0, blockIdx.x);
        issue_qk(cur_job, 0, rs_qk, tQ, tP);
        do {
#pragma unroll 1

            for (uint32_t kv_block_idx = 1;
                 kv_block_idx < cur_job.num_kv_blocks;
                 ++kv_block_idx) {
                // 先算下一块分数，再消费上一块 softmax 权重，让 softmax 和矩阵计算有重叠机会。
                issue_qk(cur_job, kv_block_idx, rs_qk, tQ, tP);
                issue_sv(cur_job, kv_block_idx - 1, rs_sv, sS, tO);
            }

            Job next_job = get_next_job(cur_job);
            if (next_job.is_valid) {
                // 提前启动下一个 Q 的首块 QK；它不写 O，仍可接着完成当前 Q 的最后一次 SV。
                issue_qk(next_job, 0, rs_qk, tQ, tP);
            }
            issue_sv(cur_job, cur_job.num_kv_blocks - 1, rs_sv, sS, tO);

            cur_job = next_job;
        } while (cur_job.is_valid);
    }

    // warp 9 的一个线程用 CLC 领取尚未执行的 CTA 任务。
    __device__ __forceinline__ void run_scheduler() {
        bool phase = 0;
        while (true) {
            // 先等所有流水线读取上次响应，再请求取消一个尚未启动的 CTA，并由当前 CTA 接手其 Q。
            smem.bar_clc_empty.wait(phase ^ 1);
            ku::issue_clc_query_multicast_cluster_all(
                smem.bar_clc_full, smem.clc_response_obj
            );

            // 同时登记线程到达和响应字节数；wait 要等异步响应真正写完，而非仅等请求已发出。
            smem.bar_clc_full.arrive_and_expect_tx(sizeof(smem.clc_response_obj)
            );

            smem.bar_clc_full.wait(phase & 1);
            ku::CLCResult clc_result =
                ku::get_clc_query_response<true>(smem.clc_response_obj);
            smem.bar_clc_empty.arrive();
            if (!clc_result.is_valid)
                break;

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

        auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
        smem.bar_indices_empty[indices_buf_idx].wait(indices_bar_phase ^ 1);
#pragma unroll
        for (uint32_t i = 0; i < NUM_INDICES_PER_THREAD; ++i) {
            // pos 是拼接索引序列中的位置；一个 warp 每线程取两个索引，合计覆盖 64 行。
            uint32_t pos = kv_block_idx * B_TOPK + i * 32 + lane_idx;
            bool in_extra;
            if constexpr (LOC == KVLocation::ORIG) {
                in_extra = false;
            } else if constexpr (LOC == KVLocation::EXTRA) {
                in_extra = true;
            } else {
                in_extra = pos >= cur_job.num_orig_slots;
            }
            // local_pos 是各自索引数组内的位置，cur_index 才是读取出来的物理 token 槽号。
            uint32_t local_pos = in_extra ? pos - cur_job.num_orig_slots : pos;
            uint32_t valid_len =
                in_extra ? cur_job.extra_topk_length : cur_job.topk_length;

            // 尾块和短有效长度可能越过索引行，必须先判断再 __ldg，不能先读出再屏蔽。
            int cur_index =
                local_pos < valid_len
                    ? __ldg(
                          (in_extra ? extra_indices_base : indices_base) +
                          local_pos
                      )
                    : -1;
            // 无效物理位置既不能参加 softmax，也不能用来读取 scale。
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
            // 无效索引用物理槽 0 计算安全地址，随后将坐标/scale 屏蔽，避免任何越界 scale 读取。
            uint32_t token_idx = is_index_valid ? (uint32_t)cur_index : 0;
            int idx_in_block;
            // 商 block_idx 是物理页号，余数 idx_in_block 是页内行号；这里的 block 不是 CUDA CTA。
            // 例：page_size=5、物理token_idx=13，则页号=13/5=2、页内行=13%5=3。
            int block_idx = fast_divmod.divmod(idx_in_block, (int)token_idx);
            uint32_t row = i * 32 + lane_idx;

            uint32_t tma_k_stride = in_extra ? ExtraKVFormat::TMA_K_STRIDE
                                             : OrigKVFormat::TMA_K_STRIDE;
            uint32_t num_scales_each_token =
                in_extra ? ExtraKVFormat::NUM_SCALES_EACH_TOKEN
                         : OrigKVFormat::NUM_SCALES_EACH_TOKEN;
            // TMA 把缓存看成等行距二维数组；把页字节步长换成虚拟行数，再加页内行号。
            // 坐标 -1 让无效数据走 TMA 越界补零，而 softmax 另用有效位屏蔽分数。
            // 若FP8页步长为3072 B、数据行距512 B，则每页占6个TMA虚拟行；物理页2行3对应坐标2×6+3=15。
            // 物理token槽号13与TMA行号15不同：后者还计入页内scale区和padding占据的地址。
            smem.decode_tma_coords[indices_buf_idx][row] =
                is_index_valid ? (src_block_stride / tma_k_stride) * block_idx +
                                     idx_in_block
                               : -1;

            uint32_t page_block_size = in_extra ? params.extra_page_block_size
                                                : params.page_block_size;
            uint8_t *kv_base =
                (uint8_t *)(in_extra ? params.extra_kv : params.kv);
            // 每页采用平面存储：先连续放全部压缩数据，再连续放每 token 的 scale，最后才是页尾 padding。
            // 同例每页5行：scale偏移=2×3072+5×512+3×16=8752 B，依次是页起点、数据区大小、scale行偏移。
            uint8_t *scale_ptr = kv_base + page_block_size * tma_k_stride +
                                 (int64_t)block_idx * src_block_stride +
                                 idx_in_block * num_scales_each_token;
            bool is_fp4_token = in_extra;

            // FP8 每 token 只有 16 字节 scale，先共享给 WG1；FP4 有 32 字节，只传指针节省 shared memory。
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
            // 将 32 个线程的有效标志压成一个位图；做两轮得到这个 KV 块的 64 个有效位。
            // 每轮32个bool压成32 bit=4 B，两轮共8 B，恰好表示64个token；第j位对应本轮lane j。
            uint32_t mask = __ballot_sync(0xFFFFFFFF, is_index_valid);
            if (lane_idx == 0) {
                *((uint32_t *)smem.is_k_valid[indices_buf_idx] + i) = mask;
            }
        }

        smem.bar_indices_full[indices_buf_idx].arrive();
        rs.update();
    }

    // warp 10：生成物理坐标、有效位和量化 scale，交给 KV 流水线。
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
        // token_idx 在此仅为线程自己的第 0/1 行；转换后才是当前 64 行 KV 块中的行号。
        // 例：WG1局部warp1、lane13，有row_in_wavefront=13%8=5，idx_in_row=13/8=1。
        // 它参与的两行是1×16+0×8+5=21和1×16+1×8+5=29；每行只负责四分之一的chunk。
        return local_warp_idx * NUM_ROWS_PER_WARP +
               token_idx * NUM_ROWS_PER_WAVEFRONT + row_in_wavefront;
    }

    template <typename F>
    // 相邻行轮换 chunk 顺序，避免 shared memory bank 冲突。
    __device__ __forceinline__ uint32_t get_chunk_base(uint32_t half) {
        constexpr uint32_t NUM_CHUNKS_PER_LANE =
            F::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW;

        // FP4 每 lane 4 个 chunk，直接轮换；FP8 每 lane 8 个 chunk，交换两个 4-chunk 半区。
        // 配合原始行的 16 字节 padding，让同一次 128-bit shared load 的各 lane 避开 bank 冲突。
        // 接上例：行内编号1、rotation=(5/4)×4=4。FP8两个chunk起点是1×8+(0 xor 4)=12、
        // 1×8+(4 xor 4)=8，访问12..15后再访问8..11；FP4起点=(1×4+4)%16=8，访问8..11。
        if constexpr (NUM_CHUNKS_PER_LANE <= 4) {
            return (idx_in_row * NUM_CHUNKS_PER_LANE + chunk_rotation) %
                   F::NUM_CHUNKS_PER_ROW;
        } else {
            return idx_in_row * NUM_CHUNKS_PER_LANE +
                   (half * 4 ^ chunk_rotation);
        }
    }

    __device__ __forceinline__ uint32_t get_raw_row_offset(uint32_t row) {
        // gather4 一次搬四行，每组起点必须 128 字节对齐；组内行距要由具体格式另行加上。
        // 例：row=21属于四行组5、组内行1；组起点=5×2176=10880 B。
        // 加组内行距后，FP8行起点为10880+528=11408 B，FP4为10880+272=11152 B。
        return row / 4 * RAW_KV_GROUP_BYTES;
    }

    template <uint32_t NUM_FP8_TOKENS>
    __device__ __forceinline__ void for_each_token(auto callable) {
        cute::for_each(
            // 用编译期整数展开两个 token，让每行的 F 固定为 FP8 或 FP4；转换指令和寄存器数组大小随之确定。
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
        // 先定位四行组，再加组内行距；每次 LDS 读取一个 16 字节 chunk 到四个 uint32 寄存器。
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
    // FP8 乘 UE8M0 scale；FP4 乘 E4M3 scale，再按 MMA 布局写回。
    __device__ __forceinline__ void dequantize_token(
        uint32_t token_idx,
        uint8_t *slot_base,
        const uint32_t (&cached_input
        )[NUM_TOKENS_PER_THREAD][MAX_CHUNKS_PER_LANE][4],
        const uint32_t (&cached_scales)[NUM_TOKENS_PER_THREAD][2]
    ) {
        // C_LANE 是本 lane 的 chunk 数；NUM_STS_PER_CHUNK 是展开后需要几次 16 字节写回。
        // 16 字节压缩数据展开为 FP8 的 16 个 BF16 或 FP4 的 32 个 BF16，因此分别写 2/4 次。
        // FP8每chunk：16个值×2 B=32 B，需2次16 B写回；FP4：32×2 B=64 B，需4次。
        // 每lane每行都展开为256 B：FP8是8×32，FP4是4×64；四个lane合计1024 B=512 BF16。
        static constexpr uint32_t C_LANE =
            F::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW;
        static constexpr uint32_t NUM_STS_PER_CHUNK =
            F::CHUNK_ELEMS * sizeof(bf16) / 16;
        ku::nvbf16x2 fp4_scales[2];
#pragma unroll
        for (uint32_t g = 0; g < C_LANE; ++g) {
            ku::nvbf16x2 data_bf16x2[F::CHUNK_ELEMS / 2];
            if constexpr (F::IS_FP4) {
                // 一个 FP4 chunk 覆盖 32 个数，需要两个 per-16 scale；一次解码四个 scale 可供相邻两个 chunk 使用。
                if (g % 2 == 0) {
                    // 一个uint32装4个E4M3 scale，解出2个BF16二元组；每chunk取其中一组中的两个scale。
                    fp8x4_to_bf16x2x2(
                        cached_scales[token_idx][g / 2], fp4_scales
                    );
                }
#pragma unroll
                for (uint32_t j = 0; j < 4; ++j) {
                    // 一个uint32有32 bit，可装8个4-bit值；展开后为8个BF16，即4个BF16二元组。
                    fp4x8_to_bf16x2x4(
                        cached_input[token_idx][g][j], data_bf16x2 + j * 4
                    );
                }
                // low/high 分别是当前 chunk 前后 16 个数的 scale，复制成 BF16 二元组后并行相乘。
                ku::nvbf16x2 scale_lo = __low2bfloat162(fp4_scales[g % 2]),
                             scale_hi = __high2bfloat162(fp4_scales[g % 2]);
#pragma unroll
                for (uint32_t k = 0; k < F::CHUNK_ELEMS / 2; ++k) {
                    data_bf16x2[k] = __hmul2(
                        data_bf16x2[k],
                        k < F::QUANT_TILE_SIZE / 2 ? scale_lo : scale_hi
                    );
                }
            } else {
                // FP8 每 32 个数共用一个 UE8M0 scale，也就是两个连续的 16-value chunk 共用一个指数。
                __nv_fp8_e8m0 scale = ((__nv_fp8_e8m0 *)cached_scales[token_idx]
                )[g * F::CHUNK_ELEMS / F::QUANT_TILE_SIZE];
#pragma unroll
                for (uint32_t k = 0; k < F::CHUNK_ELEMS / 2; ++k) {
                    data_bf16x2[k] = fp8x2_to_bf16x2_with_scale(
                        ((ku::nve4m3x2 *)cached_input[token_idx][g])[k], scale
                    );
                }
            }

            // 一个128 B输出列块容纳：FP8的128/(16×2)=4个chunk，或FP4的128/(32×2)=2个chunk。
            static constexpr uint32_t CHUNKS_PER_ATOM_ROW =
                128 / (F::CHUNK_ELEMS * 2);
            // 反量化结果不能按普通行主序写回：MMA 期望 SW128 布局。
            // atom_col 选择 128 字节列块，sts_offsets 选择块内 swizzle 位置，token 偏移选择对应行。
            // 例：FP4 chunk_base=8、g=1，atom_col=8/2+1/2=4，块内起始片段=(1%2)×4=4。
            // 本线程第二行的写地址偏移为sts_offsets[4+j]+4×8192+1×1024，j=0..3；sts_offsets已含swizzle。
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
        // 先取元数据并发射 TMA，再读取 FP4 scale，重叠两次搬运。
        auto [indices_buf_idx, indices_bar_phase] = rs.get<NUM_INDICES_BUFS>();
        smem.bar_indices_full[indices_buf_idx].wait(indices_bar_phase);

        // FP4每lane负责4个chunk，每chunk2个scale、每scale1 B，故缓存8 B=2个uint32。
        // FP8每lane负责8个chunk，每2个chunk共用1个scale，故仅用4 B=1个uint32。
        static constexpr uint32_t FP4_SCALE_BYTES_PER_LANE =
            ExtraKVFormat::NUM_CHUNKS_PER_ROW / NUM_LANES_PER_ROW * 2;

        // 每线程每行最多缓存 8 字节 scale；FP8 只用前 4 字节，FP4 使用全部 8 字节。
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
                    // FP8 数据若交换了前后四个 chunk，四个 scale 字节也必须交换两半，保持一一对应。
                    chunk_rotation ? __byte_perm(scales, scales, 0x1032)
                                   : scales;
            }
        });

        auto [kv_slot_idx, kv_bar_phase] = rs.get<NUM_KV_SLOTS>();
        smem.bar_kv_slot_empty[kv_slot_idx].wait(kv_bar_phase ^ 1);
        // slot_base 是当前 KV 环形槽的共享内存起点；同一空间先放原始压缩数据，随后原地展开。
        uint8_t *slot_base = (uint8_t *)smem.kv_slots[kv_slot_idx];

        cute::for_each(
            // 每 warp 管 16 行，发四次 gather4；每次由一个 elected lane 搬四个物理位置。
            cute::make_int_sequence<NUM_ROWS_PER_WARP / 4>{},
            [&](auto i) {
                // 每warp覆盖16行、一次gather4覆盖4行，因此每warp发16/4=4次，整个WG共16次gather4。
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
        // 坐标已交给 TMA，scale/指针已进寄存器；WG1 不再依赖这份元数据，允许索引生产者推进。
        smem.bar_indices_empty[indices_buf_idx].arrive();

        for_each_token<NUM_FP8_TOKENS>([&]<typename F>(uint32_t token_idx) {
            if constexpr (F::IS_FP4) {
                uint64_t scale_addr =
                    ((uint64_t)scale_ptrs[token_idx] + get_chunk_base<F>(0) * 2
                    );

                // 无效 FP4 行保留零 scale；memory clobber 约束编译器重排，让这次读取留在 gather4 发射之后。
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

        // 仅一个线程登记整块 TMA 的期望字节数，避免重复计数；包含行尾补零，不含四行组之间的空隙。
        if (idx_in_warpgroup == 0) {
            // 登记字节数=FP8行数×528+FP4行数×272；纯FP8为33792 B，纯FP4为17408 B。
            // 8行FP8+56行FP4的混合块为8×528+56×272=19456 B；不把每四行组之间的空隙算进去。
            smem.bar_raw_kv_full.arrive_and_expect_tx(
                num_orig_rows * OrigKVFormat::RAW_TOKEN_SMEM_STRIDE +
                (B_TOPK - num_orig_rows) * ExtraKVFormat::RAW_TOKEN_SMEM_STRIDE
            );
        }
        smem.bar_raw_kv_full.wait(rs.get<1>().second);

        // 数组上限为2行×8chunk×4个uint32=64个uint32=256 B/线程；FP4只用每行前4个chunk。
        uint32_t cached_input[NUM_TOKENS_PER_THREAD][MAX_CHUNKS_PER_LANE][4];

        for_each_token<NUM_FP8_TOKENS>([&]<typename F>(uint32_t token_idx) {
            load_raw_token<F>(token_idx, slot_base, cached_input);
        });
        // 原地反量化：所有线程先读完原始数据，再覆盖同一块 shared memory。
        // BF16 比压缩输入更大，某线程提前写回可能覆盖其它线程尚未读走的原始字节，因此必须全 WG 同步。
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

    // WG1：三个 KV 槽循环使用，完成 gather4 → 反量化 → 通知 MMA。
    __device__ __forceinline__ void run_kv() {
        {
            Tensor sKV = make_tensor(
                make_smem_ptr(smem.kv_slots[0]),
                ku::make_umma_canonical_k_major_layout<B_TOPK, D_QK, 128>()
            );
#pragma unroll
            // 一行128 B列块分成128/16=8个写片段；每片段16 B=8个BF16，所以CuTe列下标取i×8。
            for (uint32_t i = 0; i < 8; ++i) {
                // 先用 CuTe 布局算出 swizzle 后的地址差，之后各槽布局相同，可以复用这些偏移。
                sts_offsets[i] = (uint32_t)((&sKV(get_row_idx(0), i * 8) -
                                             smem.kv_slots[0]) *
                                            sizeof(bf16));
            }
        }

        Job cur_job = make_job(0, blockIdx.x);
        RingBufferState rs;
        do {
            for_each_kv_tile(cur_job, [&]<KVLocation LOC>(uint32_t tile) {
                if constexpr (LOC == KVLocation::ORIG)
                    process_block<2>(64, rs);
                else if constexpr (LOC == KVLocation::EXTRA)
                    process_block<0>(0, rs);
                else {
                    // rows 是混合块中的 FP8 行数；每 warp 管 16 行，clamp 后除以 8 得到每线程的 FP8 行数。
                    // 主 topk 必须是 8 的倍数，才能让这次格式选择在整个 warp 内一致，合法执行 gather4。
                    uint32_t rows = cur_job.num_orig_slots - tile * B_TOPK;
                    // 若混合块仅前8行是FP8：warp0有clamp(8-0,0,16)/8=1个FP8 token/线程，warp1..3为0。
                    // 若前24行是FP8，则四个warp的count依次为2、1、0、0；它决定模板实例，而不是额外循环次数。
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
};

// 参数通过 kernel 参数区传入；__grid_constant__ 保持只读参数共享，避免取引用时生成线程私有副本。
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

// 这些约束直接来自分块和向量访存：主/额外交界需对齐 8 行，页面需覆盖数据和 scale。
// FP8每token有效字节=512+512/32=528；FP4=512/2+512/16=288。
// 例如page_size=5，两种页步长向上对齐512 B后分别是ceil(2640/512)×512=3072 B、1536 B。
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
    // 上游参数沿用 bf16* 类型，但这里传的是压缩字节地址；真正读取时按 KVFormat 和字节步长解释。
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
    // 每组 8 个 head×每 head 16 个 scale=128 字节=32 个 uint32，所以组间跨 32 个 scale 行。
    // 接head11、维度160的例子：若Q编号2、sf_stride=4，字下标=2+1×(32×4)+10×4=170。
    // 字节3在总缓冲区的偏移为170×4+3=683 B；sf_stride单位是uint32个数，不是字节。
    p.stride_out_sf_wv_group = 32 * a.sf_stride;
    p.stride_out_sf_head_dim = a.sf_stride;

    return p;
}

// 两种缓存只差行宽和步长，TMA 描述符构造共用这一份代码。
template <typename F>
CUtensorMap make_kv_map(void *data, int pages, int64_t stride) {
    // TMA 按 UINT32 描述原始字节，行宽/box 宽度都除以 4，满足 box 维度上限。
    // box 比实际数据行宽 16 字节，越界部分补零；奇数个 16-byte chunk 的行距用于减少 bank 冲突。
    // FP8：全局行宽512/4=128个uint32，box宽(512+16)/4=132；FP4则是64和68。
    // 例：FP8有5页、页步长3072 B，TMA二维尺寸为[128,5×3072/512]=[128,30]，行距512 B。
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
    // shared memory 需求超过默认动态额度，启动前显式申请这份共享内存上限。
    check_cuda(cudaFuncSetAttribute(
        attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes
    ));
    // 初始 grid 为每个 Q 一个 CTA，cluster 固定为 1；已运行 CTA 再通过 CLC 接手未启动任务。
    cutlass::ClusterLaunchParams config{
        dim3(params.s_q), dim3(NUM_THREADS), dim3(1), bytes, params.stream
    };
    auto status = cutlass::launch_kernel_on_cluster(
        config, (void *)attention_kernel, params, maps, divisors
    );
    if (status != cutlass::Status::kSuccess)
        throw std::runtime_error("SM100 kernel launch failed");
    check_cuda(cudaGetLastError());
}
} // namespace mega_attn_v0

extern "C" void solve_v0(const MegaAttnParams &a, cudaStream_t stream) {
    mega_attn_v0::validate_layout(a);
    mega_attn_v0::launch(mega_attn_v0::make_params(a, stream));
}
