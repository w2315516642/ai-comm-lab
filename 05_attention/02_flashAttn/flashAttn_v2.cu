#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <stdint.h>

namespace {

constexpr int MMA_M = 16;
constexpr int MMA_K = 16;
constexpr int MMA_N = 8;

constexpr int WARP_MMA_M = 2;
constexpr int WARP_MMA_N = 2;
constexpr int WARPS_M = 4;

constexpr int WARP_M = WARP_MMA_M * MMA_M;
constexpr int WARP_N = WARP_MMA_N * MMA_N;

constexpr int BM = WARPS_M * WARP_M;
constexpr int BN = WARP_N;
constexpr int BK = 32;

constexpr int WARP_SIZE = 32;
constexpr int THREADS = WARPS_M * WARP_SIZE;

static_assert(BN % MMA_K == 0, "BN must be a multiple of the PV MMA K size");
static_assert(
    WARP_MMA_N % 2 == 0,
    "two score m16n8 fragments are needed for one PV m16k16 A fragment"
);

__device__ __forceinline__ uint32_t shared_address(const void *ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mma_row_reduce_sum(
    const float (&accum)[4], float (&row_sum)[2]
) {
    row_sum[0] = accum[0] + accum[1];
    row_sum[1] = accum[2] + accum[3];

#pragma unroll
    for (int offset = 2; offset > 0; offset >>= 1) {
        row_sum[0] += __shfl_xor_sync(0xffffffff, row_sum[0], offset, 4);
        row_sum[1] += __shfl_xor_sync(0xffffffff, row_sum[1], offset, 4);
    }
}

// One m16n8 accumulator fragment gives each lane two values from each of two
// rows. Four consecutive lanes collectively own all eight values of those rows.
__device__ __forceinline__ void mma_row_reduce_max(
    const float (&accum)[4], float (&row_max)[2]
) {
    row_max[0] = fmaxf(accum[0], accum[1]);
    row_max[1] = fmaxf(accum[2], accum[3]);

#pragma unroll
    for (int offset = 2; offset > 0; offset >>= 1) {
        row_max[0] = fmaxf(
            row_max[0], __shfl_xor_sync(0xffffffff, row_max[0], offset, 4)
        );
        row_max[1] = fmaxf(
            row_max[1], __shfl_xor_sync(0xffffffff, row_max[1], offset, 4)
        );
    }
}

__device__ __forceinline__ uint32_t pack_bf16x2(float low, float high) {
    __nv_bfloat162 value = __floats2bfloat162_rn(low, high);
    __nv_bfloat162_raw raw = static_cast<__nv_bfloat162_raw>(value);
    return static_cast<uint32_t>(raw.x) |
           (static_cast<uint32_t>(raw.y) << 16);
}

// Two adjacent m16n8 score fragments have the same per-lane ownership as one
// row-major m16k16 A fragment. Therefore P can feed PV directly from registers.
__device__ __forceinline__ void probability_to_mma_a(
    const float probability[][4], int k_fragment, uint32_t (&out)[4]
) {
    const int left = 2 * k_fragment;
    const int right = left + 1;

    out[0] = pack_bf16x2(probability[left][0], probability[left][1]);
    out[1] = pack_bf16x2(probability[left][2], probability[left][3]);
    out[2] = pack_bf16x2(probability[right][0], probability[right][1]);
    out[3] = pack_bf16x2(probability[right][2], probability[right][3]);
}

// rows/cols describe valid elements in the shared tile, including for K^T.
template <int TILE_ROWS, int TILE_COLS, bool TRANSPOSE = false>
__device__ __forceinline__ void block_load_tile(
    const __nv_bfloat16 *m_ptr,
    int stride,
    int rows,
    int cols,
    __nv_bfloat16 *out
) {
    for (int index = threadIdx.x; index < TILE_ROWS * TILE_COLS;
         index += blockDim.x) {
        int row = index / TILE_COLS;
        int col = index % TILE_COLS;
        int source_index = TRANSPOSE ? col * stride + row : row * stride + col;
        out[index] = row < rows && col < cols
                         ? m_ptr[source_index]
                         : __float2bfloat16_rn(0.0f);
    }
}

__device__ __forceinline__ void load_mma_matrix(
    uint32_t *out,
    const __nv_bfloat16 *in,
    int logical_row,
    int logical_col,
    int stride,
    int lane_id
) {
    int row = logical_row + lane_id % 16;
    int col = logical_col + (lane_id / 16) * 8;

    int index = row * stride + col;
    uint32_t address = shared_address(&in[index]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0, %1, %2, %3}, [%4];\n"
                 : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
                 : "r"(address));
}

__device__ __forceinline__ void load_mma_transpose(
    uint32_t *out,
    const __nv_bfloat16 *in,
    int logical_row,
    int logical_col,
    int stride,
    int lane_id
) {
    int row = logical_row + lane_id % 16;
    int col = logical_col;

    int index = row * stride + col;
    uint32_t address = shared_address(&in[index]);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(out[0]), "=r"(out[1])
        : "r"(address)
    );
}

__device__ __forceinline__ void mma_m16n8k16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])
    );
}

// Warp-level Q * K^T on shared tiles; the caller handles block barriers.
__device__ __forceinline__ void compute_qk_tile(
    const __nv_bfloat16 *Qs,
    const __nv_bfloat16 *KVs,
    int warp_id,
    int lane_id,
    float (&accum)[WARP_MMA_M][WARP_MMA_N][4]
) {
    for (int kk = 0; kk < BK; kk += MMA_K) {
        uint32_t a_frag[WARP_MMA_M][4];
        uint32_t b_frag[WARP_MMA_N][2];

#pragma unroll
        for (int n_frag = 0; n_frag < WARP_MMA_N; n_frag++) {
            load_mma_transpose(
                b_frag[n_frag], KVs, kk, n_frag * MMA_N, BN, lane_id
            );
        }

        int warp_row_start = warp_id * WARP_M;
#pragma unroll
        for (int m_tile = 0; m_tile < WARP_MMA_M; m_tile++) {
            int row = warp_row_start + MMA_M * m_tile;
            int col = kk;
            load_mma_matrix(a_frag[m_tile], Qs, row, col, BK, lane_id);

#pragma unroll
            for (int n_frag = 0; n_frag < WARP_MMA_N; n_frag++) {
                mma_m16n8k16(
                    accum[m_tile][n_frag],
                    a_frag[m_tile],
                    b_frag[n_frag]
                );
            }
        }
    }
}

// Update the running maximum and denominator; keep P and alpha for P * V.
__device__ __forceinline__ void online_softmax(
    float (&accum)[WARP_MMA_M][WARP_MMA_N][4],
    float (&row_max)[WARP_MMA_M][2],
    float (&log_sum)[WARP_MMA_M][2],
    float (&probability)[WARP_MMA_M][WARP_MMA_N][4],
    float (&alpha)[WARP_MMA_M][2],
    int warp_row_start,
    int lane_id,
    int n_tile,
    int seq_len,
    float scale,
    bool causal
) {
    for (int m_tile = 0; m_tile < WARP_MMA_M; m_tile++) {
        int row0 = warp_row_start + m_tile * MMA_M + lane_id / 4;
        int row1 = row0 + 8;

        bool valid[WARP_MMA_N][4];
        float tile_max[2] = {-INFINITY, -INFINITY};

#pragma unroll
        for (int n_frag = 0; n_frag < WARP_MMA_N; n_frag++) {
            int col0 = n_tile + n_frag * MMA_N + (lane_id % 4) * 2;
            int col1 = col0 + 1;

            valid[n_frag][0] = row0 < seq_len && col0 < seq_len &&
                               (!causal || col0 <= row0);
            valid[n_frag][1] = row0 < seq_len && col1 < seq_len &&
                               (!causal || col1 <= row0);
            valid[n_frag][2] = row1 < seq_len && col0 < seq_len &&
                               (!causal || col0 <= row1);
            valid[n_frag][3] = row1 < seq_len && col1 < seq_len &&
                               (!causal || col1 <= row1);

#pragma unroll
            for (int value = 0; value < 4; value++) {
                accum[m_tile][n_frag][value] =
                    valid[n_frag][value]
                        ? accum[m_tile][n_frag][value] * scale
                        : -INFINITY;
            }

            float fragment_max[2];
            mma_row_reduce_max(accum[m_tile][n_frag], fragment_max);
            tile_max[0] = fmaxf(tile_max[0], fragment_max[0]);
            tile_max[1] = fmaxf(tile_max[1], fragment_max[1]);
        }

        float new_row_max[2];
        for (int row_idx = 0; row_idx < 2; row_idx++) {
            if (tile_max[row_idx] == -INFINITY) {
                new_row_max[row_idx] = row_max[m_tile][row_idx];
                alpha[m_tile][row_idx] = 1.0f;
            } else {
                new_row_max[row_idx] =
                    fmaxf(row_max[m_tile][row_idx], tile_max[row_idx]);
                alpha[m_tile][row_idx] =
                    expf(row_max[m_tile][row_idx] - new_row_max[row_idx]);
            }
        }

        float tile_sum[2] = {};
#pragma unroll
        for (int n_frag = 0; n_frag < WARP_MMA_N; n_frag++) {
            probability[m_tile][n_frag][0] =
                valid[n_frag][0]
                    ? expf(accum[m_tile][n_frag][0] - new_row_max[0])
                    : 0.0f;
            probability[m_tile][n_frag][1] =
                valid[n_frag][1]
                    ? expf(accum[m_tile][n_frag][1] - new_row_max[0])
                    : 0.0f;
            probability[m_tile][n_frag][2] =
                valid[n_frag][2]
                    ? expf(accum[m_tile][n_frag][2] - new_row_max[1])
                    : 0.0f;
            probability[m_tile][n_frag][3] =
                valid[n_frag][3]
                    ? expf(accum[m_tile][n_frag][3] - new_row_max[1])
                    : 0.0f;

            float fragment_sum[2];
            mma_row_reduce_sum(probability[m_tile][n_frag], fragment_sum);
            tile_sum[0] += fragment_sum[0];
            tile_sum[1] += fragment_sum[1];
        }

        for (int row_idx = 0; row_idx < 2; row_idx++) {
            log_sum[m_tile][row_idx] =
                log_sum[m_tile][row_idx] * alpha[m_tile][row_idx] +
                tile_sum[row_idx];
            row_max[m_tile][row_idx] = new_row_max[row_idx];
        }
    }
}

template <int HEAD_MMA_N>
__device__ __forceinline__ void rescale_output(
    float (&output_accum)[WARP_MMA_M][HEAD_MMA_N][4],
    const float (&alpha)[WARP_MMA_M][2]
) {
    // The old numerator belongs to the previous row maximum. Rescale it
    // before accumulating the current tile's unnormalised P * V.
#pragma unroll
    for (int m_tile = 0; m_tile < WARP_MMA_M; m_tile++) {
#pragma unroll
        for (int d_frag = 0; d_frag < HEAD_MMA_N; d_frag++) {
            output_accum[m_tile][d_frag][0] *= alpha[m_tile][0];
            output_accum[m_tile][d_frag][1] *= alpha[m_tile][0];
            output_accum[m_tile][d_frag][2] *= alpha[m_tile][1];
            output_accum[m_tile][d_frag][3] *= alpha[m_tile][1];
        }
    }
}

// Accumulate one shared V tile into the persistent output fragments.
template <int HEAD_MMA_N>
__device__ __forceinline__ void compute_pv_tile(
    const float (&probability)[WARP_MMA_M][WARP_MMA_N][4],
    const __nv_bfloat16 *KVs,
    int d_tile,
    int lane_id,
    float (&output_accum)[WARP_MMA_M][HEAD_MMA_N][4]
) {
#pragma unroll
    for (int k_frag = 0; k_frag < WARP_MMA_N / 2; k_frag++) {
        uint32_t b_frag[BK / MMA_N][2];

#pragma unroll
        for (int d_frag = 0; d_frag < BK / MMA_N; d_frag++) {
            load_mma_transpose(
                b_frag[d_frag],
                KVs,
                k_frag * MMA_K,
                d_frag * MMA_N,
                BK,
                lane_id
            );
        }

#pragma unroll
        for (int m_tile = 0; m_tile < WARP_MMA_M; m_tile++) {
            uint32_t p_frag[4];
            probability_to_mma_a(probability[m_tile], k_frag, p_frag);

#pragma unroll
            for (int d_frag = 0; d_frag < BK / MMA_N; d_frag++) {
                mma_m16n8k16(
                    output_accum[m_tile][d_tile / MMA_N + d_frag],
                    p_frag,
                    b_frag[d_frag]
                );
            }
        }
    }
}

template <int HEAD_MMA_N>
__device__ __forceinline__ void store_output(
    __nv_bfloat16 *O,
    const float (&output_accum)[WARP_MMA_M][HEAD_MMA_N][4],
    const float (&log_sum)[WARP_MMA_M][2],
    int warp_row_start,
    int lane_id,
    int seq_len,
    int head_dim
) {
#pragma unroll
    for (int m_tile = 0; m_tile < WARP_MMA_M; m_tile++) {
        int row0 = warp_row_start + m_tile * MMA_M + lane_id / 4;
        int row1 = row0 + 8;
        int col_pair = (lane_id % 4) * 2;

#pragma unroll
        for (int d_frag = 0; d_frag < HEAD_MMA_N; d_frag++) {
            int col0 = d_frag * MMA_N + col_pair;
            int col1 = col0 + 1;
            if (row0 < seq_len) {
                O[row0 * head_dim + col0] = __float2bfloat16_rn(
                    output_accum[m_tile][d_frag][0] /
                    log_sum[m_tile][0]
                );
                O[row0 * head_dim + col1] = __float2bfloat16_rn(
                    output_accum[m_tile][d_frag][1] /
                    log_sum[m_tile][0]
                );
            }
            if (row1 < seq_len) {
                O[row1 * head_dim + col0] = __float2bfloat16_rn(
                    output_accum[m_tile][d_frag][2] /
                    log_sum[m_tile][1]
                );
                O[row1 * head_dim + col1] = __float2bfloat16_rn(
                    output_accum[m_tile][d_frag][3] /
                    log_sum[m_tile][1]
                );
            }
        }
    }
}

} // namespace

template <int HEAD_DIM>
__global__ void flash_attn_v2_kernel(
    const __nv_bfloat16 *__restrict__ Q,
    const __nv_bfloat16 *__restrict__ K,
    const __nv_bfloat16 *__restrict__ V,
    __nv_bfloat16 *__restrict__ O,
    int seq_len,
    int head_dim,
    bool causal
) {
    constexpr int HEAD_MMA_N = HEAD_DIM / MMA_N;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int row_start = bid * BM;
    const int block_rows = min(BM, seq_len - row_start);

    const __nv_bfloat16 *blockQ = Q + row_start * head_dim;

    int lane_id = tid & 31;
    int warp_id = tid >> 5;

    __shared__ __nv_bfloat16 Qs[BM * BK];
    __shared__ __nv_bfloat16 KVs[BK * BN];

    float row_max[WARP_MMA_M][2] = {};
    float log_sum[WARP_MMA_M][2] = {};
    float output_accum[WARP_MMA_M][HEAD_MMA_N][4] = {};
    const float scale = rsqrtf((float)head_dim);
    for (int i = 0; i < WARP_MMA_M; i++) {
        row_max[i][0] = -INFINITY;
        row_max[i][1] = -INFINITY;
    }

    for (int n_tile = 0; n_tile < seq_len; n_tile += BN) {
        float accum[WARP_MMA_M][WARP_MMA_N][4] = {};
        float probability[WARP_MMA_M][WARP_MMA_N][4];
        float alpha[WARP_MMA_M][2];

        for (int d_tile = 0; d_tile < head_dim; d_tile += BK) {
            // load Q tile 和 K tile 到共享内存里面来
            int tile_d = min(BK, head_dim - d_tile);
            block_load_tile<BM, BK>(
                blockQ + d_tile, head_dim, block_rows, tile_d, Qs
            );

            const __nv_bfloat16 *blockK = K + n_tile * head_dim + d_tile;
            int tile_n = min(BN, seq_len - n_tile);
            block_load_tile<BK, BN, true>(
                blockK, head_dim, tile_d, tile_n, KVs
            );
            __syncthreads();

            compute_qk_tile(Qs, KVs, warp_id, lane_id, accum);
            __syncthreads();
        }

        online_softmax(
            accum, row_max, log_sum, probability, alpha,
            row_start + warp_id * WARP_M, lane_id, n_tile, seq_len, scale, causal
        );
        rescale_output(output_accum, alpha);

        for (int d_tile = 0; d_tile < HEAD_DIM; d_tile += BK) {
            int tile_n = min(BN, seq_len - n_tile);
            int tile_d = min(BK, HEAD_DIM - d_tile);
            block_load_tile<BN, BK>(
                V + n_tile * head_dim + d_tile,
                head_dim,
                tile_n,
                tile_d,
                KVs
            );
            __syncthreads();

            compute_pv_tile(probability, KVs, d_tile, lane_id, output_accum);
            __syncthreads();
        }
    }

    store_output(
        O, output_accum, log_sum,
        row_start + warp_id * WARP_M, lane_id, seq_len, head_dim
    );
}

void solve_v2(
    const __nv_bfloat16 *Q,
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    __nv_bfloat16 *O,
    int seq_len,
    int head_dim,
    bool causal
) {
    dim3 block(THREADS);
    dim3 grid((seq_len + BM - 1) / BM);

    switch (head_dim) {
    case 32:
        flash_attn_v2_kernel<32><<<grid, block>>>(
            Q, K, V, O, seq_len, head_dim, causal
        );
        break;
    case 64:
        flash_attn_v2_kernel<64><<<grid, block>>>(
            Q, K, V, O, seq_len, head_dim, causal
        );
        break;
    case 128:
        flash_attn_v2_kernel<128><<<grid, block>>>(
            Q, K, V, O, seq_len, head_dim, causal
        );
        break;
    }
}
