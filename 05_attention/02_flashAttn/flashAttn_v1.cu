// Scalar CUDA implementation of FlashAttention-2 forward, Algorithm 1:
// https://arxiv.org/abs/2307.08691 (Sections 3.1-3.3).
// Single-head BF16 Q/K/V/O [N, D]; all arithmetic and probabilities are FP32.
// Each block owns a Q tile; warps own disjoint Q rows (sliced-Q).
// Forward-only interface: return O, omit the LSE used by the backward pass.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int WARP_SIZE = 32;
constexpr int WARPS = 4;
constexpr int ROWS_PER_WARP = 1;
constexpr int BM = WARPS * ROWS_PER_WARP; // Paper: Br.
constexpr int BN = 32;                   // Paper: Bc.
constexpr int MAX_HEAD_DIM = 256;
constexpr int THREADS = WARPS * WARP_SIZE;
constexpr int KEYS_PER_LANE = (BN + WARP_SIZE - 1) / WARP_SIZE;
constexpr int DIMS_PER_LANE = (MAX_HEAD_DIM + WARP_SIZE - 1) / WARP_SIZE;

static_assert(WARPS > 0 && THREADS <= 1024);
static_assert(ROWS_PER_WARP > 0 && BN > 0 && MAX_HEAD_DIM > 0);
static_assert((BM + 2 * BN) * MAX_HEAD_DIM * sizeof(__nv_bfloat16) <= 48 * 1024);

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, offset));
    return value;
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, offset);
    return value;
}

// Read contiguous global elements. K is transposed only when stored to shared
// memory, so lanes computing different keys can read Kt[d, key] together.
template <int TILE_ROWS, bool TRANSPOSE = false>
__device__ __forceinline__ void load_tile(
    const __nv_bfloat16 *src, __nv_bfloat16 *dst, int valid_rows, int head_dim
) {
    for (int i = threadIdx.x; i < TILE_ROWS * head_dim; i += blockDim.x) {
        int row = i / head_dim;
        int d = i % head_dim;
        int out = TRANSPOSE ? d * TILE_ROWS + row : i;
        dst[out] = row < valid_rows ? src[i] : __float2bfloat16_rn(0.0f);
    }
}

// Algorithm 1, line 8: S = Q_i * K_j^T / sqrt(D).
// A lane owns keys lane, lane + 32, ... for one query row.
__device__ __forceinline__ void compute_scores(
    const __nv_bfloat16 *q,
    const __nv_bfloat16 *Kt,
    int head_dim,
    int valid_keys,
    int lane,
    float scale,
    float (&scores)[KEYS_PER_LANE]
) {
#pragma unroll
    for (int s = 0; s < KEYS_PER_LANE; s++) {
        int key = s * WARP_SIZE + lane;
        float dot = 0.0f;
        if (key < valid_keys) {
            for (int d = 0; d < head_dim; d++)
                dot = fmaf(__bfloat162float(q[d]),
                           __bfloat162float(Kt[d * BN + key]), dot);
        }
        scores[s] = key < valid_keys ? dot * scale : -INFINITY;
    }
}

// Algorithm 1, line 9: update m/l, overwrite S with unnormalised P.
// m/l are replicated across lanes; scores/P stay distributed within the warp.
__device__ __forceinline__ float online_softmax(
    float (&scores)[KEYS_PER_LANE], float &row_max, float &row_sum
) {
    float tile_max = -INFINITY;
#pragma unroll
    for (int s = 0; s < KEYS_PER_LANE; s++)
        tile_max = fmaxf(tile_max, scores[s]);
    tile_max = warp_reduce_max(tile_max);

    // Padded rows or fully causal-masked tiles: preserve the running state.
    if (tile_max == -INFINITY) {
#pragma unroll
        for (int s = 0; s < KEYS_PER_LANE; s++)
            scores[s] = 0.0f;
        return 1.0f;
    }

    float new_max = fmaxf(row_max, tile_max);
    float alpha = expf(row_max - new_max);
    float tile_sum = 0.0f;
#pragma unroll
    for (int s = 0; s < KEYS_PER_LANE; s++) {
        scores[s] = expf(scores[s] - new_max);
        tile_sum += scores[s];
    }
    row_sum = alpha * row_sum + warp_reduce_sum(tile_sum);
    row_max = new_max;
    return alpha;
}

// Algorithm 1, line 10: O_tilde = alpha * O_tilde + P * V_j.
// Output columns are lane, lane + 32, ... . Broadcast each P using a warp
// shuffle; no other warp contributes to this query row's output.
__device__ __forceinline__ void accumulate_output(
    const float (&probability)[KEYS_PER_LANE],
    const __nv_bfloat16 *Vs,
    int head_dim,
    int valid_keys,
    int lane,
    float alpha,
    float (&output)[DIMS_PER_LANE]
) {
#pragma unroll
    for (int t = 0; t < DIMS_PER_LANE; t++)
        output[t] *= alpha;

#pragma unroll
    for (int s = 0; s < KEYS_PER_LANE; s++) {
        for (int owner = 0; owner < WARP_SIZE && s * WARP_SIZE + owner < valid_keys;
             owner++) {
            int key = s * WARP_SIZE + owner;
            float p = __shfl_sync(0xffffffff, probability[s], owner);
#pragma unroll
            for (int t = 0; t < DIMS_PER_LANE; t++) {
                int d = t * WARP_SIZE + lane;
                if (d < head_dim)
                    output[t] = fmaf(p, __bfloat162float(Vs[key * head_dim + d]),
                                     output[t]);
            }
        }
    }
}

// Algorithm 1, lines 12/14: normalize only once, then write O to global memory.
__device__ __forceinline__ void store_output(
    __nv_bfloat16 *O,
    const float (&output)[DIMS_PER_LANE],
    float row_sum,
    int query,
    int seq_len,
    int head_dim,
    int lane
) {
    if (query >= seq_len)
        return;
#pragma unroll
    for (int t = 0; t < DIMS_PER_LANE; t++) {
        int d = t * WARP_SIZE + lane;
        if (d < head_dim)
            O[(size_t)query * head_dim + d] = __float2bfloat16_rn(output[t] / row_sum);
    }
}

__global__ void flash_attn_v1_kernel(
    const __nv_bfloat16 *__restrict__ Q,
    const __nv_bfloat16 *__restrict__ K,
    const __nv_bfloat16 *__restrict__ V,
    __nv_bfloat16 *__restrict__ O,
    int seq_len,
    int head_dim,
    bool causal
) {
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int q_start = blockIdx.x * BM;
    const int q_rows = min(BM, seq_len - q_start);
    const float scale = rsqrtf((float)head_dim);

    extern __shared__ __nv_bfloat16 shared[];
    __nv_bfloat16 *Qs = shared;                   // [BM, D]
    __nv_bfloat16 *Kt = Qs + BM * head_dim;       // [D, BN]
    __nv_bfloat16 *Vs = Kt + BN * head_dim;       // [BN, D]

    float row_max[ROWS_PER_WARP];
    float row_sum[ROWS_PER_WARP] = {};
    float output[ROWS_PER_WARP][DIMS_PER_LANE] = {};
#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++)
        row_max[r] = -INFINITY;

    // Algorithm 1, lines 3-5: one Q tile per block, loaded once.
    load_tile<BM>(Q + (size_t)q_start * head_dim, Qs, q_rows, head_dim);
    // Causal tiles entirely to the right of this Q block can be skipped.
    const int key_end = causal ? q_start + q_rows : seq_len;
    for (int k_start = 0; k_start < key_end; k_start += BN) {
        int key_rows = min(BN, key_end - k_start);
        load_tile<BN, true>(K + (size_t)k_start * head_dim, Kt, key_rows, head_dim);
        load_tile<BN>(V + (size_t)k_start * head_dim, Vs, key_rows, head_dim);
        __syncthreads(); // Q/K/V tiles are ready for every warp.

#pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            int local_row = warp * ROWS_PER_WARP + r;
            int query = q_start + local_row;
            int valid_keys = query < seq_len ? key_rows : 0;
            if (causal)
                valid_keys = min(valid_keys, max(0, query - k_start + 1));

            float scores[KEYS_PER_LANE];
            compute_scores(Qs + local_row * head_dim, Kt, head_dim,
                           valid_keys, lane, scale, scores);
            float alpha = online_softmax(scores, row_max[r], row_sum[r]);
            accumulate_output(scores, Vs, head_dim, valid_keys, lane, alpha, output[r]);
        }
        __syncthreads(); // All warps finish reading before K/V are overwritten.
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++)
        store_output(O, output[r], row_sum[r],
                     q_start + warp * ROWS_PER_WARP + r, seq_len, head_dim, lane);
}

} // namespace

bool supports_v1(int seq_len, int head_dim, bool) {
    return seq_len > 0 && head_dim > 0 && head_dim <= MAX_HEAD_DIM;
}

void solve_v1(
    const __nv_bfloat16 *Q,
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    __nv_bfloat16 *O,
    int seq_len,
    int head_dim,
    bool causal
) {
    if (!supports_v1(seq_len, head_dim, causal)) {
        fprintf(stderr, "v1 requires N > 0 and 1 <= D <= %d\n", MAX_HEAD_DIM);
        std::abort();
    }
    size_t shared_bytes = (size_t)(BM + 2 * BN) * head_dim * sizeof(__nv_bfloat16);
    flash_attn_v1_kernel<<<(seq_len + BM - 1) / BM, THREADS, shared_bytes>>>(
        Q, K, V, O, seq_len, head_dim, causal
    );
}
