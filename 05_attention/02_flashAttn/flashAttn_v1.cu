#include <cmath>

#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

__device__ float block_reduce_max(float value, float *scratch) {
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            scratch[tid] = fmaxf(scratch[tid], scratch[tid + offset]);
        __syncthreads();
    }
    return scratch[0];
}

__device__ float block_reduce_sum(float value, float *scratch) {
    const int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset)
            scratch[tid] += scratch[tid + offset];
        __syncthreads();
    }
    return scratch[0];
}

// Baseline fused attention: one block owns one query row and keeps that row's
// probabilities in shared memory. Later versions can tile both sequence axes
// to remove this O(N) shared-memory requirement.
__global__ void flash_attn_v1_kernel(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    float *__restrict__ O,
    int seq_len,
    int head_dim,
    bool causal
) {
    extern __shared__ float scores[];
    __shared__ float reduction[kThreads];

    const int row = blockIdx.x;
    const int query_idx = row % seq_len;
    const int batch_head = row / seq_len;
    const int valid_keys = causal ? query_idx + 1 : seq_len;
    const size_t row_base = (size_t)batch_head * seq_len * head_dim;
    const float *q = Q + row_base + (size_t)query_idx * head_dim;
    const float scale = rsqrtf((float)head_dim);

    float local_max = -CUDART_INF_F;
    for (int key_idx = threadIdx.x; key_idx < valid_keys;
         key_idx += blockDim.x) {
        const float *k = K + row_base + (size_t)key_idx * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += q[d] * k[d];

        const float score = dot * scale;
        scores[key_idx] = score;
        local_max = fmaxf(local_max, score);
    }
    const float row_max = block_reduce_max(local_max, reduction);

    float local_sum = 0.0f;
    for (int key_idx = threadIdx.x; key_idx < valid_keys;
         key_idx += blockDim.x) {
        const float probability = expf(scores[key_idx] - row_max);
        scores[key_idx] = probability;
        local_sum += probability;
    }
    const float row_sum = block_reduce_sum(local_sum, reduction);

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float value = 0.0f;
        for (int key_idx = 0; key_idx < valid_keys; key_idx++) {
            const size_t v_idx =
                row_base + (size_t)key_idx * head_dim + d;
            value += scores[key_idx] * V[v_idx];
        }
        O[row_base + (size_t)query_idx * head_dim + d] = value / row_sum;
    }
}

} // namespace

void solve_v1(
    const float *Q,
    const float *K,
    const float *V,
    float *O,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    bool causal
) {
    const int rows = batch * heads * seq_len;
    const size_t shared_bytes = (size_t)seq_len * sizeof(float);
    flash_attn_v1_kernel<<<rows, kThreads, shared_bytes>>>(
        Q, K, V, O, seq_len, head_dim, causal
    );
}
