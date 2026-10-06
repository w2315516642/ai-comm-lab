#include <cmath>

#include <cuda_bf16.h>
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
    const float result = scratch[0];
    __syncthreads();
    return result;
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
__global__ void flash_attn_v0_kernel(
    const __nv_bfloat16 *__restrict__ Q,
    const __nv_bfloat16 *__restrict__ K,
    const __nv_bfloat16 *__restrict__ V,
    __nv_bfloat16 *__restrict__ O,
    int seq_len,
    int head_dim,
    bool causal
) {
    extern __shared__ float scores[];
    __shared__ float reduction[kThreads];

    const int query_idx = blockIdx.x;
    const int valid_keys = causal ? query_idx + 1 : seq_len;
    const __nv_bfloat16 *q = Q + (size_t)query_idx * head_dim;
    const float scale = rsqrtf((float)head_dim);

    float local_max = -INFINITY;
    for (int key_idx = threadIdx.x; key_idx < valid_keys;
         key_idx += blockDim.x) {
        const __nv_bfloat16 *k = K + (size_t)key_idx * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++)
            dot += __bfloat162float(q[d]) * __bfloat162float(k[d]);

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
            const size_t v_idx = (size_t)key_idx * head_dim + d;
            value += scores[key_idx] * __bfloat162float(V[v_idx]);
        }
        O[(size_t)query_idx * head_dim + d] =
            __float2bfloat16_rn(value / row_sum);
    }
}

} // namespace

void solve_v0(
    const __nv_bfloat16 *Q,
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    __nv_bfloat16 *O,
    int seq_len,
    int head_dim,
    bool causal
) {
    const size_t shared_bytes = (size_t)seq_len * sizeof(float);
    flash_attn_v0_kernel<<<seq_len, kThreads, shared_bytes>>>(
        Q, K, V, O, seq_len, head_dim, causal
    );
}
