#include <cuda_runtime.h>
__device__ void warp_max(float &x, int &idx) {
#pragma unroll
    for (int stride = 16; stride >= 1; stride /= 2) {
        float n_x = __shfl_xor_sync(0xffffffff, x, stride);
        int n_idx = __shfl_xor_sync(0xffffffff, idx, stride);
        if (n_x > x) {
            x = n_x;
            idx = n_idx;
        }
    }
}
template <int kThread> __device__ void block_max(float &x, int &idx) {
    __shared__ float S[kThread / 32];
    __shared__ int I[kThread / 32];
    warp_max(x, idx);
    if (threadIdx.x % 32 == 0) {
        S[threadIdx.x / 32] = x;
        I[threadIdx.x / 32] = idx;
    }
    __syncthreads();
    float warp_m = -INFINITY;
    int warp_i = -1;
    if (threadIdx.x < kThread / 32) {
        warp_m = S[threadIdx.x];
        warp_i = I[threadIdx.x];
    }
    __syncwarp();
    if (threadIdx.x < 32) {
        warp_max(warp_m, warp_i);
        if (threadIdx.x == 0) {
            S[0] = warp_m;
            I[0] = warp_i;
        }
    }
    __syncthreads();
    x = S[0];
    idx = I[0];
}
__global__ void moe_topk(
    const float *logits,
    float *topk_weights,
    int *topk_indices,
    int M,
    int E,
    int k
) {
    int row = blockIdx.x;
    int i_offset = row * E;
    int o_offset = row * k;
    __shared__ float S[256];
    __shared__ float W[256];
    if (threadIdx.x < E) {
        S[threadIdx.x] = logits[i_offset + threadIdx.x];
    }
    __syncthreads();
    for (int o = 0; o < k; o++) {
        float val = -INFINITY;
        int idx = -1;
        if (threadIdx.x < E) {
            val = S[threadIdx.x];
            idx = threadIdx.x;
        }
        block_max<256>(val, idx);
        if (threadIdx.x == 0) {
            topk_indices[o_offset + o] = idx;
            W[o] = val;
            S[idx] = -INFINITY;
        }
        __syncthreads();
    }
    float sum = 0;
    for (int i = 0; i < k; i++) {
        sum += __expf(W[i]);
    }
    if (threadIdx.x < k) {
        topk_weights[threadIdx.x + o_offset] = __expf(W[threadIdx.x]) / sum;
    }
}
// logits, topk_weights, topk_indices are device pointers
extern "C" void solve_ans1(
    const float *logits,
    float *topk_weights,
    int *topk_indices,
    int M,
    int E,
    int k
) {
    moe_topk<<<M, 256>>>(logits, topk_weights, topk_indices, M, E, k);
}
