#include <cuda_runtime.h>

struct topk_value {
    float value;
    int index;
};

__device__ topk_value block_max(float value, int index) {

    for (int offset = 16; offset > 0; offset /= 2) {
        float compare_value = __shfl_down_sync(0xffffffff, value, offset);
        int compare_index = __shfl_down_sync(0xffffffff, index, offset);
        if (compare_value > value) {
            value = compare_value;
            index = compare_index;
        }
    }

    __shared__ float warp_values[8];
    __shared__ int warp_indices[8];
    __shared__ topk_value block_result;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = (blockDim.x + 31) / 32;

    if (lane_id == 0) {
        warp_values[warp_id] = value;
        warp_indices[warp_id] = index;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_value =
            lane_id < num_warps ? warp_values[lane_id] : -INFINITY;
        int block_index = lane_id < num_warps ? warp_indices[lane_id] : -1;

        for (int offset = 16; offset > 0; offset /= 2) {
            float compare_value =
                __shfl_down_sync(0xffffffff, block_value, offset);
            int compare_index =
                __shfl_down_sync(0xffffffff, block_index, offset);

            if (compare_value > block_value) {
                block_value = compare_value;
                block_index = compare_index;
            }
        }

        if (lane_id == 0) {
            block_result = {block_value, block_index};
        }
    }

    __syncthreads();
    return block_result;
}

__device__ float block_sum(float value) {

    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);

    __shared__ float warp_values[8];
    __shared__ float block_sum;
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int num_warps = (blockDim.x + 31) / 32;

    if (lane_id == 0)
        warp_values[warp_id] = value;
    __syncthreads();

    if (warp_id == 0) {
        float block_value = lane_id < num_warps ? warp_values[lane_id] : 0;
        for (int offset = 16; offset > 0; offset /= 2)
            block_value += __shfl_down_sync(0xffffffff, block_value, offset);

        if (lane_id == 0)
            block_sum = block_value;
    }

    __syncthreads();
    return block_sum;
}

__device__ void softmax_kernel(
    float *topk_weights, int rowIdx, int colIdx, int k
) {
    float value = colIdx < k ? topk_weights[rowIdx * k + colIdx] : -INFINITY;
    topk_value top1 = block_max(value, colIdx);
    float expVal = __expf(value - top1.value);

    float expSum = block_sum(expVal);
    if (colIdx < k)
        topk_weights[rowIdx * k + colIdx] = expVal / expSum;
    __syncthreads();
}

__global__ void topk_kernel(
    const float *logits,
    float *topk_weights,
    int *topk_indices,
    int M,
    int E,
    int k
) {

    int row = blockIdx.x;
    int col = threadIdx.x;
    float value = col < E ? logits[row * E + col] : -INFINITY;

    for (int i = 0; i < k; i++) {
        topk_value topi = block_max(value, col);
        if (col == topi.index)
            value = -INFINITY;

        if (threadIdx.x == 0) {
            topk_indices[row * k + i] = topi.index;
            topk_weights[row * k + i] = topi.value;
        }
    }

    __syncthreads();
    softmax_kernel(topk_weights, row, col, k);
}

// logits, topk_weights, topk_indices are device pointers
extern "C" void solve_ans0(
    const float *logits,
    float *topk_weights,
    int *topk_indices,
    int M,
    int E,
    int k
) {

    topk_kernel<<<M, 256>>>(logits, topk_weights, topk_indices, M, E, k);
}
