#include <cmath>
#include <cuda_runtime.h>
#include <float.h>

__device__ void softmax_kernel_v2(float *input, int n) {
    float row_max = -INFINITY;
    for (int i = 0; i < n; i++) {
        row_max = fmaxf(row_max, input[i]);
    }

    float row_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        row_sum += expf(input[i] - row_max);
    }

    for (int i = 0; i < n; i++) {
        input[i] = expf(input[i] - row_max) / row_sum;
    }
}

__global__ void moe_topk_kernel_v2(
    const float *logits,
    float *topk_weights,
    int *topk_indices,
    int M,
    int E,
    int k
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    if (warp_id >= M)
        return;

    int m = warp_id;

    int base = m * E;
    int base_o = m * k;

    int top_idxs[8]; // 假设 k 不超过 8 个
    float top_vals[8];

    // 找前 k 个最大值和对应的 idx
    for (int r = 0; r < k; r++) {
        float best_val = -INFINITY;
        int best_idx = -1;

        // 每个 lane 遍历自己负责的那部分专家
        for (int e = lane_id; e < E; e += 32) {
            bool used = false;
            for (int prev = 0; prev < r; prev++) {
                if (top_idxs[prev] == e) {
                    used = true;
                    break;
                }
            }
            if (used)
                continue;

            float val = logits[base + e];
            // 防止 logits 里面有 -inf，导致 best_idx 保持 -1
            if (val > best_val) {
                best_val = val;
                best_idx = e;
            }
        }

        // 现在有 32 个最大值和对应的 idx
        // 取出 32 个里面最大的那部分
        for (int offset = 16; offset > 0; offset >>= 1) {
            // 从前面的线程拿值
            float other_val = __shfl_down_sync(0xffffffff, best_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, best_idx, offset);
            if (other_val > best_val ||
                (other_val == best_val && other_idx < best_idx)) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }
        // 取完后 lane0 有最大值和 idx
        // 每个线程需要记录这轮的 idx，用于判断下轮选过的 idx
        best_idx = __shfl_sync(0xffffffff, best_idx, 0);
        best_val = __shfl_sync(0xffffffff, best_val, 0);
        top_idxs[r] = best_idx;
        top_vals[r] = best_val;
    }

    softmax_kernel_v2(top_vals, k);
    if (lane_id < k) {
        topk_weights[base_o + lane_id] = top_vals[lane_id];
        topk_indices[base_o + lane_id] = top_idxs[lane_id];
    }
}

extern "C" void solve_v2(
    const float *logits,
    float *topk_weights,
    int *topk_indices,
    int M,
    int E,
    int k
) {
    int threads = 256;
    int warps = (threads + 31) / 32;
    int blocks = (M + warps - 1) / warps;
    moe_topk_kernel_v2<<<blocks, threads>>>(
        logits, topk_weights, topk_indices, M, E, k
    );
}
