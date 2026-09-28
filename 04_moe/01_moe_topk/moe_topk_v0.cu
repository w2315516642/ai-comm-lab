#include <cmath>
#include <cuda_runtime.h>
#include <float.h>

__device__ void softmax_kernel(float *input, int n) {
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

__global__ void moe_topk_kernel_v0(
    const float *logits,
    float *topk_weights,
    int *topk_indices,
    int M,
    int E,
    int k
) {
    int m = blockDim.x * blockIdx.x + threadIdx.x;
    if (m >= M)
        return;

    int base = m * E;
    int base_o = m * k;

    // 找前 k 个最大值和对应的 idx
    for (int r = 0; r < k; r++) {
        float best_val = -FLT_MAX;
        int best_idx = -1;

        // 遍历 E 个专家
        for (int e = 0; e < E; e++) {
            bool used = false;
            for (int prev = 0; prev < r; prev++) {
                if (topk_indices[base_o + prev] == e) {
                    used = true;
                    break;
                }
            }
            if (used)
                continue;

            float val = logits[base + e];
            // 防止 logits 里面有 -inf，导致 best_idx 保持 -1
            if (best_idx < 0 || val > best_val) {
                best_val = val;
                best_idx = e;
            }
        }

        topk_weights[base_o + r] = best_val;
        topk_indices[base_o + r] = best_idx;
    }

    softmax_kernel(&topk_weights[base_o], k);
}

extern "C" void solve_v0(
    const float *logits,
    float *topk_weights,
    int *topk_indices,
    int M,
    int E,
    int k
) {
    int threads = 256;
    int blocks = (M + threads - 1) / threads;
    moe_topk_kernel_v0<<<blocks, threads>>>(
        logits, topk_weights, topk_indices, M, E, k
    );
}
