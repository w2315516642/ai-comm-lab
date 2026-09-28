#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <stdio.h>

/**
 * Top-K Gating CUDA Kernel for Mixture of Experts
 * Optimized for NVIDIA A100-80GB GPU
 *
 * Each thread block processes one row (one token)
 * Uses shared memory for efficient data access
 * Implements selection sort for top-k (efficient for small k)
 */
__global__ void topk_gating_kernel(
    const float *__restrict__ logits, // [M, E]
    float *__restrict__ topk_weights, // [M, k]
    int *__restrict__ topk_indices,   // [M, k]
    const int M,
    const int E,
    const int k
) {
    const int row = blockIdx.x;
    if (row >= M)
        return;

    // Shared memory for (value, index) pairs
    extern __shared__ char shared_mem[];
    float *s_values = (float *)shared_mem;
    int *s_indices = (int *)(shared_mem + E * sizeof(float));

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Load logits into shared memory with coalesced access
    const float *row_logits = logits + row * E;
    for (int i = tid; i < E; i += num_threads) {
        s_values[i] = row_logits[i];
        s_indices[i] = i;
    }
    __syncthreads();

    // Selection sort for top-k elements (single thread for simplicity and
    // correctness) For small k (typically k=2), this is very efficient
    if (tid == 0) {
        // Find k largest elements
        for (int i = 0; i < k; i++) {
            // Find max in range [i, E)
            int max_idx = i;
            float max_val = s_values[i];

            for (int j = i + 1; j < E; j++) {
                if (s_values[j] > max_val) {
                    max_val = s_values[j];
                    max_idx = j;
                }
            }

            // Swap to position i
            if (max_idx != i) {
                s_values[max_idx] = s_values[i];
                s_values[i] = max_val;

                int temp_idx = s_indices[max_idx];
                s_indices[max_idx] = s_indices[i];
                s_indices[i] = temp_idx;
            }
        }

        // Compute softmax on top-k values
        // Find max for numerical stability
        float max_val = s_values[0];
        for (int i = 1; i < k; i++) {
            if (s_values[i] > max_val) {
                max_val = s_values[i];
            }
        }

        // Compute exp and sum
        float sum_exp = 0.0f;
        for (int i = 0; i < k; i++) {
            float exp_val = expf(s_values[i] - max_val);
            s_values[i] = exp_val;
            sum_exp += exp_val;
        }

        // Normalize and write output
        float *row_weights = topk_weights + row * k;
        int *row_indices = topk_indices + row * k;

        for (int i = 0; i < k; i++) {
            row_weights[i] = s_values[i] / sum_exp;
            row_indices[i] = s_indices[i];
        }
    }
}

/**
 * Host function for Top-K Gating
 * This is the main entry point called from external code
 */
extern "C" void solve_ans2(
    const float *logits, // [M, E] on GPU
    float *topk_weights, // [M, k] on GPU
    int *topk_indices,   // [M, k] on GPU
    int M,
    int E,
    int k
) {
    // Calculate shared memory size needed per block
    size_t shared_mem_size = E * sizeof(float) + E * sizeof(int);

    // Launch configuration: one block per row
    // Use 128 threads per block (good for A100)
    int threads_per_block = 128;
    int num_blocks = M;

    // Launch kernel
    topk_gating_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(
        logits, topk_weights, topk_indices, M, E, k
    );

    // // Synchronize to ensure completion
    // cudaError_t err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // }
}
