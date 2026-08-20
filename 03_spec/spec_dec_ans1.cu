#include <cuda_runtime.h>

namespace {

constexpr int THREADS = 256;

__device__ __forceinline__ int sample_uniform(float u, int V) {
    int token = static_cast<int>(ceilf(u * static_cast<float>(V))) - 1;
    if (token < 0) {
        token = 0;
    }
    if (token >= V) {
        token = V - 1;
    }
    return token;
}

__device__ __forceinline__ float distribution_value(
    const float *__restrict__ draft_probs,
    const float *__restrict__ target_probs,
    int base,
    int v,
    int rejected
) {
    const float q = target_probs[base + v];
    if (!rejected) {
        return q;
    }
    return fmaxf(0.0f, q - draft_probs[base + v]);
}

__global__ void verify_block_sample_kernel(
    const int *__restrict__ draft_tokens,
    const float *__restrict__ draft_probs,
    const float *__restrict__ target_probs,
    const float *__restrict__ uniform_samples,
    int *__restrict__ output_tokens,
    int B,
    int T,
    int V
) {
    const int b = blockIdx.x;
    if (b >= B) {
        return;
    }

    __shared__ int sample_pos_s;
    __shared__ int write_pos_s;
    __shared__ int rejected_s;
    __shared__ int found_s;
    __shared__ int sampled_token_s;
    __shared__ float sample_u_s;
    __shared__ float total_s;
    __shared__ float prefix_s;
    __shared__ float tile_values[THREADS];
    __shared__ float partial_sums[THREADS];

    int *out = output_tokens + b * (T + 1);
    for (int i = threadIdx.x; i <= T; i += blockDim.x) {
        out[i] = 0;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int sample_pos = T - 1;
        int write_pos = T;
        int rejected = 0;

        for (int i = 0; i < T; ++i) {
            const int token = draft_tokens[b * T + i];
            const int token_offset = (b * T + i) * V + token;
            const float p = draft_probs[token_offset];
            const float q = target_probs[token_offset];
            float alpha = q / p;
            alpha = alpha < 1.0f ? alpha : 1.0f;

            if (uniform_samples[b * (T + 1) + i] < alpha) {
                out[i] = token;
            } else {
                sample_pos = i;
                write_pos = i;
                rejected = 1;
                break;
            }
        }

        sample_pos_s = sample_pos;
        write_pos_s = write_pos;
        rejected_s = rejected;
        sample_u_s = uniform_samples[b * (T + 1) + T];
    }

    __syncthreads();

    const int dist_base = (b * T + sample_pos_s) * V;

    if (rejected_s) {
        float local_total = 0.0f;
        for (int v = threadIdx.x; v < V; v += blockDim.x) {
            local_total +=
                distribution_value(draft_probs, target_probs, dist_base, v, 1);
        }

        partial_sums[threadIdx.x] = local_total;
        __syncthreads();

        for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            total_s = partial_sums[0];
        }
        __syncthreads();

        if (total_s <= 0.0f) {
            if (threadIdx.x == 0) {
                out[write_pos_s] = sample_uniform(sample_u_s, V);
            }
            return;
        }
    } else if (threadIdx.x == 0) {
        total_s = 1.0f;
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        found_s = 0;
        sampled_token_s = V - 1;
        prefix_s = 0.0f;
    }
    __syncthreads();

    const float threshold = sample_u_s * total_s;

    for (int tile = 0; tile < V; tile += blockDim.x) {
        const int v = tile + threadIdx.x;
        const float value =
            (v < V) ? distribution_value(
                          draft_probs, target_probs, dist_base, v, rejected_s
                      )
                    : 0.0f;
        tile_values[threadIdx.x] = value;
        partial_sums[threadIdx.x] = value;
        __syncthreads();

        for (int stride = THREADS / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0 && !found_s) {
            const float tile_sum = partial_sums[0];
            if (prefix_s + tile_sum >= threshold) {
                float running = prefix_s;
                const int tile_count = min(blockDim.x, V - tile);
                for (int i = 0; i < tile_count; ++i) {
                    running += tile_values[i];
                    if (running >= threshold) {
                        sampled_token_s = tile + i;
                        found_s = 1;
                        break;
                    }
                }
            } else {
                prefix_s += tile_sum;
            }
        }

        __syncthreads();
        if (found_s) {
            break;
        }
    }

    if (threadIdx.x == 0) {
        out[write_pos_s] = sampled_token_s;
    }
}

} // namespace

extern "C" void solve_ans1(
    const int *draft_tokens,
    const float *draft_probs,
    const float *target_probs,
    const float *uniform_samples,
    int *output_tokens,
    int B,
    int T,
    int V
) {
    verify_block_sample_kernel<<<B, THREADS>>>(
        draft_tokens,
        draft_probs,
        target_probs,
        uniform_samples,
        output_tokens,
        B,
        T,
        V
    );
}
