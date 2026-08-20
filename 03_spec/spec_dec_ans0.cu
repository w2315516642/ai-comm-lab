#include <cuda_runtime.h>

#define THREADS 256

__global__ void verify_kernel(
    const int *__restrict__ draft_tokens,      // [B, T]
    const float *__restrict__ draft_probs,     // [B, T, V]
    const float *__restrict__ target_probs,    // [B, T, V]
    const float *__restrict__ uniform_samples, // [B, T+1]
    int *__restrict__ output_tokens,           // [B, T+1]
    int B,
    int T,
    int V
) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    if (b >= B)
        return;

    __shared__ int s_accepted; // number of accepted draft tokens
    __shared__ int s_rejected; // 1 if a rejection happened
    __shared__ int s_rej_pos;
    __shared__ float s_total; // sum of adjusted dist
    __shared__ float s_block[THREADS];
    __shared__ int s_pick;

    if (tid == 0) {
        s_accepted = 0;
        s_rejected = 0;
        s_rej_pos = -1;
    }
    __syncthreads();

    // ---- serial walk over draft positions ----
    for (int i = 0; i < T; ++i) {
        if (tid == 0) {
            int t = draft_tokens[b * T + i];
            float p = draft_probs[(size_t)b * T * V + (size_t)i * V + t];
            float q = target_probs[(size_t)b * T * V + (size_t)i * V + t];
            float alpha = fminf(1.0f, q / p);
            float u = uniform_samples[b * (T + 1) + i];

            if (u < alpha) {
                output_tokens[b * (T + 1) + i] = t;
                s_accepted += 1;
            } else {
                s_rejected = 1;
                s_rej_pos = i;
            }
        }
        __syncthreads();
        if (s_rejected)
            break;
    }

    const int pos = s_rejected ? s_rej_pos : T; // where we sample from
    const float u = uniform_samples[b * (T + 1) + (s_rejected ? T : T)];
    // note: spec uses u[b, T] for both the resample and the bonus token
    __syncthreads();

    // ---- build the sampling distribution ----
    // reject  -> adj(v) = max(0, q - p) at position rej_pos, normalized
    // accept  -> q at position T-1 (bonus token), already normalized
    const size_t base =
        (size_t)b * T * V + (size_t)(s_rejected ? pos : T - 1) * V;

    if (s_rejected) {
        float local = 0.0f;
        for (int v = tid; v < V; v += blockDim.x) {
            float d = target_probs[base + v] - draft_probs[base + v];
            local += fmaxf(0.0f, d);
        }
        s_block[tid] = local;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s)
                s_block[tid] += s_block[tid + s];
            __syncthreads();
        }
        if (tid == 0)
            s_total = s_block[0];
    } else {
        if (tid == 0)
            s_total = 1.0f; // q is already normalized
    }
    __syncthreads();

    const bool uniform_fallback = s_rejected && (s_total <= 0.0f);

    // ---- inverse CDF: find smallest k with cumsum(k) > u ----
    // single-pass scan by thread 0 is too slow for V=131k; use a
    // block-strided two-phase approach: per-chunk partial sums, then
    // locate the chunk, then scan within it.
    if (tid == 0)
        s_pick = V - 1;
    __syncthreads();

    if (uniform_fallback) {
        if (tid == 0) {
            int k = (int)(u * V);
            s_pick = min(max(k, 0), V - 1);
        }
    } else {
        // phase 1: each thread sums its strided slice -> chunked prefix
        // simpler: partition V into THREADS contiguous chunks
        const int chunk = (V + THREADS - 1) / THREADS;
        const int lo = tid * chunk;
        const int hi = min(lo + chunk, V);

        float local = 0.0f;
        for (int v = lo; v < hi; ++v) {
            float pv =
                s_rejected
                    ? fmaxf(
                          0.0f, target_probs[base + v] - draft_probs[base + v]
                      )
                    : target_probs[base + v];
            local += pv;
        }
        s_block[tid] = local;
        __syncthreads();

        // exclusive prefix over chunk sums (serial, THREADS=256 entries)
        if (tid == 0) {
            float run = 0.0f;
            for (int c = 0; c < THREADS; ++c) {
                float cur = s_block[c];
                s_block[c] = run;
                run += cur;
            }
        }
        __syncthreads();

        // phase 2: the owning chunk scans locally
        const float target = u * s_total;
        float run = s_block[tid];
        if (lo < V) {
            for (int v = lo; v < hi; ++v) {
                float pv =
                    s_rejected
                        ? fmaxf(
                              0.0f,
                              target_probs[base + v] - draft_probs[base + v]
                          )
                        : target_probs[base + v];
                run += pv;
                if (run > target) {
                    atomicMin(&s_pick, v);
                    break;
                }
            }
        }
    }
    __syncthreads();

    // ---- write the sampled token and zero-pad ----
    if (tid == 0) {
        int write_pos = s_rejected ? pos : T;
        output_tokens[b * (T + 1) + write_pos] = s_pick;
        for (int i = write_pos + 1; i < T + 1; ++i)
            output_tokens[b * (T + 1) + i] = 0;
    }
}

extern "C" void solve_ans0(
    const int *draft_tokens,
    const float *draft_probs,
    const float *target_probs,
    const float *uniform_samples,
    int *output_tokens,
    int B,
    int T,
    int V
) {
    cudaMemset(output_tokens, 0, (size_t)B * (T + 1) * sizeof(int));
    verify_kernel<<<B, THREADS>>>(
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
