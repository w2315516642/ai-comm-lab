#include <cmath>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

__global__ void spec_dec_kernel(
    const int *__restrict__ draft_tokens,
    const float *__restrict__ draft_probs,
    const float *__restrict__ target_probs,
    const float *__restrict__ uniform_samples,
    int *output_tokens,
    int B,
    int T,
    int V
) {
    // 判断第一个被拒绝的 token
    const int num_warps = (THREAD_PER_BLOCK + 31) / 32;
    int tid = threadIdx.x;
    int b_ptr = blockIdx.x * T;
    int u_ptr = blockIdx.x * (T + 1);
    extern __shared__ int s[];
    float *val = (float *)s;
    int *rej_id = (int *)&val[num_warps];
    int *sampled_token = (int *)&rej_id[1];
    bool *acc = (bool *)&sampled_token[1];

    if (tid < T) {
        acc[tid] = false;
        // 1. 找到 draft token
        int dtoken_id = draft_tokens[b_ptr + tid];
        // 2. 计算比值
        int qp_idx = (b_ptr + tid) * V + dtoken_id;
        float qpr = target_probs[qp_idx] / draft_probs[qp_idx];
        // 3. 每个 b 取最小值
        float rate = fmin(1.0f, qpr);
        // 4. 查看是否接受
        acc[tid] = uniform_samples[u_ptr + tid] < rate;
    }
    __syncthreads();
    // 找到第一个被拒绝的
    if (tid == 0) {
        rej_id[0] = T;
        for (int i = 0; i < T; i++) {
            if (acc[i] == false) {
                rej_id[0] = i;
                break;
            }
        }
    }
    __syncthreads();

    // 对被拒绝的那个进行重采样，否则直接对第 T+1 个进行额外采样
    bool is_reject = rej_id[0] < T;
    float local = 0.0f;

    // 计算概率和（包括被拒绝时的非归一化情况）
    int seq_ptr = (b_ptr + rej_id[0]) * V;
    for (int i = tid; i < V; i += blockDim.x) {
        float d_probs = is_reject ? draft_probs[seq_ptr + i] : 0.0f;
        float diff = target_probs[seq_ptr + i] - d_probs;
        local += fmax(0.0f, diff);
    }
    //__syncthreads();

    for (int offset = 16; offset >= 1; offset >>= 1) {
        local += __shfl_down_sync(0xffffffff, local, offset);
    }
    const int warp_id = tid / 32;
    // 每个 warp 求和的结果
    if (tid % 32 == 0) {
        val[warp_id] = local;
    }
    __syncthreads();

    if (warp_id == 0) {
        if (tid < num_warps) {
            local = val[tid];
        }
        for (int offset = num_warps / 2; offset >= 1; offset >>= 1) {
            local += __shfl_down_sync(0xffffffff, local, offset);
        }
        if (tid == 0) {
            val[0] = local;
        }
    }
    __syncthreads();
    // 算完和了

    local = val[0];
    bool uniform_fallback = is_reject && (local <= 0.0f);

    if (uniform_fallback) {
        // 整个 seq fallback 到均匀采样
        if (tid == 0) {
            float u = uniform_samples[u_ptr + T];
            int tmp = (int)(u * V);
            sampled_token[0] = min(max(0, tmp), V - 1);
        }
    } else {
        int sampled_pos = is_reject ? rej_id[0] : T - 1;
        seq_ptr = (b_ptr + sampled_pos) * V;
        if (tid == 0) {
            float prefix = 0;
            float u = uniform_samples[u_ptr + T] * local;
            for (int i = 0; i < V; i++) {
                float d_probs = is_reject ? draft_probs[seq_ptr + i] : 0.0f;
                float diff = target_probs[seq_ptr + i] - d_probs;
                prefix += fmax(0.0f, diff);
                if (prefix >= u) {
                    sampled_token[0] = i;
                    break;
                }
            }
        }
    }

    // 把数据搬进 output_tokens
    if (tid < T + 1) {
        int t_idx = u_ptr + tid;
        output_tokens[t_idx] = tid < rej_id[0] ? draft_tokens[b_ptr + tid] : 0;
        bool mask = (tid == rej_id[0]);
        output_tokens[t_idx] = mask ? sampled_token[0] : output_tokens[t_idx];
    }
}

// draft_tokens, draft_probs, target_probs, u niform_samples, output_tokens are
// device pointers
extern "C" void solve_v0(
    const int *draft_tokens,
    const float *draft_probs,
    const float *target_probs,
    const float *uniform_samples,
    int *output_tokens,
    int B,
    int T,
    int V
) {
    int size = sizeof(int) + sizeof(bool) * T +
               sizeof(float) * ((THREAD_PER_BLOCK + 31) / 32);
    spec_dec_kernel<<<B, THREAD_PER_BLOCK, size>>>(
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
