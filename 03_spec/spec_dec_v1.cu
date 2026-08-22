#include <cmath>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

__global__ void spec_dec_v1_kernel(
    const int *__restrict__ draft_tokens,
    const float *__restrict__ draft_probs,
    const float *__restrict__ target_probs,
    const float *__restrict__ uniform_samples,
    int *output_tokens,
    int B,
    int T,
    int V
) {
    constexpr int NUM_WARPS = (THREAD_PER_BLOCK + 31) / 32;
    // 判断第一个被拒绝的 token
    int tid = threadIdx.x;
    int b_ptr = blockIdx.x * T;
    int u_ptr = blockIdx.x * (T + 1);

    __shared__ int reject_pos;
    __shared__ int sampled_token;
    __shared__ int chunk_idx;
    __shared__ float chunked_sum[NUM_WARPS];
    __shared__ float sum;

    if (tid == 0) {
        reject_pos = T;
        chunk_idx = THREAD_PER_BLOCK;
        sampled_token = 0;
    }

    __syncthreads();

    // tokens 数量需要小于线程数，不然这里有 bug
    if (tid < T) {
        // 1. 找到 draft token
        int dtoken_id = draft_tokens[b_ptr + tid];
        // 2. 计算比值
        int qp_idx = (b_ptr + tid) * V + dtoken_id;
        float qpr = target_probs[qp_idx] / draft_probs[qp_idx];
        // 3. 每个 b 取最小值
        float rate = fmin(1.0f, qpr);
        // 4. 查看是否拒绝
        if (uniform_samples[u_ptr + tid] >= rate) {
            atomicMin(&reject_pos, tid);
        }
    }
    __syncthreads();

    // 对被拒绝的那个进行重采样，否则直接对第 T+1 个进行额外采样
    bool is_reject = reject_pos < T;

    // 计算概率和
    float local_sum = 0.0f;
    int pos = is_reject ? reject_pos : T - 1;
    int seq_ptr = (b_ptr + pos) * V;
    int num_v_per_thread = (V + THREAD_PER_BLOCK - 1) / THREAD_PER_BLOCK;
    int lo = tid * num_v_per_thread;
    int hi = min((tid + 1) * num_v_per_thread, V);
    // 1. 每个线程计算自己段的部分和
    for (int i = lo; i < hi; i++) {
        float diff =
            is_reject ? (target_probs[seq_ptr + i] - draft_probs[seq_ptr + i])
                      : target_probs[seq_ptr + i];
        local_sum += fmax(0.0f, diff);
    }

    // 2. 每个 warp 内的线程算 warp 内前缀段和
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    for (int offset = 1; offset < 32; offset <<= 1) {
        // 获取前第 offset 位线程发射过来的段和
        float n = __shfl_up_sync(0xffffffff, local_sum, offset);
        if (lane_id >= offset)
            local_sum += n;
    }
    // 3. 每个 warp 最后一个线程持有 warp 内总和
    if (lane_id == 31)
        chunked_sum[warp_id] = local_sum;
    __syncthreads();

    // 4. 算前面 warp 的总和
    float carry = 0.0f;
    for (int i = 0; i < warp_id; i++) {
        carry += chunked_sum[i];
    }
    float inclusive_scan = carry + local_sum;
    // 5. 拿前一个线程的总和当 exclusive_scan 结果
    float up = __shfl_up_sync(0xffffffff, inclusive_scan, 1);
    float exclusive_scan = lane_id == 0 ? carry : up;

    // 计算完毕，每个线程取总和
    if (tid == THREAD_PER_BLOCK - 1) {
        sum = inclusive_scan;
    }
    __syncthreads();
    local_sum = sum;

    bool uniform_fallback = is_reject && (local_sum <= 0.0f);

    if (uniform_fallback) {
        // 整个 seq fallback 到均匀采样
        if (tid == 0) {
            float u = uniform_samples[u_ptr + T];
            int tmp = (int)(u * V);
            sampled_token = min(max(0, tmp), V - 1);
        }
    } else {
        // 找出目标 token 在哪个段里面
        // seq_ptr 是这个 token probs 的起始位置
        float u = uniform_samples[u_ptr + T] * local_sum;
        if (u <= inclusive_scan) {
            atomicMin(&chunk_idx, tid);
        }
        __syncthreads();

        if (tid == chunk_idx) {
            float t_sum = exclusive_scan;
            for (int i = lo; i < hi; i++) {
                float diff = is_reject ? target_probs[seq_ptr + i] -
                                             draft_probs[seq_ptr + i]
                                       : target_probs[seq_ptr + i];
                t_sum += fmax(0.0f, diff);
                if (t_sum >= u) {
                    sampled_token = i;
                    break;
                }
            }
        }
    }
    __syncthreads();

    // 把数据搬进 output_tokens
    if (tid < T + 1) {
        int t_idx = u_ptr + tid;
        output_tokens[t_idx] = tid < reject_pos ? draft_tokens[b_ptr + tid] : 0;
        bool mask = (tid == reject_pos);
        output_tokens[t_idx] = mask ? sampled_token : output_tokens[t_idx];
    }
}

// draft_tokens, draft_probs, target_probs, u niform_samples, output_tokens are
// device pointers
extern "C" void solve_v1(
    const int *draft_tokens,
    const float *draft_probs,
    const float *target_probs,
    const float *uniform_samples,
    int *output_tokens,
    int B,
    int T,
    int V
) {
    spec_dec_v1_kernel<<<B, THREAD_PER_BLOCK>>>(
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
