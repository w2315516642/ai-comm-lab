#include <cmath>
#include <cuda_runtime.h>

#define THREAD_PER_BLOCK 256

__global__ void spec_dec_v3_kernel(
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

    __shared__ int reject_pos;    // 被拒绝的位置
    __shared__ int sampled_token; // 采样 token
    __shared__ int chunk_idx;     // 对 V 分段后，看目标采样 token 落在哪个段内
    __shared__ float chunked_sum[NUM_WARPS]; // 每个 warp 的段和

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
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    int pos = is_reject ? reject_pos : T - 1;
    int seq_ptr = (b_ptr + pos) * V;
    int num_v_per_warp = (V + NUM_WARPS - 1) / NUM_WARPS;
    int lo = min(warp_id * num_v_per_warp, V);
    int hi = min((warp_id + 1) * num_v_per_warp, V);
    // 1. 每个warp计算自己段的部分和
    float local_sum = 0.0f;
    for (int i = lo + lane_id; i < hi; i += 32) {
        float diff =
            is_reject ? (target_probs[seq_ptr + i] - draft_probs[seq_ptr + i])
                      : target_probs[seq_ptr + i];
        local_sum += fmax(0.0f, diff);
    }

    // 2. 每个 warp 计算 warp 内总和
    for (int offset = 16; offset > 0; offset >>= 1) {
        // 用 xor 进行蝶形规约，算完后每个 thread 都有段和
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    }
    if (lane_id == 0)
        chunked_sum[warp_id] = local_sum;
    __syncthreads();

    float exclusive_scan = 0.0f; // 以 warp 为单位的 exclusive_scan
    for (int i = 0; i < warp_id; i++) {
        exclusive_scan += chunked_sum[i];
    }
    float inclusive_scan = exclusive_scan + local_sum;

    // 3. 求总和
    bool mask = lane_id < NUM_WARPS;
    float sum = mask ? chunked_sum[lane_id]
                     : 0.0f; // 每个 warp 的前 NUM_WARPS 个线程取和

    for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    // 广播一下，所有线程都有总和了
    sum = __shfl_sync(0xffffffff, sum, 0);

    // 4. fallback 到均匀采样
    bool uniform_fallback = is_reject && (sum <= 0.0f);
    if (uniform_fallback) {
        // 整个 seq fallback 到均匀采样
        if (tid == 0) {
            float u = uniform_samples[u_ptr + T];
            int tmp = (int)(u * V);
            sampled_token = min(max(0, tmp), V - 1);
        }
        // 5. 找阈值
        // 每个线程有一个 local_sum：warp 段和，以及 sum：总和
    } else {
        // 找出目标 token 在哪个 warp 段里面
        // seq_ptr 是这个 token probs 的起始位置
        float u = uniform_samples[u_ptr + T] * sum;
        if (lane_id == 0 && u <= inclusive_scan) {
            atomicMin(&chunk_idx, warp_id);
        }
        __syncthreads();

        // 6. 先按 lane 求段和，找到 u 落在哪个 lane 段内，然后这个 lane
        // 单独扫描
        if (warp_id == chunk_idx) {

            float lane_sum = 0.0f;
            int num_v_per_lane = (hi - lo + 31) / 32;
            int lane_lo = min(hi, lo + num_v_per_lane * lane_id);
            int lane_hi = min(hi, lo + num_v_per_lane * (lane_id + 1));
            for (int i = lane_lo; i < lane_hi; i++) {
                float diff = is_reject ? target_probs[seq_ptr + i] -
                                             draft_probs[seq_ptr + i]
                                       : target_probs[seq_ptr + i];
                lane_sum += fmax(0.0f, diff);
            }

            for (int offset = 1; offset < 32; offset <<= 1) {
                float n = __shfl_up_sync(0xffffffff, lane_sum, offset);
                if (lane_id >= offset)
                    lane_sum += n;
            }

            lane_sum += exclusive_scan;
            float up = __shfl_up_sync(0xffffffff, lane_sum, 1);
            float lane_ex_scan = lane_id == 0 ? exclusive_scan : up;

            bool hit = u <= lane_sum || lane_id == 31;
            unsigned hit_mask = __ballot_sync(0xffffffff, hit);
            int hit_lane = __ffs(hit_mask) - 1;
            // __syncthreads();    // 不能放在分支里，有的 warp
            // 走不到这里会卡死！！！
            if (lane_id == hit_lane) {
                for (int i = lane_lo; i < lane_hi; i++) {
                    float diff = is_reject ? target_probs[seq_ptr + i] -
                                                 draft_probs[seq_ptr + i]
                                           : target_probs[seq_ptr + i];
                    lane_ex_scan += fmax(0.0f, diff);
                    if (lane_ex_scan >= u) {
                        sampled_token = i;
                        break;
                    }
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
extern "C" void solve_v3(
    const int *draft_tokens,
    const float *draft_probs,
    const float *target_probs,
    const float *uniform_samples,
    int *output_tokens,
    int B,
    int T,
    int V
) {
    spec_dec_v3_kernel<<<B, THREAD_PER_BLOCK>>>(
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
