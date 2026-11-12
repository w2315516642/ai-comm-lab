#include <cuda_runtime.h>

__global__ void gemv_kernel(
    const float *__restrict__ A, // [M, N]
    const float *__restrict__ b, // [N]
    float *__restrict__ x,       // [M]
    int M, int N
) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x;

    constexpr int WARP_NUM = 4;
    constexpr int TILE_SIZE = 128;
    constexpr int WARP_SIZE = 32;
    __shared__ float smem[WARP_NUM];

    // 1. 搬数据
    float acc = 0.0f;
    for (int col = tid; col < N; col += TILE_SIZE) {
        acc += A[row * N + col] * b[col];
    }

    // 2. warp 内规约
    for (int offset = WARP_SIZE >> 2; offset >= 1; offset >>= 2) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    // 3. warp 间规约
    int lane_id = tid & 31;
    int warp_id = tid >> 5;
    if (lane_id == 0)
        smem[warp_id] = acc;
    __syncthreads();

    float sum = 0.0f;
    if (tid < WARP_NUM) {
        // for (int i = 0; i < WARP_NUM; i++) {
        //     sum += smem[i];
        // }
        // x[row] = sum;
        acc = smem[tid];
        for (int offset = WARP_NUM >> 2; offset >= 1; offset >>= 2) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }

        if (tid == 0) {
            x[row] = acc;
        }
    }
}

extern "C" void solve() {}
