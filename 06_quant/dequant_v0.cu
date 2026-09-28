#include <cuda_runtime.h>

__global__ void dequant_kernel_v0(
    const float *X, const float *S, float *Y, int M, int N, int TILE_SIZE
) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < M && j < N) {
        int tile_i = i / TILE_SIZE;
        int tile_j = j / TILE_SIZE;
        int num_tile_cols = (N + TILE_SIZE - 1) / TILE_SIZE;

        float scale = S[tile_i * num_tile_cols + tile_j];
        Y[i * N + j] = scale * X[i * N + j];
    }
}

// X, S, Y are device pointers
extern "C" void solve(
    const float *X, const float *S, float *Y, int M, int N, int TILE_SIZE
) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    dequant_kernel_v0<<<grid, block>>>(X, S, Y, M, N, TILE_SIZE);
}
