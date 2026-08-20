#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define ELEMENTS_PER_BLOCK (2 * BLOCK_SIZE)
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

__global__ void scanBlockKernel(
    const float *input, float *output, float *blockSums, int n
) {
    extern __shared__ float temp[];
    int tid = threadIdx.x;
    int blockOffset = blockIdx.x * ELEMENTS_PER_BLOCK;

    int ai = tid, bi = tid + BLOCK_SIZE;
    int oA = CONFLICT_FREE_OFFSET(ai), oB = CONFLICT_FREE_OFFSET(bi);

    temp[ai + oA] = (blockOffset + ai < n) ? input[blockOffset + ai] : 0.0f;
    temp[bi + oB] = (blockOffset + bi < n) ? input[blockOffset + bi] : 0.0f;

    int offset = 1;
    for (int d = ELEMENTS_PER_BLOCK >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int a = offset * (2 * tid + 1) - 1;
            int b = offset * (2 * tid + 2) - 1;
            a += CONFLICT_FREE_OFFSET(a);
            b += CONFLICT_FREE_OFFSET(b);
            temp[b] += temp[a];
        }
        offset <<= 1;
    }

    if (tid == 0) {
        int last = ELEMENTS_PER_BLOCK - 1;
        last += CONFLICT_FREE_OFFSET(last);
        if (blockSums)
            blockSums[blockIdx.x] = temp[last];
        temp[last] = 0.0f;
    }

    for (int d = 1; d < ELEMENTS_PER_BLOCK; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int a = offset * (2 * tid + 1) - 1;
            int b = offset * (2 * tid + 2) - 1;
            a += CONFLICT_FREE_OFFSET(a);
            b += CONFLICT_FREE_OFFSET(b);
            float t = temp[a];
            temp[a] = temp[b];
            temp[b] += t;
        }
    }
    __syncthreads();

    // exclusive scan 结果 + 原值 = inclusive scan
    if (blockOffset + ai < n)
        output[blockOffset + ai] = temp[ai + oA] + input[blockOffset + ai];
    if (blockOffset + bi < n)
        output[blockOffset + bi] = temp[bi + oB] + input[blockOffset + bi];
}

// blockOffsets 已经是“每个 block 之前所有 block 的总和”(exclusive)
__global__ void addOffsetsKernel(
    float *output, const float *blockOffsets, int n
) {
    int blockOffset = blockIdx.x * ELEMENTS_PER_BLOCK;
    float add = blockOffsets[blockIdx.x];
    int ai = blockOffset + threadIdx.x;
    int bi = blockOffset + threadIdx.x + BLOCK_SIZE;
    if (ai < n)
        output[ai] += add;
    if (bi < n)
        output[bi] += add;
}

// 对长度 n 的 block sums 做 EXCLUSIVE scan，写入 out
static void scanExclusiveBlockSums(const float *in, float *out, int n);

// 主递归：inclusive scan，结果写 output
static void scanInclusive(const float *input, float *output, int n) {
    int numBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    size_t sh = (ELEMENTS_PER_BLOCK +
                 CONFLICT_FREE_OFFSET(ELEMENTS_PER_BLOCK - 1) + 1) *
                sizeof(float);

    if (numBlocks == 1) {
        scanBlockKernel<<<1, BLOCK_SIZE, sh>>>(input, output, nullptr, n);
        return;
    }

    float *blockSums, *blockOffsets;
    cudaMalloc(&blockSums, numBlocks * sizeof(float));
    cudaMalloc(&blockOffsets, numBlocks * sizeof(float));

    scanBlockKernel<<<numBlocks, BLOCK_SIZE, sh>>>(input, output, blockSums, n);

    // 对 blockSums 做 EXCLUSIVE scan -> blockOffsets
    scanExclusiveBlockSums(blockSums, blockOffsets, numBlocks);

    addOffsetsKernel<<<numBlocks, BLOCK_SIZE>>>(output, blockOffsets, n);

    cudaFree(blockSums);
    cudaFree(blockOffsets);
}

// 把 inclusive scan 转 exclusive：exclusive[i] = inclusive[i] - in[i]
__global__ void inclusiveToExclusive(
    const float *in, float *incl, float *excl, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        excl[i] = incl[i] - in[i];
}

static void scanExclusiveBlockSums(const float *in, float *out, int n) {
    // 先做 inclusive，再转 exclusive
    float *incl;
    cudaMalloc(&incl, n * sizeof(float));
    scanInclusive(in, incl, n);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    inclusiveToExclusive<<<blocks, threads>>>(in, incl, out, n);
    cudaFree(incl);
}

extern "C" void solve(const float *input, float *output, int N) {
    if (N <= 0)
        return;
    scanInclusive(input, output, N);
}
