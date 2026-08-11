/*
 * HBM (GPU 显存) 带宽测试
 * 用 copy kernel: dst[i] = src[i]，有效带宽 = 读 + 写 = size * 2
 */

#include "bandwidth_common.cuh"

using namespace profiling::bandwidth::common;

constexpr int THREADS = 256;

__global__ void copy_kernel(
    const float *__restrict__ src, float *__restrict__ dst, size_t n
) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx];
}

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Theoretical BW: %.0f GB/s\n\n",
           prop.memoryClockRate * (prop.memoryBusWidth / 8) * 2.0 / 1e6);

    auto sizes = BandwidthTest::makeTestData(64, 1024, 4);

    size_t max_elems = sizes.back() / sizeof(float);
    float *d_src, *d_dst;
    CHECK(cudaMalloc(&d_src, sizes.back()));
    CHECK(cudaMalloc(&d_dst, sizes.back()));

    std::vector<float> h_init(max_elems, 1.0f);
    CHECK(cudaMemcpy(d_src, h_init.data(), sizes.back(), cudaMemcpyHostToDevice));

    // warmup
    int max_blocks = (max_elems + THREADS - 1) / THREADS;
    copy_kernel<<<max_blocks, THREADS>>>(d_src, d_dst, max_elems);
    CHECK(cudaDeviceSynchronize());

    printf("%-12s | %-15s | %-15s\n", "Size", "Avg ms", "BW (GB/s)");
    printf("-------------|-----------------|-----------------\n");

    for (size_t size : sizes) {
        size_t elems = size / sizeof(float);
        int blocks = (elems + THREADS - 1) / THREADS;

        auto fn = [&]() {
            copy_kernel<<<blocks, THREADS>>>(d_src, d_dst, elems);
        };

        auto r = BandwidthTest::run(fn, size * 2, 10);   // read + write

        std::string size_str = (size >= SIZE_GB)
            ? std::to_string(size / SIZE_GB) + " GB"
            : std::to_string(size / SIZE_MB) + " MB";

        printf("%-12s | %-15.4f | %-15.2f\n",
               size_str.c_str(), r.ms / 10, r.bw_gbs);
    }

    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    return 0;
}
