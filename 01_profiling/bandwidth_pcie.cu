/*
 * PCIe 带宽测试
 * 用 cudaMemcpyAsync 测 Host <-> Device 单向传输
 */

#include "bandwidth_common.cuh"

using namespace profiling::bandwidth::common;

int main() {
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n\n", prop.name);

    auto sizes = BandwidthTest::makeTestData(1, 1024, 4);

    float *h_buf, *d_buf;
    CHECK(cudaMallocHost(&h_buf, sizes.back()));
    CHECK(cudaMalloc(&d_buf, sizes.back()));

    memset(h_buf, 0, sizes.back());

    // warmup
    CHECK(cudaMemcpyAsync(d_buf, h_buf, SIZE_MB, cudaMemcpyHostToDevice, 0));
    CHECK(cudaDeviceSynchronize());

    printf("%-12s | %-15s | %-15s\n", "Size", "H2D (GB/s)", "D2H (GB/s)");
    printf("-------------|-----------------|-----------------\n");

    for (size_t size : sizes) {
        auto fn_h2d = [&]() {
            CHECK(cudaMemcpyAsync(d_buf, h_buf, size,
                                 cudaMemcpyHostToDevice, 0));
        };
        auto r_h2d = BandwidthTest::run(fn_h2d, size, 10);

        auto fn_d2h = [&]() {
            CHECK(cudaMemcpyAsync(h_buf, d_buf, size,
                                 cudaMemcpyDeviceToHost, 0));
        };
        auto r_d2h = BandwidthTest::run(fn_d2h, size, 10);

        std::string size_str = (size >= SIZE_GB)
            ? std::to_string(size / SIZE_GB) + " GB"
            : std::to_string(size / SIZE_MB) + " MB";

        printf("%-12s | %-15.2f | %-15.2f\n",
               size_str.c_str(), r_h2d.bw_gbs, r_d2h.bw_gbs);
    }

    CHECK(cudaFreeHost(h_buf));
    CHECK(cudaFree(d_buf));
    return 0;
}
