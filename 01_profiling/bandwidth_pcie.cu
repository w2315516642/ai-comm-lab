/*
测量 PCIE 带宽
*/

#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdlib>
#include <string>
#include <vector>

#define CHECK(cmd)                                              \
    do {                                                        \
        cudaError_t e = cmd;                                    \
        if (e != cudaSuccess) {                                 \
            printf("CUDA error: %s\n", cudaGetErrorString(e));  \
            exit(1);                                            \
        }                                                       \
    } while (0)

constexpr size_t SIZE_MB = 1024 * 1024;
constexpr size_t SIZE_GB = 1024 * SIZE_MB;

int main() {
    std::vector<size_t> sizes{
        1 * SIZE_MB,     // 1 MB
        16 * SIZE_MB,    // 16 MB
        64 * SIZE_MB,    // 64 MB
        256 * SIZE_MB,   // 256 MB
        1024 * SIZE_MB,  // 1 GB
    };
    const int num_sizes = sizes.size();

    // Print GPU and PCIe query hint.
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("PCIe Gen: (check with: nvidia-smi --query-gpu=pcie.link.gen.max --format=csv)\n\n");

    // Allocate pinned host memory and device memory for the largest size.
    float *h_buf, *d_buf;
    CHECK(cudaMallocHost(&h_buf, sizes[num_sizes - 1]));
    CHECK(cudaMalloc(&d_buf, sizes[num_sizes - 1]));

    // warmup
    CHECK(cudaMemcpyAsync(d_buf, h_buf, SIZE_MB, cudaMemcpyHostToDevice, 0));
    CHECK(cudaDeviceSynchronize());

    printf("%-12s | %-15s | %-15s\n", "Size", "H2D (GB/s)", "D2H (GB/s)");
    printf("-------------|-----------------|-----------------\n");

    for (int i = 0; i < num_sizes; i++) {
        size_t n = sizes[i];
        cudaEvent_t start, stop;
        float ms;

        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));

        // Host -> Device
        CHECK(cudaEventRecord(start, 0));
        CHECK(cudaMemcpyAsync(d_buf, h_buf, n, cudaMemcpyHostToDevice, 0));
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        float bw_h2d = n / (ms / 1000.0f) / SIZE_GB;

        // Device -> Host
        CHECK(cudaEventRecord(start, 0));
        CHECK(cudaMemcpyAsync(h_buf, d_buf, n, cudaMemcpyDeviceToHost, 0));
        CHECK(cudaEventRecord(stop, 0));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        float bw_d2h = n / (ms / 1000.0f) / SIZE_GB;

        std::string size_str;
        if (n >= SIZE_GB) {
            size_str = std::to_string(n / SIZE_GB) + " GB";
        } else {
            size_str = std::to_string(n / SIZE_MB) + " MB";
        }

        printf("%-12s | %-15.2f | %-15.2f\n", size_str.c_str(), bw_h2d, bw_d2h);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    CHECK(cudaFreeHost(h_buf));
    CHECK(cudaFree(d_buf));
    return 0;
}
