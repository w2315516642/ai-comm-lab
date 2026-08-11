#pragma once
#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>

#include <string>
#include <utility>
#include <vector>

namespace profiling {
namespace bandwidth {
namespace common {

#define CHECK(cmd)                                                             \
    do {                                                                       \
        cudaError_t e = cmd;                                                   \
        if (e != cudaSuccess) {                                                \
            printf("CUDA error: %s\n", cudaGetErrorString(e));                 \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

constexpr size_t SIZE_MB = 1024 * 1024;
constexpr size_t SIZE_GB = 1024 * SIZE_MB;

class BandwidthTest {
  public:
    static std::vector<size_t> makeTestData(
        int min_mb, int max_mb, int stripe
    ) {
        std::vector<size_t> data;
        for (int i = min_mb; i <= max_mb; i *= stripe) {
            data.push_back(i * SIZE_MB);
        }
        return data;
    }

    struct Result {
        size_t bytes{0};
        float ms{0.0f};
        float bw_gbs{0.0f};
    };

    template <typename Func>
    static Result run(Func &&fn, size_t bytes, int iter) {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        Result result{};

        float ms;
        for (int i = 0; i < iter; ++i) {
            ms = 0.0f;
            CHECK(cudaEventRecord(start, 0));
            fn();
            CHECK(cudaEventRecord(stop, 0));
            CHECK(cudaEventSynchronize(stop));
            CHECK(cudaEventElapsedTime(&ms, start, stop));
            result.bytes += bytes;
            result.ms += ms;
        }
        result.bw_gbs = result.bytes / (result.ms / 1000.0f) / SIZE_GB;

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));

        return result;
    }
};

} // namespace common
} // namespace bandwidth
} // namespace profiling
