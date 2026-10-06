#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <map>
#include <tuple>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>

namespace cutlass_bench {

inline void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUTLASS %s: %s\n", operation, cudaGetErrorString(status));
        std::abort();
    }
}

inline void check_cutlass(cutlass::Status status, const char *operation) {
    if (status != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "CUTLASS %s: %s\n", operation, cutlass::cutlassGetStatusString(status));
        std::abort();
    }
}

struct Problem {
    const __nv_bfloat16 *a = nullptr;
    const __nv_bfloat16 *b = nullptr;
    float *c = nullptr;
    int m = 0, n = 0, k = 0;
    int device = -1, sm_count = 0, max_shared = 0;

    bool same_call(const Problem &other) const {
        return a == other.a && b == other.b && c == other.c &&
               m == other.m && n == other.n && k == other.k && device == other.device;
    }
};

// 架构文件只负责给出 Gemm 类型和参数；初始化、计时、缓存由这里复用。
struct Kernel {
    const char *name;
    explicit Kernel(const char *name) : name(name) {}
    virtual ~Kernel() = default;
    virtual bool prepare(const Problem &p) = 0;
    virtual void run() = 0;
};

template<class Backend>
class Candidate final : public Kernel {
    using Gemm = typename Backend::Gemm;
    Gemm gemm_;
    void *workspace_ = nullptr;
    size_t workspace_bytes_ = 0;
    int workspace_device_ = -1;

    void release_workspace() {
        if (!workspace_)
            return;
        // 允许调用者切换设备；释放在原设备上创建的 workspace。
        int current = -1;
        if (cudaGetDevice(&current) == cudaSuccess) {
            if (cudaSetDevice(workspace_device_) == cudaSuccess)
                cudaFree(workspace_);
            cudaSetDevice(current);
        }
        workspace_ = nullptr;
        workspace_bytes_ = 0;
    }

public:
    explicit Candidate(const char *name) : Kernel(name) {}
    ~Candidate() override { release_workspace(); }

    bool prepare(const Problem &p) override {
        if (sizeof(typename Gemm::GemmKernel::SharedStorage) > size_t(p.max_shared))
            return false;
        auto args = Backend::arguments(p);
        if (Gemm::can_implement(args) != cutlass::Status::kSuccess)
            return false;
        size_t bytes = Gemm::get_workspace_size(args);
        if (workspace_device_ != p.device || bytes > workspace_bytes_) {
            release_workspace();
            workspace_device_ = p.device;
            if (bytes) {
                check_cuda(cudaMalloc(&workspace_, bytes), "allocate workspace");
                workspace_bytes_ = bytes;
            }
        }
        check_cutlass(gemm_.initialize(args, workspace_), "initialize");
        return true;
    }

    void run() override { check_cutlass(gemm_.run(), name); }
};

class Autotuner {
    const char *architecture_;
    std::vector<Kernel *> kernels_;
    // 最优编号按设备与 M/N/K 缓存；指针变化只重建参数，不重新测速。
    std::map<std::tuple<int, int, int, int>, size_t> winners_;
    Problem active_;
    size_t selected_ = 0;

    size_t tune(const Problem &p) {
        constexpr int WARMUP = 5;
        constexpr int REPEATS = 20;
        constexpr int ROUNDS = 3;
        std::vector<size_t> valid;
        for (size_t i = 0; i < kernels_.size(); ++i) {
            if (kernels_[i]->prepare(p)) {
                valid.push_back(i);
                for (int j = 0; j < WARMUP; ++j)
                    kernels_[i]->run();
            }
        }
        if (valid.empty()) {
            std::fprintf(stderr, "CUTLASS: no candidate supports M=%d N=%d K=%d\n", p.m, p.n, p.k);
            std::abort();
        }

        cudaEvent_t begin, end;
        check_cuda(cudaEventCreate(&begin), "create tuning event");
        check_cuda(cudaEventCreate(&end), "create tuning event");
        std::vector<std::array<float, ROUNDS>> times(kernels_.size());
        // 每轮轮换候选顺序，用三轮中位数降低时钟波动和测量顺序的影响。
        for (int round = 0; round < ROUNDS; ++round) {
            for (size_t j = 0; j < valid.size(); ++j) {
                size_t i = valid[(j + round) % valid.size()];
                check_cuda(cudaEventRecord(begin), "start tuning");
                for (int repeat = 0; repeat < REPEATS; ++repeat)
                    kernels_[i]->run();
                check_cuda(cudaEventRecord(end), "stop tuning");
                check_cuda(cudaEventSynchronize(end), "wait tuning");
                float ms = 0;
                check_cuda(cudaEventElapsedTime(&ms, begin, end), "time tuning");
                times[i][round] = ms / REPEATS;
            }
        }
        check_cuda(cudaEventDestroy(begin), "destroy tuning event");
        check_cuda(cudaEventDestroy(end), "destroy tuning event");
        float best_ms = std::numeric_limits<float>::max();
        size_t best = valid.front();
        for (size_t i : valid) {
            std::sort(times[i].begin(), times[i].end());
            if (times[i][ROUNDS / 2] < best_ms) {
                best_ms = times[i][ROUNDS / 2];
                best = i;
            }
        }
        std::fprintf(stderr,
            "[CUTLASS %s] M=%d N=%d K=%d: %s (%.4f ms, %zu candidates)\n",
            architecture_, p.m, p.n, p.k, kernels_[best]->name, best_ms, valid.size());
        return best;
    }

public:
    Autotuner(const char *architecture, std::initializer_list<Kernel *> kernels)
        : architecture_(architecture), kernels_(kernels) {}

    void run(const __nv_bfloat16 *A, const __nv_bfloat16 *B, float *C, int M, int N, int K) {
        Problem p;
        p.a = A; p.b = B; p.c = C;
        p.m = M; p.n = N; p.k = K;
        check_cuda(cudaGetDevice(&p.device), "get device");
        if (!active_.same_call(p)) {
            check_cuda(cudaDeviceGetAttribute(&p.sm_count, cudaDevAttrMultiProcessorCount, p.device), "get SM count");
            check_cuda(cudaDeviceGetAttribute(&p.max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, p.device), "get shared limit");
            auto key = std::make_tuple(p.device, M, N, K);
            auto found = winners_.find(key);
            if (found == winners_.end()) {
                selected_ = tune(p);
                winners_[key] = selected_;
            } else {
                selected_ = found->second;
                if (!kernels_[selected_]->prepare(p)) {
                    std::fprintf(stderr, "CUTLASS: cached candidate cannot handle these pointers\n");
                    std::abort();
                }
            }
            active_ = p;
        }
        // 必须再执行被选中的内核，确保外层正确性检查验证的是它的结果。
        // 正式计时只执行此路径，不构造描述符、不分配 workspace、不重新调优。
        kernels_[selected_]->run();
    }
};

} // namespace cutlass_bench
