// Row-major GEMM benchmark: C[M, N] = A[M, K] * B[K, N].
//
// Usage:
//   ./bench_gemm [M] [N] [K] [iters] [warmup]

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

using SolveFn =
    void (*)(const float *A, const float *B, float *C, int M, int N, int K);
using SolveBf16Fn = void (*)(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    float *C,
    int M,
    int N,
    int K
);
using SupportsFn = bool (*)(int M, int N, int K);

void solve(const float *A, const float *B, float *C, int M, int N, int K);
void solve_v1(const float *A, const float *B, float *C, int M, int N, int K);
void solve_v3(const float *A, const float *B, float *C, int M, int N, int K);
void solve_v4(const float *A, const float *B, float *C, int M, int N, int K);
void solve_v5(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    float *C,
    int M,
    int N,
    int K
);
void solve_v6(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    float *C,
    int M,
    int N,
    int K
);
void solve_v7(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    float *C,
    int M,
    int N,
    int K
);
void solve_v8(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    float *C,
    int M,
    int N,
    int K
);
bool supports_v8(int M, int N, int K);
#ifdef ENABLE_GEMM_V9
void solve_v9(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    float *C,
    int M, int N, int K
);
bool supports_v9(int M, int N, int K);
#endif
#ifdef ENABLE_GEMM_V10
void solve_v10(
    const __nv_bfloat16 *A, const __nv_bfloat16 *B, float *C,
    int M, int N, int K
);
bool supports_v10(int M, int N, int K);
#endif
#ifdef ENABLE_GEMM_V11
void solve_v11(
    const __nv_bfloat16 *A, const __nv_bfloat16 *B, float *C,
    int M, int N, int K
);
bool supports_v11(int M, int N, int K);
#endif
#ifdef ENABLE_GEMM_V12
void solve_v12(
    const __nv_bfloat16 *A, const __nv_bfloat16 *B, float *C,
    int M, int N, int K
);
bool supports_v12(int M, int N, int K);
#endif
void solve_cutlass(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    float *C,
    int M,
    int N,
    int K
);

struct Impl {
    const char *name;
    SolveFn fp32_fn;
    SolveBf16Fn bf16_fn;
    SupportsFn supports;
};

static bool supports_all(int, int, int) { return true; }
static bool supports_v3(int M, int N, int K) {
    return M % 128 == 0 && N % 256 == 0 && K % 32 == 0;
}
static bool supports_v5(int M, int N, int K) {
    return M % 32 == 0 && N % 32 == 0 && K % 16 == 0;
}
static bool supports_cutlass_bf16(int, int N, int K) {
    return N % 8 == 0 && K % 8 == 0;
}

static const Impl kImpls[] = {
    {"v0-cuda", solve, nullptr, supports_all},
    {"v1-smem", solve_v1, nullptr, supports_all},
    {"v3-regtile", solve_v3, nullptr, supports_v3},
    {"v4-cpasync", solve_v4, nullptr, supports_v3},
    {"v5-bf16-wmma", nullptr, solve_v5, supports_v5},
    {"v6-bf16-async", nullptr, solve_v6, supports_v5},
    {"v7-ldmatrix", nullptr, solve_v7, supports_v5},
    {"v8-ldmatrix", nullptr, solve_v8, supports_v8},
#ifdef ENABLE_GEMM_V9
    {"v9-wgmma", nullptr, solve_v9, supports_v9},
#endif
#ifdef ENABLE_GEMM_V10
    {"v10-tma-wgmma", nullptr, solve_v10, supports_v10},
#endif
#ifdef ENABLE_GEMM_V11
    {"v11-tma-wgmma", nullptr, solve_v11, supports_v11},
#endif
#ifdef ENABLE_GEMM_V12
    {"v12-wg-pipeline", nullptr, solve_v12, supports_v12},
#endif
    {"cutlass-bf16", nullptr, solve_cutlass, supports_cutlass_bf16},
};
static constexpr int kNumImpls = sizeof(kImpls) / sizeof(kImpls[0]);

static void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static void fill_input(float *data, size_t n, unsigned *seed) {
    for (size_t i = 0; i < n; i++) {
        *seed = *seed * 1103515245u + 12345u;
        unsigned value = (*seed >> 8) & 0xffffffu;
        data[i] = (float)value / (float)(1 << 23) - 1.0f;
    }
}

static void reference_cpu(
    const float *A, const float *B, float *C, int M, int N, int K
) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++)
                sum +=
                    (double)A[(size_t)m * K + k] * (double)B[(size_t)k * N + n];
            C[(size_t)m * N + n] = (float)sum;
        }
    }
}

static void reference_cpu_bf16(
    const __nv_bfloat16 *A,
    const __nv_bfloat16 *B,
    float *C,
    int M,
    int N,
    int K
) {
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                float a = __bfloat162float(A[(size_t)m * K + k]);
                float b = __bfloat162float(B[(size_t)k * N + n]);
                sum += (double)a * (double)b;
            }
            C[(size_t)m * N + n] = (float)sum;
        }
    }
}

static bool check_result(
    const char *name,
    const float *expected,
    const float *actual,
    int M,
    int N,
    int K
) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    int worst = 0;

    for (int i = 0; i < M * N; i++) {
        float abs_err = fabsf(actual[i] - expected[i]);
        float rel_err = abs_err / fmaxf(fabsf(expected[i]), 1e-6f);
        if (abs_err > max_abs) {
            max_abs = abs_err;
            worst = i;
        }
        max_rel = fmaxf(max_rel, rel_err);

        // The reference accumulates in double while the kernel accumulates in
        // float, so use a tolerance that grows mildly with K.
        float tol = 2e-5f * K + 2e-4f * fabsf(expected[i]);
        if (!isfinite(actual[i]) || abs_err > tol) {
            fprintf(
                stderr,
                "%s mismatch: row=%d col=%d expected=%.8g actual=%.8g "
                "abs_err=%.3g tol=%.3g\n",
                name,
                i / N,
                i % N,
                expected[i],
                actual[i],
                abs_err,
                tol
            );
            return false;
        }
    }

    printf(
        "%s check: max_abs=%.3g max_rel=%.3g worst=(%d,%d)\n",
        name,
        max_abs,
        max_rel,
        worst / N,
        worst % N
    );
    return true;
}

static void launch_impl(
    const Impl &impl,
    const float *d_A,
    const float *d_B,
    const __nv_bfloat16 *d_A_bf16,
    const __nv_bfloat16 *d_B_bf16,
    float *d_C,
    int M,
    int N,
    int K
) {
    if (impl.bf16_fn)
        impl.bf16_fn(d_A_bf16, d_B_bf16, d_C, M, N, K);
    else
        impl.fp32_fn(d_A, d_B, d_C, M, N, K);
}

static float bench_one(
    const Impl &impl,
    const float *d_A,
    const float *d_B,
    const __nv_bfloat16 *d_A_bf16,
    const __nv_bfloat16 *d_B_bf16,
    float *d_C,
    int M,
    int N,
    int K,
    int iters,
    int warmup
) {
    for (int i = 0; i < warmup; i++)
        launch_impl(impl, d_A, d_B, d_A_bf16, d_B_bf16, d_C, M, N, K);
    check_cuda(cudaGetLastError(), "warmup launch");
    check_cuda(cudaDeviceSynchronize(), "warmup execution");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iters; i++)
        launch_impl(impl, d_A, d_B, d_A_bf16, d_B_bf16, d_C, M, N, K);
    check_cuda(cudaGetLastError(), "benchmark launch");
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "benchmark execution");

    float total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "elapsed time");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return total_ms / iters;
}

int main(int argc, char **argv) {
    int M = 256, N = 256, K = 256, iters = 100, warmup = 10;
    if (argc > 1)
        M = atoi(argv[1]);
    if (argc > 2)
        N = atoi(argv[2]);
    if (argc > 3)
        K = atoi(argv[3]);
    if (argc > 4)
        iters = atoi(argv[4]);
    if (argc > 5)
        warmup = atoi(argv[5]);

    if (M <= 0 || N <= 0 || K <= 0 || iters <= 0 || warmup < 0) {
        fprintf(stderr, "require M,N,K,iters > 0 and warmup >= 0\n");
        return 1;
    }

    const size_t a_count = (size_t)M * K;
    const size_t b_count = (size_t)K * N;
    const size_t c_count = (size_t)M * N;
    const size_t a_bytes = a_count * sizeof(float);
    const size_t b_bytes = b_count * sizeof(float);
    const size_t c_bytes = c_count * sizeof(float);

    float *h_A = new float[a_count];
    float *h_B = new float[b_count];
    __nv_bfloat16 *h_A_bf16 = new __nv_bfloat16[a_count];
    __nv_bfloat16 *h_B_bf16 = new __nv_bfloat16[b_count];
    float *h_ref = new float[c_count];
    float *h_ref_bf16 = new float[c_count];
    float *h_out = new float[c_count];
    unsigned seed = 12345;
    fill_input(h_A, a_count, &seed);
    fill_input(h_B, b_count, &seed);
    for (size_t i = 0; i < a_count; i++)
        h_A_bf16[i] = __float2bfloat16_rn(h_A[i]);
    for (size_t i = 0; i < b_count; i++)
        h_B_bf16[i] = __float2bfloat16_rn(h_B[i]);
    reference_cpu(h_A, h_B, h_ref, M, N, K);
    reference_cpu_bf16(h_A_bf16, h_B_bf16, h_ref_bf16, M, N, K);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    __nv_bfloat16 *d_A_bf16 = nullptr, *d_B_bf16 = nullptr;
    check_cuda(cudaMalloc(&d_A, a_bytes), "cudaMalloc A");
    check_cuda(cudaMalloc(&d_B, b_bytes), "cudaMalloc B");
    check_cuda(cudaMalloc(&d_C, c_bytes), "cudaMalloc C");
    check_cuda(
        cudaMalloc(&d_A_bf16, a_count * sizeof(__nv_bfloat16)),
        "cudaMalloc BF16 A"
    );
    check_cuda(
        cudaMalloc(&d_B_bf16, b_count * sizeof(__nv_bfloat16)),
        "cudaMalloc BF16 B"
    );
    check_cuda(cudaMemcpy(d_A, h_A, a_bytes, cudaMemcpyHostToDevice), "copy A");
    check_cuda(cudaMemcpy(d_B, h_B, b_bytes, cudaMemcpyHostToDevice), "copy B");
    check_cuda(
        cudaMemcpy(
            d_A_bf16,
            h_A_bf16,
            a_count * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice
        ),
        "copy BF16 A"
    );
    check_cuda(
        cudaMemcpy(
            d_B_bf16,
            h_B_bf16,
            b_count * sizeof(__nv_bfloat16),
            cudaMemcpyHostToDevice
        ),
        "copy BF16 B"
    );

    double operations = 2.0 * (double)M * N * K;
    bool all_correct = true;

    printf("impl          correct   avg_ms      GFLOP/s\n");
    printf("------------------------------------------------\n");
    for (int impl = 0; impl < kNumImpls; impl++) {
        if (!kImpls[impl].supports(M, N, K)) {
            printf(
                "%-13s %-9s %-11s %10s\n", kImpls[impl].name, "SKIP", "-", "-"
            );
            continue;
        }

        check_cuda(cudaMemset(d_C, 0xa5, c_bytes), "memset C");
        launch_impl(kImpls[impl], d_A, d_B, d_A_bf16, d_B_bf16, d_C, M, N, K);
        check_cuda(cudaGetLastError(), "correctness launch");
        check_cuda(cudaDeviceSynchronize(), "correctness execution");
        check_cuda(
            cudaMemcpy(h_out, d_C, c_bytes, cudaMemcpyDeviceToHost), "copy C"
        );
        const float *expected = kImpls[impl].bf16_fn ? h_ref_bf16 : h_ref;
        bool correct =
            check_result(kImpls[impl].name, expected, h_out, M, N, K);
        all_correct = all_correct && correct;

        float ms = bench_one(
            kImpls[impl],
            d_A,
            d_B,
            d_A_bf16,
            d_B_bf16,
            d_C,
            M,
            N,
            K,
            iters,
            warmup
        );
        double gflops = operations / ((double)ms * 1e6);
        printf(
            "%-13s %-9s %-11.4f %10.2f\n",
            kImpls[impl].name,
            correct ? "PASS" : "FAIL",
            ms,
            gflops
        );
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_A_bf16);
    cudaFree(d_B_bf16);
    delete[] h_A;
    delete[] h_B;
    delete[] h_A_bf16;
    delete[] h_B_bf16;
    delete[] h_ref;
    delete[] h_ref_bf16;
    delete[] h_out;
    return all_correct ? 0 : 2;
}
