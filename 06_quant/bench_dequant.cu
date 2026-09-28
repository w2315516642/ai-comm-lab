// Per-tile dequantization benchmark: Y[i, j] = X[i, j] * S[i/T, j/T].
//
// Usage:
//   ./bench_dequant [M] [N] [TILE_SIZE] [iters] [warmup]

// X and Y are row-major M x N matrices. S is a row-major
// ceil(M / TILE_SIZE) x ceil(N / TILE_SIZE) matrix.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

using SolveFn = void (*)(
    const float *X, const float *S, float *Y, int M, int N, int TILE_SIZE
);

extern "C" void solve(
    const float *X, const float *S, float *Y, int M, int N, int TILE_SIZE
);

struct Impl {
    const char *name;
    SolveFn fn;
};

static const Impl kImpls[] = {
    {"v0-baseline", solve},
};
static constexpr int kNumImpls = sizeof(kImpls) / sizeof(kImpls[0]);

static void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static void fill_input(float *data, size_t count, unsigned *seed) {
    for (size_t i = 0; i < count; i++) {
        *seed = *seed * 1103515245u + 12345u;
        const unsigned value = (*seed >> 8) & 0xffffffu;
        data[i] = (float)value / (float)(1 << 23) - 1.0f;
    }
}

static void fill_scales(float *data, size_t count, unsigned *seed) {
    for (size_t i = 0; i < count; i++) {
        *seed = *seed * 1103515245u + 12345u;
        const unsigned value = (*seed >> 8) & 0xffffffu;
        // Keep scales positive and away from zero so missed output writes are
        // easy to distinguish from valid results.
        data[i] = 0.01f + (float)value / (float)(1 << 24) * 1.99f;
    }
}

static void reference_cpu(
    const float *X, const float *S, float *Y, int M, int N, int tile_size
) {
    const int scale_cols = (N + tile_size - 1) / tile_size;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            const int scale_idx = (i / tile_size) * scale_cols + j / tile_size;
            Y[(size_t)i * N + j] = X[(size_t)i * N + j] * S[scale_idx];
        }
    }
}

static bool check_result(
    const char *name, const float *expected, const float *actual, int M, int N
) {
    float max_abs = 0.0f;
    size_t worst = 0;
    const size_t count = (size_t)M * N;

    for (size_t i = 0; i < count; i++) {
        const float abs_err = fabsf(actual[i] - expected[i]);
        if (abs_err > max_abs) {
            max_abs = abs_err;
            worst = i;
        }

        const float tol = 1e-6f * fmaxf(1.0f, fabsf(expected[i]));
        if (!isfinite(actual[i]) || abs_err > tol) {
            fprintf(
                stderr,
                "%s mismatch: row=%zu col=%zu expected=%.8g actual=%.8g "
                "abs_err=%.3g tol=%.3g\n",
                name,
                i / (size_t)N,
                i % (size_t)N,
                expected[i],
                actual[i],
                abs_err,
                tol
            );
            return false;
        }
    }

    printf(
        "%s check: max_abs=%.3g worst=(%zu,%zu)\n",
        name,
        max_abs,
        worst / (size_t)N,
        worst % (size_t)N
    );
    return true;
}

static float bench_one(
    SolveFn fn,
    const float *d_X,
    const float *d_S,
    float *d_Y,
    int M,
    int N,
    int tile_size,
    int iters,
    int warmup
) {
    for (int i = 0; i < warmup; i++)
        fn(d_X, d_S, d_Y, M, N, tile_size);
    check_cuda(cudaGetLastError(), "warmup launch");
    check_cuda(cudaDeviceSynchronize(), "warmup execution");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iters; i++)
        fn(d_X, d_S, d_Y, M, N, tile_size);
    check_cuda(cudaGetLastError(), "benchmark launch");
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "benchmark execution");

    float total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "elapsed time");
    check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
    return total_ms / iters;
}

int main(int argc, char **argv) {
    int M = 4096, N = 4096, tile_size = 128, iters = 200, warmup = 20;
    if (argc > 1)
        M = atoi(argv[1]);
    if (argc > 2)
        N = atoi(argv[2]);
    if (argc > 3)
        tile_size = atoi(argv[3]);
    if (argc > 4)
        iters = atoi(argv[4]);
    if (argc > 5)
        warmup = atoi(argv[5]);

    if (M <= 0 || N <= 0 || tile_size <= 0 || iters <= 0 || warmup < 0) {
        fprintf(stderr, "require M,N,TILE_SIZE,iters > 0 and warmup >= 0\n");
        return 1;
    }

    const size_t value_count = (size_t)M * N;
    const int scale_rows = (M + tile_size - 1) / tile_size;
    const int scale_cols = (N + tile_size - 1) / tile_size;
    const size_t scale_count = (size_t)scale_rows * scale_cols;
    const size_t value_bytes = value_count * sizeof(float);
    const size_t scale_bytes = scale_count * sizeof(float);

    float *h_X = new float[value_count];
    float *h_S = new float[scale_count];
    float *h_ref = new float[value_count];
    float *h_out = new float[value_count];
    unsigned seed = 12345;
    fill_input(h_X, value_count, &seed);
    fill_scales(h_S, scale_count, &seed);
    reference_cpu(h_X, h_S, h_ref, M, N, tile_size);

    float *d_X = nullptr, *d_S = nullptr, *d_Y = nullptr;
    check_cuda(cudaMalloc(&d_X, value_bytes), "cudaMalloc X");
    check_cuda(cudaMalloc(&d_S, scale_bytes), "cudaMalloc S");
    check_cuda(cudaMalloc(&d_Y, value_bytes), "cudaMalloc Y");
    check_cuda(
        cudaMemcpy(d_X, h_X, value_bytes, cudaMemcpyHostToDevice), "copy X"
    );
    check_cuda(
        cudaMemcpy(d_S, h_S, scale_bytes, cudaMemcpyHostToDevice), "copy S"
    );

    bool correct[kNumImpls] = {false};
    for (int i = 0; i < kNumImpls; i++) {
        check_cuda(cudaMemset(d_Y, 0xa5, value_bytes), "memset Y");
        kImpls[i].fn(d_X, d_S, d_Y, M, N, tile_size);
        check_cuda(cudaGetLastError(), "correctness launch");
        check_cuda(cudaDeviceSynchronize(), "correctness execution");
        check_cuda(
            cudaMemcpy(h_out, d_Y, value_bytes, cudaMemcpyDeviceToHost),
            "copy Y back"
        );
        correct[i] = check_result(kImpls[i].name, h_ref, h_out, M, N);
    }

    printf(
        "M=%d N=%d TILE_SIZE=%d scales=%dx%d iters=%d warmup=%d\n",
        M,
        N,
        tile_size,
        scale_rows,
        scale_cols,
        iters,
        warmup
    );
    printf("impl          correct   avg_ms       Gelems/s    GB/s\n");
    printf("-------------------------------------------------------\n");
    for (int i = 0; i < kNumImpls; i++) {
        const float ms = bench_one(
            kImpls[i].fn, d_X, d_S, d_Y, M, N, tile_size, iters, warmup
        );
        const double seconds = ms / 1000.0;
        const double gelems_per_s = value_count / seconds / 1e9;
        // Logical traffic: one X read, one scale read and one Y write per item.
        const double gb_per_s =
            value_count * 3.0 * sizeof(float) / seconds / 1e9;
        printf(
            "%-13s %-9s %-12.4f %10.2f %9.2f\n",
            kImpls[i].name,
            correct[i] ? "PASS" : "FAIL",
            ms,
            gelems_per_s,
            gb_per_s
        );
    }

    check_cuda(cudaFree(d_X), "cudaFree X");
    check_cuda(cudaFree(d_S), "cudaFree S");
    check_cuda(cudaFree(d_Y), "cudaFree Y");
    delete[] h_X;
    delete[] h_S;
    delete[] h_ref;
    delete[] h_out;
    return 0;
}
