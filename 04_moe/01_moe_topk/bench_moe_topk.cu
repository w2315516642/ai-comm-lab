// MoE top-k version comparison benchmark.
//
// Semantics:
//   topk_values/topk_indices = topk(logits, k, dim=-1)
//   topk_weights = softmax(topk_values, dim=-1)
//
// Usage:
//   ./bench_moe_topk [M] [E] [k] [iters] [warmup] [impl_idx...]
//   If impl_idx is omitted, all registered implementations are benchmarked.

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "moe_topk_api.h"

struct Impl {
    const char *name;
    SolveFn fn;
};

static const Impl kImpls[] = {
    {"v0-baseline", solve_v0},
    {"ans0", solve_ans0},
    {"ans1", solve_ans1},
    {"ans2", solve_ans2},
    {"v1-warp-per-t", solve_v1},
    {"v2-local", solve_v2},
    // {"v1-xxx", solve_v1},
};
static const int kNumImpls = (int)(sizeof(kImpls) / sizeof(kImpls[0]));

static void die(const char *msg) {
    fprintf(
        stderr, "error: %s: %s\n", msg, cudaGetErrorString(cudaGetLastError())
    );
    exit(1);
}

static void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static void fill_logits(float *arr, int n, unsigned *seed) {
    for (int i = 0; i < n; i++) {
        *seed = *seed * 1103515245u + 12345u;
        unsigned r = (*seed >> 8) & 0xffffffu;
        arr[i] = ((float)r / (float)(1 << 24)) * 20.0f - 10.0f;

        // Inject a few deterministic ties to catch unstable tie-breaking.
        if ((i % 257) == 0)
            arr[i] = 1.0f;
    }
}

static bool better_pair(
    float lhs_val, int lhs_idx, float rhs_val, int rhs_idx
) {
    return lhs_val > rhs_val || (lhs_val == rhs_val && lhs_idx < rhs_idx);
}

static void reference_cpu(
    const float *logits,
    float *topk_weights,
    int *topk_indices,
    int M,
    int E,
    int k
) {
    for (int m = 0; m < M; m++) {
        for (int r = 0; r < k; r++) {
            float best_val = -INFINITY;
            int best_idx = E;

            for (int e = 0; e < E; e++) {
                bool used = false;
                for (int prev = 0; prev < r; prev++) {
                    if (topk_indices[m * k + prev] == e) {
                        used = true;
                        break;
                    }
                }
                if (used)
                    continue;

                float v = logits[m * E + e];
                if (better_pair(v, e, best_val, best_idx)) {
                    best_val = v;
                    best_idx = e;
                }
            }

            topk_weights[m * k + r] = best_val;
            topk_indices[m * k + r] = best_idx;
        }

        float topk_max = -INFINITY;
        for (int r = 0; r < k; r++)
            topk_max = fmaxf(topk_max, topk_weights[m * k + r]);

        float topk_sum = 0.0f;
        for (int r = 0; r < k; r++) {
            float v = expf(topk_weights[m * k + r] - topk_max);
            topk_weights[m * k + r] = v;
            topk_sum += v;
        }

        for (int r = 0; r < k; r++)
            topk_weights[m * k + r] /= topk_sum;
    }
}

static bool check_result(
    const char *name,
    const float *expected_weights,
    const int *expected_indices,
    const float *actual_weights,
    const int *actual_indices,
    int M,
    int k
) {
    const int n = M * k;
    for (int i = 0; i < n; i++) {
        if (expected_indices[i] != actual_indices[i]) {
            fprintf(
                stderr,
                "%s index mismatch: row=%d rank=%d expected=%d actual=%d\n",
                name,
                i / k,
                i % k,
                expected_indices[i],
                actual_indices[i]
            );
            return false;
        }

        float diff = fabsf(expected_weights[i] - actual_weights[i]);
        float tol = 1e-5f * fmaxf(1.0f, fabsf(expected_weights[i]));
        if (!(diff <= tol)) {
            fprintf(
                stderr,
                "%s weight mismatch: row=%d rank=%d expected=%.8g "
                "actual=%.8g\n",
                name,
                i / k,
                i % k,
                expected_weights[i],
                actual_weights[i]
            );
            return false;
        }
    }
    return true;
}

static float bench_one(
    SolveFn fn,
    const float *d_logits,
    float *d_topk_weights,
    int *d_topk_indices,
    int M,
    int E,
    int k,
    int iters,
    int warmup
) {
    for (int i = 0; i < warmup; i++)
        fn(d_logits, d_topk_weights, d_topk_indices, M, E, k);
    check_cuda(cudaDeviceSynchronize(), "warmup");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iters; i++)
        fn(d_logits, d_topk_weights, d_topk_indices, M, E, k);
    check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

int main(int argc, char **argv) {
    int M = 4096, E = 128, k = 2, iters = 200, warmup = 20;
    int run_idx[16] = {0};
    int n_run = 0;

    if (argc > 1)
        M = atoi(argv[1]);
    if (argc > 2)
        E = atoi(argv[2]);
    if (argc > 3)
        k = atoi(argv[3]);
    if (argc > 4)
        iters = atoi(argv[4]);
    if (argc > 5)
        warmup = atoi(argv[5]);
    for (int i = 6; i < argc; i++) {
        if (n_run < 16)
            run_idx[n_run++] = atoi(argv[i]);
    }
    if (n_run == 0) {
        for (int i = 0; i < kNumImpls; i++)
            run_idx[n_run++] = i;
    }

    if (M <= 0 || E <= 0 || k <= 0 || k > E || iters <= 0 || warmup < 0) {
        fprintf(stderr, "invalid args: require M>0, E>0, 0<k<=E, iters>0\n");
        return 1;
    }

    const size_t logits_bytes = sizeof(float) * (size_t)M * E;
    const size_t weights_bytes = sizeof(float) * (size_t)M * k;
    const size_t indices_bytes = sizeof(int) * (size_t)M * k;

    unsigned seed = 12345;
    float *h_logits = new float[(size_t)M * E];
    float *h_ref_weights = new float[(size_t)M * k];
    int *h_ref_indices = new int[(size_t)M * k];
    float *h_out_weights = new float[(size_t)M * k];
    int *h_out_indices = new int[(size_t)M * k];

    fill_logits(h_logits, M * E, &seed);
    const bool skip_correct = getenv("MOE_TOPK_SKIP_CORRECT") != nullptr;
    if (!skip_correct)
        reference_cpu(h_logits, h_ref_weights, h_ref_indices, M, E, k);

    float *d_logits = nullptr;
    float *d_topk_weights = nullptr;
    int *d_topk_indices = nullptr;
    check_cuda(cudaMalloc(&d_logits, logits_bytes), "cudaMalloc logits");
    check_cuda(
        cudaMalloc(&d_topk_weights, weights_bytes), "cudaMalloc weights"
    );
    check_cuda(
        cudaMalloc(&d_topk_indices, indices_bytes), "cudaMalloc indices"
    );
    check_cuda(
        cudaMemcpy(d_logits, h_logits, logits_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy logits"
    );
    check_cuda(cudaDeviceSynchronize(), "input copy");

    bool correct[kNumImpls] = {false};
    if (!skip_correct) {
        for (int impl = 0; impl < kNumImpls; impl++) {
            check_cuda(
                cudaMemset(d_topk_weights, 0xa5, weights_bytes),
                "memset weights"
            );
            check_cuda(
                cudaMemset(d_topk_indices, 0xa5, indices_bytes),
                "memset indices"
            );
            kImpls[impl].fn(d_logits, d_topk_weights, d_topk_indices, M, E, k);
            if (cudaGetLastError() != cudaSuccess)
                die("impl launch");
            check_cuda(cudaDeviceSynchronize(), "impl execution");
            check_cuda(
                cudaMemcpy(
                    h_out_weights,
                    d_topk_weights,
                    weights_bytes,
                    cudaMemcpyDeviceToHost
                ),
                "copy weights back"
            );
            check_cuda(
                cudaMemcpy(
                    h_out_indices,
                    d_topk_indices,
                    indices_bytes,
                    cudaMemcpyDeviceToHost
                ),
                "copy indices back"
            );
            correct[impl] = check_result(
                kImpls[impl].name,
                h_ref_weights,
                h_ref_indices,
                h_out_weights,
                h_out_indices,
                M,
                k
            );
        }
    }

    printf("M=%d E=%d k=%d iters=%d warmup=%d\n", M, E, k, iters, warmup);
    printf("impl          correct   avg_ms       Mrows/s      Gelems/s\n");
    printf("----------------------------------------------------------\n");
    for (int i = 0; i < n_run; i++) {
        int impl = run_idx[i];
        if (impl < 0 || impl >= kNumImpls) {
            fprintf(
                stderr, "impl idx %d out of range [0, %d)\n", impl, kNumImpls
            );
            continue;
        }

        float ms = bench_one(
            kImpls[impl].fn,
            d_logits,
            d_topk_weights,
            d_topk_indices,
            M,
            E,
            k,
            iters,
            warmup
        );
        double mrows_per_s = (double)M / (ms / 1000.0) / 1e6;
        double gelems_per_s = (double)M * E / (ms / 1000.0) / 1e9;
        const char *status =
            skip_correct ? "SKIP" : (correct[impl] ? "PASS" : "FAIL");
        printf(
            "%-13s %-9s %-12.4f %10.2f %11.2f\n",
            kImpls[impl].name,
            status,
            ms,
            mrows_per_s,
            gelems_per_s
        );
    }

    cudaFree(d_logits);
    cudaFree(d_topk_weights);
    cudaFree(d_topk_indices);
    delete[] h_logits;
    delete[] h_ref_weights;
    delete[] h_ref_indices;
    delete[] h_out_weights;
    delete[] h_out_indices;
    return 0;
}
