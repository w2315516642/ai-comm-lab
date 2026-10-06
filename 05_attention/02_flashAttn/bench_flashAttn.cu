// Single-head scaled dot-product attention forward benchmark.
//
// Semantics:
//   O = softmax(Q * K^T / sqrt(D) + causal_mask) * V
//
// Q, K, V and O are contiguous BF16 matrices with layout [N, D]. Dot products,
// softmax statistics, and output accumulation use FP32.
// Dropout is not included.
//
// Usage:
//   ./bench_flashAttn [N] [D] [causal] [iters] [warmup]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

using SolveFn = void (*)(
    const __nv_bfloat16 *Q,
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    __nv_bfloat16 *O,
    int seq_len,
    int head_dim,
    bool causal
);
using SupportsFn = bool (*)(int seq_len, int head_dim, bool causal);

void solve_v0(
    const __nv_bfloat16 *Q,
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    __nv_bfloat16 *O,
    int seq_len,
    int head_dim,
    bool causal
);
void solve_v1(
    const __nv_bfloat16 *Q,
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    __nv_bfloat16 *O,
    int seq_len,
    int head_dim,
    bool causal
);
bool supports_v1(int seq_len, int head_dim, bool causal);

void solve_v2(
    const __nv_bfloat16 *Q,
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    __nv_bfloat16 *O,
    int seq_len,
    int head_dim,
    bool causal
);

struct Impl {
    const char *name;
    SolveFn fn;
    SupportsFn supports;
};

static bool supports_v0(int seq_len, int, bool) {
    // v0 uses one FP32 score per key plus a 256-float reduction buffer.
    constexpr int kPortableSharedBytes = 48 * 1024;
    constexpr int kReductionBytes = 256 * sizeof(float);
    return (size_t)seq_len * sizeof(float) + kReductionBytes <=
           kPortableSharedBytes;
}

static bool supports_v2(int, int head_dim, bool) {
    return head_dim == 32 || head_dim == 64 || head_dim == 128;
}

static const Impl kImpls[] = {
    {"v0-row", solve_v0, supports_v0},
    {"v1-fa2", solve_v1, supports_v1},
    {"v2-mma", solve_v2, supports_v2},
};
static constexpr int kNumImpls = sizeof(kImpls) / sizeof(kImpls[0]);

static void check_cuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

static void fill_input(__nv_bfloat16 *data, size_t count, unsigned *seed) {
    for (size_t i = 0; i < count; i++) {
        *seed = *seed * 1103515245u + 12345u;
        const unsigned value = (*seed >> 8) & 0xffffffu;
        const float x = (float)value / (float)(1 << 23) - 1.0f;
        data[i] = __float2bfloat16_rn(x);
    }
}

static void reference_cpu(
    const __nv_bfloat16 *Q,
    const __nv_bfloat16 *K,
    const __nv_bfloat16 *V,
    float *O,
    int seq_len,
    int head_dim,
    bool causal
) {
    const double scale = 1.0 / sqrt((double)head_dim);
    std::vector<double> scores(seq_len);

    for (int query_idx = 0; query_idx < seq_len; query_idx++) {
        const int valid_keys = causal ? query_idx + 1 : seq_len;
        const __nv_bfloat16 *q = Q + (size_t)query_idx * head_dim;
        double row_max = -INFINITY;

        for (int key_idx = 0; key_idx < valid_keys; key_idx++) {
            const __nv_bfloat16 *k = K + (size_t)key_idx * head_dim;
            double dot = 0.0;
            for (int d = 0; d < head_dim; d++)
                dot += (double)__bfloat162float(q[d]) *
                       __bfloat162float(k[d]);
            scores[key_idx] = dot * scale;
            row_max = fmax(row_max, scores[key_idx]);
        }

        double row_sum = 0.0;
        for (int key_idx = 0; key_idx < valid_keys; key_idx++) {
            scores[key_idx] = exp(scores[key_idx] - row_max);
            row_sum += scores[key_idx];
        }

        float *out = O + (size_t)query_idx * head_dim;
        for (int d = 0; d < head_dim; d++) {
            double value = 0.0;
            for (int key_idx = 0; key_idx < valid_keys; key_idx++) {
                const size_t v_idx = (size_t)key_idx * head_dim + d;
                value += scores[key_idx] * __bfloat162float(V[v_idx]);
            }
            out[d] = (float)(value / row_sum);
        }
    }
}

static bool check_result(
    const char *name,
    const float *expected,
    const __nv_bfloat16 *actual,
    int seq_len,
    int head_dim
) {
    const size_t count = (size_t)seq_len * head_dim;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t worst = 0;

    for (size_t i = 0; i < count; i++) {
        const float actual_value = __bfloat162float(actual[i]);
        const float abs_err = fabsf(actual_value - expected[i]);
        const float rel_err = abs_err / fmaxf(fabsf(expected[i]), 1e-6f);
        if (abs_err > max_abs) {
            max_abs = abs_err;
            worst = i;
        }
        max_rel = fmaxf(max_rel, rel_err);

        const float tol = 2e-2f + 2e-2f * fabsf(expected[i]);
        if (!isfinite(actual_value) || abs_err > tol) {
            const int d = i % head_dim;
            const int query_idx = i / head_dim;
            fprintf(
                stderr,
                "%s mismatch: q=%d d=%d expected=%.8g actual=%.8g "
                "abs_err=%.3g tol=%.3g\n",
                name,
                query_idx,
                d,
                expected[i],
                actual_value,
                abs_err,
                tol
            );
            return false;
        }
    }

    printf(
        "%s check: max_abs=%.3g max_rel=%.3g worst=(%zu,%zu)\n",
        name,
        max_abs,
        max_rel,
        worst / head_dim,
        worst % head_dim
    );
    return true;
}

static float bench_one(
    SolveFn fn,
    const __nv_bfloat16 *d_Q,
    const __nv_bfloat16 *d_K,
    const __nv_bfloat16 *d_V,
    __nv_bfloat16 *d_O,
    int seq_len,
    int head_dim,
    bool causal,
    int iters,
    int warmup
) {
    for (int i = 0; i < warmup; i++)
        fn(d_Q, d_K, d_V, d_O, seq_len, head_dim, causal);
    check_cuda(cudaGetLastError(), "warmup launch");
    check_cuda(cudaDeviceSynchronize(), "warmup execution");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iters; i++)
        fn(d_Q, d_K, d_V, d_O, seq_len, head_dim, causal);
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
    int seq_len = 128;
    int head_dim = 64;
    int causal_arg = 0;
    int iters = 100;
    int warmup = 10;

    if (argc > 1)
        seq_len = atoi(argv[1]);
    if (argc > 2)
        head_dim = atoi(argv[2]);
    if (argc > 3)
        causal_arg = atoi(argv[3]);
    if (argc > 4)
        iters = atoi(argv[4]);
    if (argc > 5)
        warmup = atoi(argv[5]);

    if (seq_len <= 0 || head_dim <= 0 ||
        (causal_arg != 0 && causal_arg != 1) || iters <= 0 || warmup < 0) {
        fprintf(
            stderr, "require N,D,iters > 0, causal in {0,1}, warmup >= 0\n"
        );
        return 1;
    }
    const bool causal = causal_arg != 0;

    const size_t value_count = (size_t)seq_len * head_dim;
    const size_t value_bytes = value_count * sizeof(__nv_bfloat16);
    __nv_bfloat16 *h_Q = new __nv_bfloat16[value_count];
    __nv_bfloat16 *h_K = new __nv_bfloat16[value_count];
    __nv_bfloat16 *h_V = new __nv_bfloat16[value_count];
    float *h_ref = new float[value_count];
    __nv_bfloat16 *h_out = new __nv_bfloat16[value_count];

    unsigned seed = 12345;
    fill_input(h_Q, value_count, &seed);
    fill_input(h_K, value_count, &seed);
    fill_input(h_V, value_count, &seed);
    reference_cpu(h_Q, h_K, h_V, h_ref, seq_len, head_dim, causal);

    __nv_bfloat16 *d_Q = nullptr;
    __nv_bfloat16 *d_K = nullptr;
    __nv_bfloat16 *d_V = nullptr;
    __nv_bfloat16 *d_O = nullptr;
    check_cuda(cudaMalloc(&d_Q, value_bytes), "cudaMalloc Q");
    check_cuda(cudaMalloc(&d_K, value_bytes), "cudaMalloc K");
    check_cuda(cudaMalloc(&d_V, value_bytes), "cudaMalloc V");
    check_cuda(cudaMalloc(&d_O, value_bytes), "cudaMalloc O");
    check_cuda(
        cudaMemcpy(d_Q, h_Q, value_bytes, cudaMemcpyHostToDevice), "copy Q"
    );
    check_cuda(
        cudaMemcpy(d_K, h_K, value_bytes, cudaMemcpyHostToDevice), "copy K"
    );
    check_cuda(
        cudaMemcpy(d_V, h_V, value_bytes, cudaMemcpyHostToDevice), "copy V"
    );

    const double attended_pairs = causal ? (double)seq_len * (seq_len + 1) / 2.0
                                         : (double)seq_len * seq_len;
    // Counts the QK^T and PV multiply-adds; softmax operations are omitted.
    const double operations = 4.0 * attended_pairs * head_dim;
    bool all_correct = true;

    printf(
        "N=%d D=%d causal=%d iters=%d warmup=%d\n",
        seq_len,
        head_dim,
        causal_arg,
        iters,
        warmup
    );
    printf("impl          correct   avg_ms      GFLOP/s\n");
    printf("------------------------------------------------\n");
    for (int i = 0; i < kNumImpls; i++) {
        if (!kImpls[i].supports(seq_len, head_dim, causal)) {
            printf("%-13s %-9s %-11s %10s\n", kImpls[i].name, "SKIP", "-", "-");
            continue;
        }

        check_cuda(cudaMemset(d_O, 0xa5, value_bytes), "memset O");
        kImpls[i].fn(d_Q, d_K, d_V, d_O, seq_len, head_dim, causal);
        check_cuda(cudaGetLastError(), "correctness launch");
        check_cuda(cudaDeviceSynchronize(), "correctness execution");
        check_cuda(
            cudaMemcpy(h_out, d_O, value_bytes, cudaMemcpyDeviceToHost),
            "copy O back"
        );
        const bool correct = check_result(
            kImpls[i].name, h_ref, h_out, seq_len, head_dim
        );
        all_correct = all_correct && correct;

        const float ms = bench_one(
            kImpls[i].fn,
            d_Q,
            d_K,
            d_V,
            d_O,
            seq_len,
            head_dim,
            causal,
            iters,
            warmup
        );
        const double gflops = operations / ((double)ms * 1e6);
        printf(
            "%-13s %-9s %-11.4f %10.2f\n",
            kImpls[i].name,
            correct ? "PASS" : "FAIL",
            ms,
            gflops
        );
    }

    check_cuda(cudaFree(d_Q), "cudaFree Q");
    check_cuda(cudaFree(d_K), "cudaFree K");
    check_cuda(cudaFree(d_V), "cudaFree V");
    check_cuda(cudaFree(d_O), "cudaFree O");
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_ref;
    delete[] h_out;
    return all_correct ? 0 : 2;
}
