// Scaled dot-product attention forward benchmark.
//
// Semantics:
//   O = softmax(Q * K^T / sqrt(D) + causal_mask) * V
//
// Q, K, V and O are contiguous FP32 tensors with layout [B, H, N, D].
// Dropout is not included.
//
// Usage:
//   ./bench_flashAttn [B] [H] [N] [D] [causal] [iters] [warmup]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

using SolveFn = void (*)(
    const float *Q,
    const float *K,
    const float *V,
    float *O,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    bool causal
);
using SupportsFn =
    bool (*)(int batch, int heads, int seq_len, int head_dim, bool causal);

void solve_v1(
    const float *Q,
    const float *K,
    const float *V,
    float *O,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    bool causal
);

struct Impl {
    const char *name;
    SolveFn fn;
    SupportsFn supports;
};

static bool supports_v1(
    int, int, int seq_len, int, bool
) {
    // v1 uses one FP32 score per key plus a 256-float reduction buffer.
    constexpr int kPortableSharedBytes = 48 * 1024;
    constexpr int kReductionBytes = 256 * sizeof(float);
    return (size_t)seq_len * sizeof(float) + kReductionBytes <=
           kPortableSharedBytes;
}

static const Impl kImpls[] = {
    {"v1-row", solve_v1, supports_v1},
    // {"v2-...", solve_v2, supports_v2},
    // {"v3-...", solve_v3, supports_v3},
    // {"v4-...", solve_v4, supports_v4},
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

static void reference_cpu(
    const float *Q,
    const float *K,
    const float *V,
    float *O,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    bool causal
) {
    const double scale = 1.0 / sqrt((double)head_dim);
    std::vector<double> scores(seq_len);

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            const size_t row_base =
                ((size_t)b * heads + h) * seq_len * head_dim;

            for (int query_idx = 0; query_idx < seq_len; query_idx++) {
                const int valid_keys = causal ? query_idx + 1 : seq_len;
                const float *q =
                    Q + row_base + (size_t)query_idx * head_dim;
                double row_max = -INFINITY;

                for (int key_idx = 0; key_idx < valid_keys; key_idx++) {
                    const float *k =
                        K + row_base + (size_t)key_idx * head_dim;
                    double dot = 0.0;
                    for (int d = 0; d < head_dim; d++)
                        dot += (double)q[d] * k[d];
                    scores[key_idx] = dot * scale;
                    row_max = fmax(row_max, scores[key_idx]);
                }

                double row_sum = 0.0;
                for (int key_idx = 0; key_idx < valid_keys; key_idx++) {
                    scores[key_idx] = exp(scores[key_idx] - row_max);
                    row_sum += scores[key_idx];
                }

                float *out =
                    O + row_base + (size_t)query_idx * head_dim;
                for (int d = 0; d < head_dim; d++) {
                    double value = 0.0;
                    for (int key_idx = 0; key_idx < valid_keys; key_idx++) {
                        const size_t v_idx =
                            row_base + (size_t)key_idx * head_dim + d;
                        value += scores[key_idx] * V[v_idx];
                    }
                    out[d] = (float)(value / row_sum);
                }
            }
        }
    }
}

static bool check_result(
    const char *name,
    const float *expected,
    const float *actual,
    int batch,
    int heads,
    int seq_len,
    int head_dim
) {
    const size_t count = (size_t)batch * heads * seq_len * head_dim;
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t worst = 0;

    for (size_t i = 0; i < count; i++) {
        const float abs_err = fabsf(actual[i] - expected[i]);
        const float rel_err =
            abs_err / fmaxf(fabsf(expected[i]), 1e-6f);
        if (abs_err > max_abs) {
            max_abs = abs_err;
            worst = i;
        }
        max_rel = fmaxf(max_rel, rel_err);

        const float tol = 3e-4f + 3e-4f * fabsf(expected[i]);
        if (!isfinite(actual[i]) || abs_err > tol) {
            size_t position = i;
            const int d = position % head_dim;
            position /= head_dim;
            const int query_idx = position % seq_len;
            position /= seq_len;
            const int h = position % heads;
            const int b = position / heads;
            fprintf(
                stderr,
                "%s mismatch: b=%d h=%d q=%d d=%d expected=%.8g "
                "actual=%.8g abs_err=%.3g tol=%.3g\n",
                name,
                b,
                h,
                query_idx,
                d,
                expected[i],
                actual[i],
                abs_err,
                tol
            );
            return false;
        }
    }

    size_t position = worst;
    const int worst_d = position % head_dim;
    position /= head_dim;
    const int worst_query = position % seq_len;
    position /= seq_len;
    const int worst_head = position % heads;
    const int worst_batch = position / heads;
    printf(
        "%s check: max_abs=%.3g max_rel=%.3g "
        "worst=(%d,%d,%d,%d)\n",
        name,
        max_abs,
        max_rel,
        worst_batch,
        worst_head,
        worst_query,
        worst_d
    );
    return true;
}

static float bench_one(
    SolveFn fn,
    const float *d_Q,
    const float *d_K,
    const float *d_V,
    float *d_O,
    int batch,
    int heads,
    int seq_len,
    int head_dim,
    bool causal,
    int iters,
    int warmup
) {
    for (int i = 0; i < warmup; i++) {
        fn(
            d_Q,
            d_K,
            d_V,
            d_O,
            batch,
            heads,
            seq_len,
            head_dim,
            causal
        );
    }
    check_cuda(cudaGetLastError(), "warmup launch");
    check_cuda(cudaDeviceSynchronize(), "warmup execution");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start), "cudaEventRecord start");
    for (int i = 0; i < iters; i++) {
        fn(
            d_Q,
            d_K,
            d_V,
            d_O,
            batch,
            heads,
            seq_len,
            head_dim,
            causal
        );
    }
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
    int batch = 1;
    int heads = 4;
    int seq_len = 128;
    int head_dim = 64;
    int causal_arg = 0;
    int iters = 100;
    int warmup = 10;

    if (argc > 1)
        batch = atoi(argv[1]);
    if (argc > 2)
        heads = atoi(argv[2]);
    if (argc > 3)
        seq_len = atoi(argv[3]);
    if (argc > 4)
        head_dim = atoi(argv[4]);
    if (argc > 5)
        causal_arg = atoi(argv[5]);
    if (argc > 6)
        iters = atoi(argv[6]);
    if (argc > 7)
        warmup = atoi(argv[7]);

    if (batch <= 0 || heads <= 0 || seq_len <= 0 || head_dim <= 0 ||
        (causal_arg != 0 && causal_arg != 1) || iters <= 0 || warmup < 0) {
        fprintf(
            stderr,
            "require B,H,N,D,iters > 0, causal in {0,1}, warmup >= 0\n"
        );
        return 1;
    }
    const bool causal = causal_arg != 0;

    const size_t value_count =
        (size_t)batch * heads * seq_len * head_dim;
    const size_t value_bytes = value_count * sizeof(float);
    float *h_Q = new float[value_count];
    float *h_K = new float[value_count];
    float *h_V = new float[value_count];
    float *h_ref = new float[value_count];
    float *h_out = new float[value_count];

    unsigned seed = 12345;
    fill_input(h_Q, value_count, &seed);
    fill_input(h_K, value_count, &seed);
    fill_input(h_V, value_count, &seed);
    reference_cpu(
        h_Q,
        h_K,
        h_V,
        h_ref,
        batch,
        heads,
        seq_len,
        head_dim,
        causal
    );

    float *d_Q = nullptr;
    float *d_K = nullptr;
    float *d_V = nullptr;
    float *d_O = nullptr;
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

    const double attended_pairs = causal
                                      ? (double)seq_len * (seq_len + 1) / 2.0
                                      : (double)seq_len * seq_len;
    // Counts the QK^T and PV multiply-adds; softmax operations are omitted.
    const double operations =
        4.0 * batch * heads * attended_pairs * head_dim;
    bool all_correct = true;

    printf(
        "B=%d H=%d N=%d D=%d causal=%d iters=%d warmup=%d\n",
        batch,
        heads,
        seq_len,
        head_dim,
        causal_arg,
        iters,
        warmup
    );
    printf("impl          correct   avg_ms      GFLOP/s\n");
    printf("------------------------------------------------\n");
    for (int i = 0; i < kNumImpls; i++) {
        if (!kImpls[i].supports(
                batch, heads, seq_len, head_dim, causal
            )) {
            printf(
                "%-13s %-9s %-11s %10s\n", kImpls[i].name, "SKIP", "-", "-"
            );
            continue;
        }

        check_cuda(cudaMemset(d_O, 0xa5, value_bytes), "memset O");
        kImpls[i].fn(
            d_Q,
            d_K,
            d_V,
            d_O,
            batch,
            heads,
            seq_len,
            head_dim,
            causal
        );
        check_cuda(cudaGetLastError(), "correctness launch");
        check_cuda(cudaDeviceSynchronize(), "correctness execution");
        check_cuda(
            cudaMemcpy(h_out, d_O, value_bytes, cudaMemcpyDeviceToHost),
            "copy O back"
        );
        const bool correct = check_result(
            kImpls[i].name,
            h_ref,
            h_out,
            batch,
            heads,
            seq_len,
            head_dim
        );
        all_correct = all_correct && correct;

        const float ms = bench_one(
            kImpls[i].fn,
            d_Q,
            d_K,
            d_V,
            d_O,
            batch,
            heads,
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
