// spec-dec 版本对比 benchmark
//
// 用法:
//   ./bench_spec_dec [B] [T] [V] [iters] [warmup] [impl_idx...]
//   不指定 impl_idx 时跑注册表里所有版本;指定时只跑对应下标的版本。
//
// 每次改动算子后重编译即可,新增版本文件只需在注册表里加一行。

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

#include "spec_dec_api.h"

// ================= 版本注册表 =================
// 新增版本: 1) 在 spec_dec_api.h 里声明 solve_vN  2) 在下面加一行
struct Impl {
    const char *name;
    SolveFn fn;
};

static const Impl kImpls[] = {
    {"v0-baseline", solve_v0},
    {"v1-approved", solve_v1},
    {"ans0", solve_ans0},
    {"ans1", solve_ans1},
    // {"v1-xxx", solve_v1},
};
static const int kNumImpls = (int)(sizeof(kImpls) / sizeof(kImpls[0]));

static void die(const char *msg) {
    fprintf(
        stderr, "error: %s: %s\n", msg, cudaGetErrorString(cudaGetLastError())
    );
    exit(1);
}

// 线性同余伪随机,固定种子 => 每次运行生成同样的输入,
// 保证不同版本吃到完全相同的输入,对比才公平。
static void fill_randf(float *arr, int n, unsigned *seed) {
    for (int i = 0; i < n; i++) {
        *seed = *seed * 1103515245u + 12345u;
        arr[i] = (float)((*seed >> 8) & 0xffffff) / (float)(1 << 24);
    }
}

static void fill_randi(int *arr, int n, int max, unsigned *seed) {
    for (int i = 0; i < n; i++) {
        *seed = *seed * 1103515245u + 12345u;
        arr[i] = (int)(((*seed >> 8) & 0xffffff) % (unsigned)max);
    }
}

// 生成合法概率分布:先填随机数,再按行归一化
static void fill_probs(float *arr, int rows, int V, unsigned *seed) {
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        for (int i = 0; i < V; i++) {
            *seed = *seed * 1103515245u + 12345u;
            arr[r * V + i] =
                (float)((*seed >> 8) & 0xffffff) / (float)(1 << 24);
            sum += arr[r * V + i];
        }
        for (int i = 0; i < V; i++)
            arr[r * V + i] /= sum;
    }
}

// 对某个版本跑 iters 次,返回平均每次的耗时(ms)
// 注意参数含义:d_q = target 分布(q),d_p = draft 分布(p)。
// SolveFn 的顺序是 (draft_tokens, draft_probs, target_probs, ...),
// 所以调用时 draft 传 d_p、target 传 d_q —— 这里传反过一次,别重蹈覆辙。
static float bench_one(
    SolveFn fn,
    const int *d_draft,
    const float *d_q,
    const float *d_p,
    const float *d_u,
    int *d_out,
    int B,
    int T,
    int V,
    int iters,
    int warmup
) {
    for (int i = 0; i < warmup; i++)
        fn(d_draft, d_p, d_q, d_u, d_out, B, T, V);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        fn(d_draft, d_p, d_q, d_u, d_out, B, T, V);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

int main(int argc, char **argv) {
    int B = 8, T = 8, V = 32768, iters = 200, warmup = 20;
    int run_idx[16] = {0};
    int n_run = 0;

    if (argc > 1)
        B = atoi(argv[1]);
    if (argc > 2)
        T = atoi(argv[2]);
    if (argc > 3)
        V = atoi(argv[3]);
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

    // ---------- 生成输入(主机侧) ----------
    unsigned seed = 12345;
    int *h_draft = new int[B * T];
    float *h_q = new float[B * T * V];
    float *h_p = new float[B * T * V];
    float *h_u = new float[B * (T + 1)];
    int *h_out = new int[B * (T + 1)];
    int *h_out2 = new int[B * (T + 1)];

    fill_randi(h_draft, B * T, V, &seed);
    fill_probs(h_q, B * T, V, &seed);
    fill_probs(h_p, B * T, V, &seed);
    fill_randf(h_u, B * (T + 1), &seed);

    // ---------- 拷到设备端 ----------
    int *d_draft, *d_out;
    float *d_q, *d_p, *d_u;
    if (cudaMalloc(&d_draft, sizeof(int) * B * T) != cudaSuccess)
        die("cudaMalloc");
    if (cudaMalloc(&d_q, sizeof(float) * B * T * V) != cudaSuccess)
        die("cudaMalloc");
    if (cudaMalloc(&d_p, sizeof(float) * B * T * V) != cudaSuccess)
        die("cudaMalloc");
    if (cudaMalloc(&d_u, sizeof(float) * B * (T + 1)) != cudaSuccess)
        die("cudaMalloc");
    if (cudaMalloc(&d_out, sizeof(int) * B * (T + 1)) != cudaSuccess)
        die("cudaMalloc");
    cudaMemcpy(d_draft, h_draft, sizeof(int) * B * T, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, h_q, sizeof(float) * B * T * V, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, h_p, sizeof(float) * B * T * V, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_u, sizeof(float) * B * (T + 1), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // ---------- 正确性:所有版本输出必须和 v0 一致 ----------
    kImpls[0].fn(d_draft, d_p, d_q, d_u, d_out, B, T, V);
    if (cudaGetLastError() != cudaSuccess)
        die("solve_v0 launch");
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sizeof(int) * B * (T + 1), cudaMemcpyDeviceToHost);

    bool correct[kNumImpls] = {true};
    for (int k = 1; k < kNumImpls; k++) {
        kImpls[k].fn(d_draft, d_p, d_q, d_u, d_out, B, T, V);
        if (cudaGetLastError() != cudaSuccess)
            die("impl launch");
        cudaDeviceSynchronize();
        cudaMemcpy(
            h_out2, d_out, sizeof(int) * B * (T + 1), cudaMemcpyDeviceToHost
        );
        correct[k] = memcmp(h_out, h_out2, sizeof(int) * B * (T + 1)) == 0;
    }

    // ---------- 计时对比 ----------
    printf("impl          correct   avg_ms     tokens/s\n");
    printf("----------------------------------------------\n");
    for (int i = 0; i < n_run; i++) {
        int k = run_idx[i];
        if (k < 0 || k >= kNumImpls) {
            fprintf(stderr, "impl idx %d out of range [0, %d)\n", k, kNumImpls);
            continue;
        }
        float ms = bench_one(
            kImpls[k].fn, d_draft, d_q, d_p, d_u, d_out, B, T, V, iters, warmup
        );
        double tok_per_s = (double)B * (T + 1) / (ms / 1000.0);
        printf(
            "%-13s %-9s %-12.4f %10.1f\n",
            kImpls[k].name,
            correct[k] ? "PASS" : "FAIL",
            ms,
            tok_per_s
        );
    }

    // ---------- 清理 ----------
    cudaFree(d_draft);
    cudaFree(d_q);
    cudaFree(d_p);
    cudaFree(d_u);
    cudaFree(d_out);
    delete[] h_draft;
    delete[] h_q;
    delete[] h_p;
    delete[] h_u;
    delete[] h_out;
    delete[] h_out2;
    return 0;
}
