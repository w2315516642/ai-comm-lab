#pragma once
// 所有 spec-dec 版本算子的统一签名。
// 每个版本文件(.cu)定义自己的 solve_vN,bench 通过这张签名把它们统一调起来。

// draft_tokens:  [B, T]   int32, 设备端
// draft_probs:   [B, T, V] float, 设备端
// target_probs:  [B, T, V] float, 设备端
// uniform_samples: [B, T+1] float, 设备端
// output_tokens: [B, T+1] int32, 设备端
typedef void (*SolveFn)(
    const int *draft_tokens,
    const float *draft_probs,
    const float *target_probs,
    const float *uniform_samples,
    int *output_tokens,
    int B,
    int T,
    int V
);

extern "C" {
void solve_v0(
    const int *draft_tokens,
    const float *draft_probs,
    const float *target_probs,
    const float *uniform_samples,
    int *output_tokens,
    int B,
    int T,
    int V
);

void solve_v1(
    const int *draft_tokens,
    const float *draft_probs,
    const float *target_probs,
    const float *uniform_samples,
    int *output_tokens,
    int B,
    int T,
    int V
);

void solve_v2(
    const int *draft_tokens,
    const float *draft_probs,
    const float *target_probs,
    const float *uniform_samples,
    int *output_tokens,
    int B,
    int T,
    int V
);

void solve_v3(
    const int *draft_tokens,
    const float *draft_probs,
    const float *target_probs,
    const float *uniform_samples,
    int *output_tokens,
    int B,
    int T,
    int V
);

// 新版本在这里补声明,例如:
// void solve_v1(const int *draft_tokens, ...);
void solve_ans0(
    const int *draft_tokens,
    const float *draft_probs,
    const float *target_probs,
    const float *uniform_samples,
    int *output_tokens,
    int B,
    int T,
    int V
);

void solve_ans1(
    const int *draft_tokens,
    const float *draft_probs,
    const float *target_probs,
    const float *uniform_samples,
    int *output_tokens,
    int B,
    int T,
    int V
);
}
