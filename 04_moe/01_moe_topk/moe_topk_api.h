#pragma once

// logits:       [M, E] float, device
// topk_weights: [M, k] float, device
// topk_indices: [M, k] int32, device
//
// The benchmark oracle first selects top-k raw logits, then applies softmax
// over those k selected values. Ties are resolved by smaller expert index
// first.
typedef void (*SolveFn)(
    const float *logits,
    float *topk_weights,
    int *topk_indices,
    int M,
    int E,
    int k
);

extern "C" {
void solve_v0(
    const float *logits,
    float *topk_weights,
    int *topk_indices,
    int m,
    int e,
    int k
);

void solve_v1(
    const float *logits,
    float *topk_weights,
    int *topk_indices,
    int m,
    int e,
    int k
);

void solve_v2(
    const float *logits,
    float *topk_weights,
    int *topk_indices,
    int m,
    int e,
    int k
);

void solve_ans0(
    const float *logits,
    float *topk_weights,
    int *topk_indices,
    int M,
    int E,
    int k
);

void solve_ans1(
    const float *logits,
    float *topk_weights,
    int *topk_indices,
    int M,
    int E,
    int k
);

void solve_ans2(
    const float *logits,
    float *topk_weights,
    int *topk_indices,
    int M,
    int E,
    int k
);

// Add new versions here, for example:
// void solve_v1(
//     const float *logits,
//     float *topk_weights,
//     int *topk_indices,
//     int M,
//     int E,
//     int k
// );
}
