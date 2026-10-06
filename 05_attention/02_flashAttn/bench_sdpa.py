#!/usr/bin/env python3
"""Benchmark available BF16 scaled-dot-product-attention backends."""

import argparse
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def benchmark(fn, warmup, iters):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        stop.synchronize()
    return start.elapsed_time(stop) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=128)
    parser.add_argument("D", type=int, nargs="?", default=64)
    parser.add_argument("causal", type=int, nargs="?", choices=(0, 1), default=0)
    parser.add_argument("iters", type=int, nargs="?", default=500)
    parser.add_argument("warmup", type=int, nargs="?", default=50)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this PyTorch installation")

    torch.manual_seed(12345)
    q = torch.randn(1, 1, args.N, args.D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    is_causal = bool(args.causal)

    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        reference = F.scaled_dot_product_attention(
            q.float(),
            k.float(),
            v.float(),
            dropout_p=0.0,
            is_causal=is_causal,
        )

    attended_pairs = (
        args.N * (args.N + 1) / 2 if is_causal else args.N * args.N
    )
    operations = 4.0 * attended_pairs * args.D

    backends = [("sdpa-auto", None), ("sdpa-math", SDPBackend.MATH)]
    for name in ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN_ATTENTION"):
        backend = getattr(SDPBackend, name, None)
        if backend is not None:
            backends.append(("sdpa-" + name.lower().replace("_attention", ""), backend))

    print(
        f"N={args.N} D={args.D} causal={args.causal} "
        f"dtype=bf16 iters={args.iters} warmup={args.warmup}"
    )
    print("impl             correct   avg_ms      TFLOP/s")
    print("---------------------------------------------------")

    for name, backend in backends:
        def run(backend=backend):
            context = nullcontext() if backend is None else sdpa_kernel(backend)
            with context:
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    dropout_p=0.0,
                    is_causal=is_causal,
                )

        try:
            actual = run()
            torch.testing.assert_close(
                actual.float(), reference, atol=2e-2, rtol=2e-2
            )
            ms = benchmark(run, args.warmup, args.iters)
            tflops = operations / (ms * 1e9)
            print(f"{name:<16} {'PASS':<9} {ms:<11.4f} {tflops:10.2f}")
        except (RuntimeError, AssertionError) as exc:
            reason = str(exc).splitlines()[0]
            print(f"{name:<16} {'SKIP':<9} {'-':<11} {'-':>10}  {reason}")

    try:
        from flash_attn import flash_attn_func
    except ImportError:
        print(f"{'flash-attn-2':<16} {'SKIP':<9} {'-':<11} {'-':>10}  not installed")
        return

    q_fa = q.transpose(1, 2).contiguous()
    k_fa = k.transpose(1, 2).contiguous()
    v_fa = v.transpose(1, 2).contiguous()

    def run_fa2():
        return flash_attn_func(
            q_fa, k_fa, v_fa, dropout_p=0.0, causal=is_causal
        )

    try:
        actual = run_fa2().transpose(1, 2)
        torch.testing.assert_close(
            actual.float(), reference, atol=2e-2, rtol=2e-2
        )
        ms = benchmark(run_fa2, args.warmup, args.iters)
        tflops = operations / (ms * 1e9)
        print(f"{'flash-attn-2':<16} {'PASS':<9} {ms:<11.4f} {tflops:10.2f}")
    except (RuntimeError, AssertionError) as exc:
        reason = str(exc).splitlines()[0]
        print(f"{'flash-attn-2':<16} {'SKIP':<9} {'-':<11} {'-':>10}  {reason}")


if __name__ == "__main__":
    main()
