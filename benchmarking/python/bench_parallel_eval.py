#!/usr/bin/env python3
"""Benchmark parallel vs sequential tensor_eval to find the optimal threshold.

Evaluates tensors of increasing size at a single time point and at an array
of time points, comparing sequential (high threshold) and parallel (low
threshold) execution.  Prints a table and optionally plots the results.

Usage:
    python bench_parallel_eval.py
    python bench_parallel_eval.py --plot
    python bench_parallel_eval.py --sizes 10 50 100 500 1000 5000
"""

import argparse
import timeit

import numpy as np

import masspcf as mpcf
from masspcf.random import noisy_sin


def bench_eval(X, t, n_repeat=5):
    """Time tensor evaluation, return median seconds."""
    times = timeit.repeat(lambda: X(t), number=1, repeat=n_repeat)
    return np.median(times)


def run_benchmark(sizes, n_points_list, n_times, n_repeat):
    results = []

    for n_points in n_points_list:
        print(f"--- {n_points} breakpoints per PCF ---")
        for n in sizes:
            X = noisy_sin((n,), n_points=n_points, dtype=mpcf.pcf64)
            t_scalar = np.float64(0.5)
            t_array = np.linspace(0, 1, n_times, dtype=np.float64)

            for label, t in [("scalar", t_scalar),
                              (f"array({n_times})", t_array)]:
                # Sequential (threshold set very high)
                mpcf.system.set_parallel_eval_threshold(10**9)
                t_seq = bench_eval(X, t, n_repeat)

                # Parallel (threshold set to 1)
                mpcf.system.set_parallel_eval_threshold(1)
                t_par = bench_eval(X, t, n_repeat)

                speedup = t_seq / t_par if t_par > 0 else float("inf")
                results.append({
                    "n": n,
                    "n_points": n_points,
                    "eval": label,
                    "sequential_ms": t_seq * 1000,
                    "parallel_ms": t_par * 1000,
                    "speedup": speedup,
                })

                print(
                    f"  n={n:>6d}  pts={n_points:>4d}  {label:<12s}  "
                    f"seq={t_seq*1000:8.3f}ms  par={t_par*1000:8.3f}ms  "
                    f"speedup={speedup:.2f}x"
                )
        print()

    # Restore default
    mpcf.system.set_parallel_eval_threshold(500)
    return results


def plot_results(results, n_points_list, n_times):
    import matplotlib.pyplot as plt

    eval_types = ["scalar", f"array({n_times})"]
    fig, axes = plt.subplots(len(n_points_list), len(eval_types),
                             figsize=(12, 4 * len(n_points_list)),
                             squeeze=False)

    for row, n_points in enumerate(n_points_list):
        for col, eval_type in enumerate(eval_types):
            ax = axes[row, col]
            subset = [r for r in results
                      if r["eval"] == eval_type and r["n_points"] == n_points]
            ns = [r["n"] for r in subset]
            seq = [r["sequential_ms"] for r in subset]
            par = [r["parallel_ms"] for r in subset]

            ax.plot(ns, seq, "o-", label="sequential")
            ax.plot(ns, par, "s-", label="parallel")
            ax.set_xlabel("Tensor size")
            ax.set_ylabel("Time (ms)")
            ax.set_title(f"{eval_type}, {n_points} breakpoints")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.legend()
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig("bench_parallel_eval.png", dpi=150)
    print("Saved bench_parallel_eval.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark parallel vs sequential tensor_eval")
    parser.add_argument(
        "--sizes", nargs="+", type=int,
        default=[10, 50, 100, 200, 500, 1000, 2000, 5000],
        help="Tensor sizes to benchmark")
    parser.add_argument(
        "--n-points", nargs="+", type=int, default=[10, 50, 200],
        help="Number of breakpoints per PCF (multiple values supported)")
    parser.add_argument(
        "--n-times", type=int, default=20,
        help="Number of query times for array evaluation")
    parser.add_argument(
        "--n-repeat", type=int, default=7,
        help="Number of repeats per measurement (takes median)")
    parser.add_argument(
        "--n-cpus", type=int, default=6,
        help="Number of CPUs to use (default 6)")
    parser.add_argument(
        "--plot", action="store_true",
        help="Plot results with matplotlib")
    args = parser.parse_args()

    mpcf.system.limit_cpus(args.n_cpus)

    print(f"CPUs: {args.n_cpus}, breakpoints: {args.n_points}, "
          f"query times: {args.n_times}, repeats: {args.n_repeat}")
    print()

    results = run_benchmark(args.sizes, args.n_points, args.n_times,
                            args.n_repeat)

    if args.plot:
        plot_results(results, args.n_points, args.n_times)
