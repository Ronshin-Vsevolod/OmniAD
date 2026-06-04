"""
Performance benchmark: scalability, model size, GPU vs CPU.

Usage:
    python -m benchmarks.performance.run --algo IsolationForest --mode scalability
    python -m benchmarks.performance.run --algo LSTM --mode scalability
    python -m benchmarks.performance.run --algo all --mode scalability --n-runs 5
    python -m benchmarks.performance.run --mode model_size
    python -m benchmarks.performance.run --mode model_size --algo LSTM
    python -m benchmarks.performance.run --mode model_size --domain tabular timeseries
    python -m benchmarks.performance.run --algo LSTM --mode gpu_vs_cpu
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, cast

import pandas as pd

SCALABILITY_SIZES = [1000, 5000, 10000, 50000, 100000]

PERF_ALGOS: dict[str, dict[str, Any]] = {
    "IsolationForest": {
        "kwargs": {"random_state": 42, "n_jobs": 1},
        "domain": "tabular",
    },
    "LSTM": {
        "kwargs": {
            "window_size": 50,
            "epochs": 5,
            "hidden_dim": 32,
            "random_state": 42,
            "verbose": 0,
        },
        "domain": "timeseries",
    },
    "TfidfDetector": {
        "kwargs": {"random_state": 42},
        "domain": "text",
    },
    "ConvAutoencoder": {
        "kwargs": {"hidden_dim": 32, "epochs": 10, "verbose": 0, "random_state": 42},
        "domain": "cv",
    },
}


def _run_worker(config: dict[str, Any]) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.worker"],
        input=json.dumps(config),
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )

    if proc.returncode != 0:
        return {"error": proc.stderr or f"Exit code {proc.returncode}"}

    try:
        return cast(dict[str, Any], json.loads(proc.stdout))
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON: {proc.stdout[:200]}"}


def run_scalability(
    algos: list[str],
    sizes: list[int],
    n_runs: int = 3,
) -> pd.DataFrame:
    """Measure fit/predict time at different data sizes."""
    results: list[dict[str, Any]] = []

    for algo_name in algos:
        if algo_name not in PERF_ALGOS:
            print(f"Unknown algorithm: {algo_name}")
            continue

        spec = PERF_ALGOS[algo_name]
        print(f"\n--- {algo_name} ---")

        for n in sizes:
            config: dict[str, Any] = {
                "mode": "performance",
                "algorithm": {
                    "name": algo_name,
                    "domain": spec["domain"],
                    "kwargs": spec["kwargs"],
                },
                "dataset": "synthetic",
                "domain": spec["domain"],
                "n_runs": n_runs,
                "warmup_runs": 1,
                "native_baseline": None,
                "n_samples": n,
                "n_features": 20,
            }

            print(f"  N={n:>8}...", end=" ", flush=True)
            result = _run_worker(config)

            if result.get("error"):
                print("ERROR")
            else:
                ft = result.get("fit_time", 0)
                pt = result.get("predict_time", 0)
                ram = result.get("ram_mb", 0)
                print(f"fit={ft:.3f}s predict={pt:.4f}s ram={ram:.0f}MB")

            result.setdefault("algorithm", algo_name)
            results.append(result)

    return pd.DataFrame(results)


def run_model_size(
    domains: list[str],
    algo_name: str = "all",
    n_samples: int = 5000,
) -> pd.DataFrame:
    """Measure model archive size, save/load time."""
    results: list[dict[str, Any]] = []

    for name, spec in PERF_ALGOS.items():
        if algo_name != "all" and name != algo_name:
            continue
        if spec["domain"] not in domains:
            continue

        config: dict[str, Any] = {
            "mode": "performance",
            "algorithm": {
                "name": name,
                "domain": spec["domain"],
                "kwargs": spec["kwargs"],
            },
            "dataset": "synthetic",
            "domain": spec["domain"],
            "n_runs": 1,
            "warmup_runs": 0,
            "native_baseline": None,
            "n_samples": n_samples,
            "n_features": 20,
        }

        print(f"  {name} ({spec['domain']})...", end=" ", flush=True)
        result = _run_worker(config)

        if result.get("error"):
            print(f"ERROR: {result['error'][:100]}")
        else:
            ft = result.get("fit_time", 0)
            pt = result.get("predict_time", 0)
            sz = result.get("model_size_mb", 0)
            st = result.get("save_time", 0)
            lt = result.get("load_time", 0)
            ram = result.get("ram_mb", 0)
            print(
                f"fit={ft:.3f}s predict={pt:.4f}s "
                f"size={sz:.2f}MB save={st:.3f}s load={lt:.3f}s ram={ram:.0f}MB"
            )

        result.setdefault("algorithm", name)
        results.append(result)

    return pd.DataFrame(results)


def run_gpu_vs_cpu(
    algos: list[str],
    n_samples: int = 10000,
    n_runs: int = 3,
) -> pd.DataFrame:
    """Compare GPU vs CPU for a DL algorithm."""
    results: list[dict[str, Any]] = []

    for algo_name in algos:
        if algo_name not in PERF_ALGOS:
            print(f"Unknown algorithm: {algo_name}")
            return pd.DataFrame()

        spec = PERF_ALGOS[algo_name]

        for device in ["cpu", "cuda"]:
            kwargs = {**spec["kwargs"], "device": device}

            config: dict[str, Any] = {
                "mode": "performance",
                "algorithm": {
                    "name": algo_name,
                    "domain": spec["domain"],
                    "kwargs": kwargs,
                },
                "dataset": "synthetic",
                "domain": spec["domain"],
                "n_runs": n_runs,
                "warmup_runs": 2,
                "native_baseline": None,
                "n_samples": n_samples,
                "n_features": 20,
            }

            print(f"  {device}...", end=" ", flush=True)
            result = _run_worker(config)

            if result.get("error"):
                if device == "cuda" and "CUDA" in str(result.get("error", "")):
                    print("SKIPPED (no GPU)")
                    continue
                print("ERROR")
            else:
                ft = result.get("fit_time", 0)
                pt = result.get("predict_time", 0)
                vram = result.get("vram_mb", 0)
                print(f"fit={ft:.2f}s predict={pt:.4f}s vram={vram:.0f}MB")

            result["device"] = device
            result.setdefault("algorithm", algo_name)
            results.append(result)

    return pd.DataFrame(results)


def main() -> None:
    all_domains = sorted({s["domain"] for s in PERF_ALGOS.values()})

    parser = argparse.ArgumentParser(description="OmniAD Performance Benchmark")
    parser.add_argument(
        "--algo",
        default=["all"],
        help="Algorithm name or 'all' (for model_size mode)",
    )
    parser.add_argument(
        "--mode",
        choices=["scalability", "model_size", "gpu_vs_cpu"],
        default="scalability",
    )
    parser.add_argument(
        "--domain",
        nargs="+",
        default=["all"],
        help=f"Domains for model_size mode ({', '.join(all_domains)}, or 'all')",
    )
    parser.add_argument("--n-runs", type=int, default=3)

    args = parser.parse_args()

    domains = all_domains if "all" in args.domain else args.domain
    algos = list(PERF_ALGOS.keys()) if args.algo == "all" else [args.algo]

    print(f"\n{'=' * 60}")
    print(f"OmniAD Performance Benchmark: {args.mode}")
    print(f"Algorithms: {algos}")
    if args.mode == "model_size":
        print(f"Domains:   {domains}")
    print(f"{'=' * 60}\n")

    if args.mode == "scalability":
        df = run_scalability(algos, SCALABILITY_SIZES, args.n_runs)
    elif args.mode == "model_size":
        df = run_model_size(domains, args.algo)
    elif args.mode == "gpu_vs_cpu":
        df = run_gpu_vs_cpu(algos, n_runs=args.n_runs)
    else:
        df = pd.DataFrame()

    if not df.empty:
        if args.mode == "model_size":
            cols = [
                "algorithm",
                "domain",
                "fit_time",
                "predict_time",
                "model_size_mb",
                "save_time",
                "load_time",
                "ram_mb",
            ]
        else:
            cols = list(df.columns)
        display_cols = [c for c in cols if c in df.columns]
        print(f"\n{df[display_cols].to_string(index=False)}\n")

    os.makedirs("benchmarks/results", exist_ok=True)
    csv_path = f"benchmarks/results/perf_{args.mode}_{args.algo}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
