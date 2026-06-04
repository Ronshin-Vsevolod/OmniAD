"""
Overhead benchmark: omniad wrapper vs native library.

Usage:
    python -m benchmarks.overhead.run

    # Presets
    python -m benchmarks.overhead.run --preset fast
    python -m benchmarks.overhead.run --preset precise

    # Algorithms
    python -m benchmarks.overhead.run --algo IsolationForest
    python -m benchmarks.overhead.run --algo all

    # Datasets
    python -m benchmarks.overhead.run --dataset thyroid shuttle
    python -m benchmarks.overhead.run --algo LSTM --dataset synthetic nab_ec2

    # Repetitions
    python -m benchmarks.overhead.run --n-runs 5
    python -m benchmarks.overhead.run --n-runs 10 1
    python -m benchmarks.overhead.run --n-runs 20 5

    # Dataset sizes
    python -m benchmarks.overhead.run --n-samples 10000
    python -m benchmarks.overhead.run --n-samples 100000 1000

    # Stability
    python -m benchmarks.overhead.run --min-total-seconds 5.0

    # Diagnostics
    python -m benchmarks.overhead.run --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd

from benchmarks.data.registry import DATA_DIR

OVERHEAD_ALGOS: dict[str, dict[str, Any]] = {
    "IsolationForest": {
        "omniad_kwargs": {"n_estimators": 100, "random_state": 42, "n_jobs": 1},
        "native": {"runner": "benchmarks.overhead.natives.iforest_native"},
        "native_label": "sklearn IsolationForest",
        "domain": "tabular",
        "datasets": ["synthetic", "thyroid", "shuttle"],
    },
    "TfidfDetector": {
        "omniad_kwargs": {"random_state": 42},
        "native": {"runner": "benchmarks.overhead.natives.tfidf_native"},
        "native_label": "sklearn Tfidf+IForest",
        "domain": "text",
        "datasets": ["synthetic"],
    },
    "LSTM": {
        "omniad_kwargs": {
            "window_size": 50,
            "epochs": 5,
            "hidden_dim": 32,
            "random_state": 42,
            "verbose": 0,
        },
        "native": {"runner": "benchmarks.overhead.natives.lstm_native"},
        "native_label": "PyTorch LSTM",
        "domain": "timeseries",
        "datasets": ["synthetic", "nab_ec2"],
    },
    "ConvAutoencoder": {
        "omniad_kwargs": {
            "hidden_dim": 32,
            "epochs": 10,
            "verbose": 0,
            "random_state": 42,
        },
        "native": {"runner": "benchmarks.overhead.natives.conv_ae_native"},
        "native_label": "PyTorch ConvAE",
        "domain": "cv",
        "datasets": ["synthetic", "mnist_anomaly"],
    },
}


def _run_memory_probe(
    config: dict[str, Any],
    target: str,
) -> dict[str, Any]:
    probe = dict(config)
    probe["_memory_probe"] = target

    return _run_worker(probe)


def _run_worker(config: dict[str, Any]) -> dict[str, Any]:
    """Run a single worker subprocess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    proc = subprocess.run(
        [sys.executable, "-m", "benchmarks.worker"],
        input=json.dumps(config),
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )

    if proc.returncode != 0:
        return {"error": proc.stderr or f"Worker exited with code {proc.returncode}"}

    try:
        return cast(dict[str, Any], json.loads(proc.stdout))
    except json.JSONDecodeError:
        return {"error": f"Invalid JSON from worker: {proc.stdout[:200]}"}


def _dataset_available(domain: str, name: str) -> bool:
    """Check if a real dataset .npz exists (synthetic is always available)."""
    if name == "synthetic":
        return True
    return (DATA_DIR / domain / f"{name}.npz").exists()


def run_overhead(
    algos: list[str],
    datasets: list[str],
    throughput_samples: int = 10000,
    latency_samples: int = 600,
    n_features: int = 20,
    throughput_runs: int = 5,
    latency_runs: int = 1,
    min_total_seconds: float = 2.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run overhead benchmarks for selected algorithms × datasets."""
    results: list[dict[str, Any]] = []

    if verbose:
        print(
            "Note: actual runs = max(--n-runs, --min-total-seconds).\n"
            "Use --n-runs to set a fixed minimum, --min-total-seconds for stability.\n"
        )

    for algo_name in algos:
        if algo_name not in OVERHEAD_ALGOS:
            print(f"WARNING: No native baseline for '{algo_name}', skipping.")
            continue

        spec = OVERHEAD_ALGOS[algo_name]

        target_datasets = (
            spec["datasets"]
            if "all" in datasets
            else [d for d in datasets if d in spec["datasets"]]
        )

        if not target_datasets:
            print(
                f"WARNING: No matching datasets for '{algo_name}'. "
                f"Available: {', '.join(spec['datasets'])}"
            )
            continue

        for ds_name in target_datasets:
            if not _dataset_available(spec["domain"], ds_name):
                print(
                    f"  SKIP {algo_name} / {ds_name} "
                    f"(run: python benchmarks/prepare_datasets.py)"
                )
                continue

            for mode in ["throughput", "latency"]:
                current_runs = throughput_runs if mode == "throughput" else latency_runs

                current_samples = (
                    throughput_samples if mode == "throughput" else latency_samples
                )

                config: dict[str, Any] = {
                    "mode": mode,
                    "algorithm": {
                        "name": algo_name,
                        "domain": spec["domain"],
                        "kwargs": spec["omniad_kwargs"],
                    },
                    "dataset": ds_name,
                    "domain": spec["domain"],
                    "native_baseline": spec["native"],
                    "n_runs": current_runs,
                    "warmup_runs": 2,
                    "min_total_seconds": min_total_seconds,
                    "n_samples": current_samples,
                    "n_features": n_features,
                }

                label = spec["native_label"]

                print(
                    f"  [{mode}] {algo_name} / {ds_name} (vs {label})...",
                    end=" ",
                    flush=True,
                )

                result = _run_worker(config)

                if result.get("error"):
                    err = result["error"]
                    lines = err.strip().splitlines()
                    short = "\n".join(lines[-3:]) if len(lines) > 3 else err
                    print(f"ERROR:\n{short}")
                else:
                    # --- Memory probes ---

                    omniad_mem = _run_memory_probe(
                        config,
                        "omniad",
                    )

                    native_mem = _run_memory_probe(
                        config,
                        "native",
                    )

                    result["ram_mb"] = omniad_mem.get("ram_mb")
                    result["native_ram_mb"] = native_mem.get("ram_mb")

                    result["ram_overhead_mb"] = (
                        result["ram_mb"] - result["native_ram_mb"]
                    )

                    result["vram_mb"] = omniad_mem.get("vram_mb")
                    result["native_vram_mb"] = native_mem.get("vram_mb")

                    result["vram_overhead_mb"] = (
                        result["vram_mb"] - result["native_vram_mb"]
                    )

                    # --- Existing reporting ---

                    ratio = result.get("overhead_ratio", 0)

                    if verbose:
                        n_actual = result.get("n_actual_runs", 0)
                        overhead_ms = result.get("overhead_ms", 0)
                        parity_diff = result.get("parity_max_diff", 0)

                        print(
                            f"ratio={ratio:.4f}x "
                            f"overhead={overhead_ms:+.2f}ms "
                            f"parity_diff={parity_diff:.2e} "
                            f"runs={n_actual}"
                        )
                    else:
                        print(
                            f"ratio={ratio:.4f}x   "
                            f"parity_diff={result.get('parity_max_diff', 0):.2e}"
                        )

                result.setdefault("algorithm", algo_name)
                result.setdefault("dataset", ds_name)
                result["native_label"] = label

                results.append(result)

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniAD Overhead Benchmark")
    parser.add_argument(
        "--n-runs",
        nargs="+",
        type=int,
        default=[5, 1],
        help=(
            "Benchmark repetitions. "
            "'5' means throughput=5, latency=5. "
            "'5 1' means throughput=5, latency=1."
            "Default: 5 1"
        ),
    )
    parser.add_argument(
        "--algo",
        nargs="+",
        default=["all"],
        help="Algorithm names or 'all'",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        help="Dataset names or 'all' (default: all available for each algo)",
    )
    parser.add_argument(
        "--preset",
        choices=["fast", "precise"],
        default=None,
        help=(
            "Measurement preset. "
            "'fast': n_samples=10000, min_total=2s, expected ±2-3%%. "
            "'precise': n_samples=100000, min_total=30s, expected ±0.2%%. "
            "Overrides --n-samples, --n-runs, --min-total-seconds when set."
        ),
    )
    parser.add_argument(
        "--min-total-seconds",
        type=float,
        default=2.0,
        help=(
            "Minimum total time (seconds) spent timing each algorithm. "
            "Actual runs = max(--n-runs, runs needed for this duration). "
            "Increase for more stable results on fast operations."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help=("Print measurement details (actual runs, timing stability). "),
    )
    parser.add_argument(
        "--n-samples",
        nargs="+",
        type=int,
        default=[10000, 600],
        help=(
            "Number of samples.\n"
            "One value: applies to both modes.\n"
            "Two values: throughput latency.\n"
            "Default: 10000 600"
        ),
    )
    parser.add_argument("--n-features", type=int, default=20)

    args = parser.parse_args()

    runs = args.n_runs
    if len(runs) == 1:
        throughput_runs = runs[0]
        latency_runs = runs[0]
    else:
        throughput_runs = runs[0]
        latency_runs = runs[1]

    samples = args.n_samples
    if len(samples) == 1:
        throughput_samples = samples[0]
        latency_samples = samples[0]
    else:
        throughput_samples = samples[0]
        latency_samples = samples[1]

    if args.preset is not None:
        preset_path = Path(__file__).parent.parent / "presets" / f"{args.preset}.json"
        with open(preset_path) as f:
            preset = json.load(f)
        args.algo = ["all"]
        args.dataset = ["all"]
        args.n_samples = preset["n_samples"]
        args.n_runs = preset["n_runs"]
        args.min_total_seconds = preset["min_total_seconds"]
        print(f"Preset: {args.preset} — {preset['description']}")

    algos = list(OVERHEAD_ALGOS.keys()) if "all" in args.algo else args.algo

    print(f"\n{'=' * 60}")
    print("OmniAD Overhead Benchmark")
    print(f"Algorithms: {algos}")
    print(f"Datasets:   {args.dataset}")
    print(f"{'=' * 60}\n")

    df = run_overhead(
        algos,
        args.dataset,
        throughput_samples,
        latency_samples,
        args.n_features,
        throughput_runs,
        latency_runs,
        min_total_seconds=args.min_total_seconds,
        verbose=args.verbose,
    )
    if not df.empty:
        df_display = df.rename(
            columns={
                "n_actual_runs": "runs",
            }
        )

        # THROUGHPUT

        print("\n" + "=" * 60)
        print("THROUGHPUT BENCHMARKS")
        print("=" * 60)

        df_throughput = df_display[df_display["mode"] == "throughput"].copy()

        if not df_throughput.empty:
            df_throughput["overhead_ratio"] = df_throughput["overhead_ratio"].map(
                lambda x: f"{x:.3f}x" if x is not None else "N/A"
            )

            cols_tp = [
                "algorithm",
                "dataset",
                "native_label",
                "predict_time",
                "native_time",
                "overhead_ratio",
                "ram_overhead_mb",
                "vram_overhead_mb",
                "parity_check",
                "parity_max_diff",
                "runs",
            ]

            print(
                df_throughput[
                    [c for c in cols_tp if c in df_throughput.columns]
                ].to_string(index=False)
            )

        # LATENCY

        print("\n" + "=" * 60)
        print("LATENCY BENCHMARKS")
        print("=" * 60)

        df_latency = df_display[df_display["mode"] == "latency"].copy()

        if not df_latency.empty:
            cols_lat = [
                "algorithm",
                "dataset",
                "native_label",
                "predict_time",
                "native_time",
                "overhead_ms",
                "ram_mb",
                "vram_mb",
                "parity_check",
                "parity_max_diff",
                "runs",
            ]

            print(
                df_latency[[c for c in cols_lat if c in df_latency.columns]].to_string(
                    index=False
                )
            )

        os.makedirs("benchmarks/results", exist_ok=True)

        csv_path = "benchmarks/results/overhead_latest.csv"

        df.to_csv(
            csv_path,
            index=False,
        )

        print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
