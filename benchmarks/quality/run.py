"""
Quality benchmark: PR-AUC, ROC-AUC on real/synthetic datasets.

Usage:
    python -m benchmarks.quality.run
    python -m benchmarks.quality.run --domain tabular
    python -m benchmarks.quality.run --domain tabular --dataset thyroid
    python -m benchmarks.quality.run --domain tabular --dataset all
    python -m benchmarks.quality.run --domain all --dataset synthetic
    python -m benchmarks.quality.run --domain timeseries --dataset nab_ec2
    python -m benchmarks.quality.run --domain all --dataset all --n-runs 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, cast

import pandas as pd

from benchmarks.data.registry import DATA_LOADERS

DOMAIN_ALGOS: dict[str, list[dict[str, Any]]] = {
    "tabular": [
        {"name": "IsolationForest", "kwargs": {"random_state": 42, "n_jobs": 1}},
    ],
    "timeseries": [
        {
            "name": "LSTM",
            "kwargs": {
                "window_size": 50,
                "epochs": 20,
                "hidden_dim": 32,
                "random_state": 42,
                "verbose": 0,
            },
        },
    ],
    "text": [
        {"name": "TfidfDetector", "kwargs": {"random_state": 42}},
    ],
    "cv": [
        {
            "name": "ConvAutoencoder",
            "kwargs": {
                "hidden_dim": 32,
                "epochs": 10,
                "verbose": 0,
                "random_state": 42,
            },
        },
    ],
}


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
        timeout=600,
    )

    if proc.returncode != 0:
        return {
            "error": proc.stderr or f"Worker exited with code {proc.returncode}",
            "algorithm": config.get("algorithm", {}).get("name", "unknown"),
        }

    try:
        return cast(dict[str, Any], json.loads(proc.stdout))
    except json.JSONDecodeError:
        return {
            "error": f"Invalid JSON: {proc.stdout[:200]}",
            "algorithm": config.get("algorithm", {}).get("name", "unknown"),
        }


def run_quality(
    domains: list[str],
    datasets: list[str],
    n_runs: int = 3,
) -> pd.DataFrame:
    """Run quality benchmarks across domains and datasets."""
    results: list[dict[str, Any]] = []

    for domain in domains:
        if domain not in DOMAIN_ALGOS:
            print(f"WARNING: No algorithms configured for domain '{domain}'")
            continue

        available_datasets = list(DATA_LOADERS.get(domain, {}).keys())

        target_datasets = (
            available_datasets
            if "all" in datasets
            else [d for d in datasets if d in available_datasets]
        )

        if not target_datasets:
            print(f"WARNING: No matching datasets for domain '{domain}'")
            continue

        for ds_name in target_datasets:
            print(f"\n--- {domain} / {ds_name} ---")

            for algo_spec in DOMAIN_ALGOS[domain]:
                config: dict[str, Any] = {
                    "mode": "quality",
                    "algorithm": {
                        "name": algo_spec["name"],
                        "domain": domain,
                        "kwargs": algo_spec["kwargs"],
                    },
                    "dataset": ds_name,
                    "domain": domain,
                    "n_runs": n_runs,
                    "warmup_runs": 1,
                    "native_baseline": None,
                    "n_samples": None,
                    "n_features": None,
                }

                print(f"  {algo_spec['name']}...", end=" ", flush=True)
                result = _run_worker(config)

                if result.get("error"):
                    print("ERROR")
                else:
                    pr = result.get("pr_auc", 0)
                    roc = result.get("roc_auc", 0)
                    ft = result.get("fit_time", 0)
                    print(f"PR-AUC={pr:.4f} ROC-AUC={roc:.4f} fit={ft:.2f}s")

                result.setdefault("algorithm", algo_spec["name"])
                result.setdefault("dataset", ds_name)
                result.setdefault("domain", domain)
                results.append(result)

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniAD Quality Benchmark")
    parser.add_argument(
        "--domain",
        nargs="+",
        default=["tabular"],
        help="Domains to benchmark (tabular, timeseries, text, cv, or 'all')",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["synthetic"],
        help="Dataset names or 'all'",
    )
    parser.add_argument("--n-runs", type=int, default=3)

    args = parser.parse_args()

    domains = list(DOMAIN_ALGOS.keys()) if "all" in args.domain else args.domain

    print(f"\n{'='*60}")
    print("OmniAD Quality Benchmark")
    print(f"Domains: {domains}")
    print(f"Datasets: {args.dataset}")
    print(f"{'='*60}")

    df = run_quality(domains, args.dataset, args.n_runs)

    if not df.empty:
        cols = [
            "algorithm",
            "dataset",
            "domain",
            "pr_auc",
            "roc_auc",
            "fit_time",
            "predict_time",
            "ram_mb",
        ]
        display_cols = [c for c in cols if c in df.columns]
        print(f"\n{df[display_cols].to_string(index=False)}\n")

    os.makedirs("benchmarks/results", exist_ok=True)
    csv_path = "benchmarks/results/quality_latest.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()
