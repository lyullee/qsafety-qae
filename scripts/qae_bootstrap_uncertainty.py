#!/usr/bin/env python
"""Propagate clustered PHMSA bootstrap uncertainty through ideal MLQAE and MC.

The bootstrap draw is the data/model uncertainty unit.  For every draw, ideal
MLQAE and classical Monte Carlo are independently sampled at an equal
oracle-query budget.  This separates conditional algorithmic sampling error from
the wider uncertainty of the safety-data model.  No IBM QPU is contacted.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "results" / "psp_bootstrap"


def batched_mle(
    counts: np.ndarray,
    shots: int,
    powers: list[int],
    grid_points: int,
    batch_size: int = 100,
) -> np.ndarray:
    theta = np.linspace(1e-10, math.pi / 2 - 1e-10, grid_points)
    probabilities = np.asarray([np.sin((2 * power + 1) * theta) ** 2 for power in powers])
    probabilities = np.clip(probabilities, 1e-14, 1 - 1e-14)
    log_probability = np.log(probabilities)
    log_complement = np.log1p(-probabilities)
    estimates = np.empty(counts.shape[0], dtype=float)
    for start in range(0, len(counts), batch_size):
        stop = min(start + batch_size, len(counts))
        batch = counts[start:stop]
        likelihood = batch @ log_probability + (shots - batch) @ log_complement
        estimates[start:stop] = np.sin(theta[np.argmax(likelihood, axis=1)]) ** 2
    return estimates


def quantile(values: np.ndarray, level: float) -> float:
    return float(np.quantile(values, level))


def bootstrap_probabilities(scenario_qubits: list[int], max_bootstraps: int | None) -> pd.DataFrame:
    raw = pd.read_csv(INPUT_DIR / "risk_strata_bootstrap_raw_full.csv")
    base = pd.read_csv(INPUT_DIR / "risk_strata_full.csv")
    raw = raw[raw.scenario_qubits.isin(scenario_qubits)].copy()
    base = base[["scenario_qubits", "stratum", "weight"]].copy()
    merged = raw.merge(base, on=["scenario_qubits", "stratum"], how="inner", validate="many_to_one")
    probabilities = merged.assign(weighted=lambda frame: frame.weight * frame.probability_mean).groupby(
        ["repeat", "scenario_qubits"], as_index=False
    ).weighted.sum().rename(columns={"weighted": "bootstrap_probability"})
    if max_bootstraps is not None:
        keep = sorted(probabilities.repeat.unique())[:max_bootstraps]
        probabilities = probabilities[probabilities.repeat.isin(keep)].copy()
    return probabilities.sort_values(["scenario_qubits", "repeat"]).reset_index(drop=True)


def method_rows(
    probabilities: pd.DataFrame,
    method: str,
    shots: int,
    powers: list[int],
    algorithm_repeats: int,
    grid_points: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    expanded = probabilities.loc[probabilities.index.repeat(algorithm_repeats)].copy()
    expanded["algorithm_repeat"] = np.tile(np.arange(algorithm_repeats), len(probabilities))
    exact = expanded.bootstrap_probability.to_numpy(float)
    if method == "ideal_mlqae":
        theta = np.arcsin(np.sqrt(exact))
        probability_matrix = np.column_stack([
            np.sin((2 * power + 1) * theta) ** 2 for power in powers
        ])
        counts = rng.binomial(shots, probability_matrix)
        estimate = batched_mle(counts, shots, powers, grid_points)
        query_budget = shots * sum(2 * power + 1 for power in powers)
    elif method == "classical_mc":
        query_budget = shots * sum(2 * power + 1 for power in powers)
        estimate = rng.binomial(query_budget, exact) / query_budget
    else:
        raise ValueError(method)
    expanded["method"] = method
    expanded["query_budget"] = query_budget
    expanded["estimate"] = estimate
    expanded["signed_algorithm_error"] = estimate - exact
    expanded["absolute_algorithm_error"] = np.abs(estimate - exact)
    return expanded


def summary_rows(probabilities: pd.DataFrame, estimates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for n, data_group in probabilities.groupby("scenario_qubits"):
        data = data_group.bootstrap_probability.to_numpy(float)
        for method, estimate_group in estimates[estimates.scenario_qubits == n].groupby("method"):
            errors = estimate_group.signed_algorithm_error.to_numpy(float)
            values = estimate_group.estimate.to_numpy(float)
            data_std = float(np.std(data, ddof=1))
            rmse = float(np.sqrt(np.mean(errors**2)))
            rows.append({
                "scenario_qubits": int(n),
                "risk_strata": int(2**n),
                "bootstrap_draws": int(len(data)),
                "algorithm_repeats_per_draw": int(estimate_group.algorithm_repeat.nunique()),
                "method": method,
                "data_probability_q025": quantile(data, 0.025),
                "data_probability_median": quantile(data, 0.5),
                "data_probability_q975": quantile(data, 0.975),
                "data_probability_std": data_std,
                "algorithm_bias": float(np.mean(errors)),
                "algorithm_mae": float(np.mean(np.abs(errors))),
                "algorithm_rmse": rmse,
                "algorithm_rmse_over_data_std": rmse / data_std if data_std else np.nan,
                "combined_estimate_q025": quantile(values, 0.025),
                "combined_estimate_median": quantile(values, 0.5),
                "combined_estimate_q975": quantile(values, 0.975),
                "query_budget": int(estimate_group.query_budget.iloc[0]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--scenario-qubits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--powers", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--algorithm-repeats", type=int)
    parser.add_argument("--grid-points", type=int)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if args.mode == "quick":
        max_bootstraps = 100
        repeats = args.algorithm_repeats or 5
        grid_points = args.grid_points or 16385
    else:
        max_bootstraps = None
        repeats = args.algorithm_repeats or 30
        grid_points = args.grid_points or 65537
    if args.shots < 1 or repeats < 1:
        raise ValueError("shots and algorithm repeats must be positive")

    probabilities = bootstrap_probabilities(args.scenario_qubits, max_bootstraps)
    rng = np.random.default_rng(args.seed)
    outputs = [
        method_rows(probabilities, "classical_mc", args.shots, args.powers, repeats, grid_points, rng),
        method_rows(probabilities, "ideal_mlqae", args.shots, args.powers, repeats, grid_points, rng),
    ]
    estimates = pd.concat(outputs, ignore_index=True)
    summary = summary_rows(probabilities, estimates)

    suffix = f"{args.mode}_s{args.shots}_r{repeats}"
    probability_path = INPUT_DIR / f"qae_bootstrap_probabilities_{suffix}.csv"
    estimate_path = INPUT_DIR / f"qae_bootstrap_estimates_{suffix}.csv"
    summary_path = INPUT_DIR / f"qae_bootstrap_uncertainty_summary_{suffix}.csv"
    probabilities.to_csv(probability_path, index=False)
    estimates.to_csv(estimate_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Bootstrap probabilities: {probability_path}")
    print(f"Estimates: {estimate_path}")
    print(f"Summary: {summary_path}")
    print("No QPU workload was submitted.")


if __name__ == "__main__":
    main()
