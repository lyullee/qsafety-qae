"""Ideal ML-QAE scaling for the HyRAM+-anchored rare trajectory amplitude.

High Grover powers are evaluated analytically through binomial count models;
their circuits are not materialized or executed. Resource estimates extrapolate
the exactly transpiled m=0,1,2 factorized explicit-trajectory circuits.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research_schedule import schedule_for_ratio


TARGET_BUDGETS = [
    32_768,
    65_536,
    131_072,
    262_144,
    524_288,
    1_048_576,
    2_097_152,
    4_194_304,
]
PRECISIONS = [0.50, 0.25, 0.10, 0.05]


def bounded_batched_mle(
    counts: np.ndarray,
    shots: int,
    powers: list[int],
    probability_upper: float,
    grid_points: int,
    batch_size: int = 100,
) -> np.ndarray:
    """Grid MLE on a predeclared rare-event probability interval."""
    theta = np.linspace(1e-12, math.asin(math.sqrt(probability_upper)), grid_points)
    probabilities = np.asarray(
        [np.sin((2 * power + 1) * theta) ** 2 for power in powers]
    )
    probabilities = np.clip(probabilities, 1e-15, 1 - 1e-15)
    log_probability = np.log(probabilities)
    log_complement = np.log1p(-probabilities)
    estimates = np.empty(counts.shape[0], dtype=float)
    for start in range(0, len(counts), batch_size):
        stop = min(start + batch_size, len(counts))
        batch = counts[start:stop]
        likelihood = batch @ log_probability
        likelihood += (shots - batch) @ log_complement
        best = np.argmax(likelihood, axis=1)
        estimates[start:stop] = np.sin(theta[best]) ** 2
    return estimates


def summarize(estimates: np.ndarray, exact: float) -> dict[str, float]:
    errors = estimates - exact
    return {
        "mean_estimate": float(estimates.mean()),
        "bias": float(errors.mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "relative_rmse": float(np.sqrt(np.mean(errors**2)) / exact),
        "mae": float(np.mean(np.abs(errors))),
        "q025": float(np.quantile(estimates, 0.025)),
        "q975": float(np.quantile(estimates, 0.975)),
        "zero_estimate_fraction": float(np.mean(estimates <= 1e-20)),
    }


def resource_linear_model(resources: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    selected = resources.loc[
        resources["method"].eq("explicit_trajectory__factorized_state")
    ].sort_values("grover_power")
    if selected["grover_power"].tolist() != [0, 1, 2]:
        raise ValueError("Resource extrapolation requires exactly m=0,1,2 rows")
    slope, intercept = np.polyfit(selected["grover_power"], selected[metric], 1)
    predicted = intercept + slope * selected["grover_power"].to_numpy()
    residual = selected[metric].to_numpy() - predicted
    r_squared = 1 - float(np.sum(residual**2)) / float(
        np.sum((selected[metric] - selected[metric].mean()) ** 2)
    )
    return float(intercept), float(slope), r_squared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--grid-points", type=int)
    parser.add_argument("--probability-upper", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    repeats = args.repeats or (500 if args.mode == "quick" else 10_000)
    grid_points = args.grid_points or (32_769 if args.mode == "quick" else 65_537)
    root = args.project_root

    trajectory_dir = root / "results" / "hyram" / "stochastic_trajectory_oracle"
    trajectory = json.loads(
        (trajectory_dir / "trajectory_summary.json").read_text(encoding="utf-8")
    )
    exact = float(trajectory["random_opportunity_hazard_probability"])
    if not 0 < exact < args.probability_upper < 1:
        raise ValueError("Exact amplitude must lie inside the predeclared probability bound")
    resources = pd.read_csv(trajectory_dir / "circuit_resources.csv")
    models = {
        metric: resource_linear_model(resources, metric)
        for metric in ("transpiled_depth", "transpiled_gate_count", "transpiled_cx")
    }

    rng = np.random.default_rng(args.seed)
    theta = math.asin(math.sqrt(exact))
    rows: list[dict] = []
    for target_budget in TARGET_BUDGETS:
        powers, shots, actual_budget = schedule_for_ratio(1.5, target_budget)
        amplified = np.asarray(
            [math.sin((2 * power + 1) * theta) ** 2 for power in powers]
        )
        counts = np.column_stack(
            [rng.binomial(shots, value, repeats) for value in amplified]
        )
        qae_estimates = bounded_batched_mle(
            counts,
            shots,
            powers,
            args.probability_upper,
            grid_points,
        )
        classical_estimates = rng.binomial(actual_budget, exact, repeats) / actual_budget
        for method, estimates in (
            ("classical_bernoulli_mc", classical_estimates),
            ("ideal_bounded_mlqae", qae_estimates),
        ):
            row = {
                "method": method,
                "exact_probability": exact,
                "probability_search_upper": args.probability_upper,
                "target_query_budget": target_budget,
                "actual_query_budget": actual_budget,
                "powers": "|".join(map(str, powers)),
                "maximum_power": max(powers),
                "shots_per_power": shots,
                "repetitions": repeats,
                "grid_points": grid_points,
                **summarize(estimates, exact),
            }
            if method == "ideal_bounded_mlqae":
                for metric, (intercept, slope, r_squared) in models.items():
                    values = intercept + slope * np.asarray(powers)
                    row[f"estimated_max_circuit_{metric}"] = float(values.max())
                    row[f"estimated_total_executed_{metric}"] = float(shots * values.sum())
                    row[f"resource_linear_fit_{metric}_r_squared"] = r_squared
            rows.append(row)
        print(
            f"finished target_budget={target_budget}, actual_budget={actual_budget}, "
            f"max_power={max(powers)}",
            flush=True,
        )

    scaling = pd.DataFrame(rows)
    qae = scaling.loc[scaling["method"].eq("ideal_bounded_mlqae")].copy()
    classical = scaling.loc[scaling["method"].eq("classical_bernoulli_mc")].copy()
    precision_rows: list[dict] = []
    for precision in PRECISIONS:
        target = precision * exact
        qae_pass = qae.loc[qae["rmse"].le(target)].sort_values("actual_query_budget")
        qae_budget = int(qae_pass.iloc[0]["actual_query_budget"]) if len(qae_pass) else np.nan
        classical_budget = math.ceil(exact * (1 - exact) / target**2)
        precision_rows.append(
            {
                "target_relative_rmse": precision,
                "target_absolute_rmse": target,
                "minimum_observed_qae_queries": qae_budget,
                "minimum_theoretical_classical_queries": classical_budget,
                "classical_to_qae_query_ratio": (
                    classical_budget / qae_budget if np.isfinite(qae_budget) else np.nan
                ),
                "qae_target_observed_in_grid": bool(len(qae_pass)),
            }
        )
    precision = pd.DataFrame(precision_rows)

    slope_rows: list[dict] = []
    for method, group in scaling.groupby("method"):
        valid = group.loc[group["rmse"].gt(0)].sort_values("actual_query_budget")
        slope, intercept = np.polyfit(
            np.log(valid["actual_query_budget"]), np.log(valid["rmse"]), 1
        )
        slope_rows.append(
            {
                "method": method,
                "loglog_rmse_slope": float(slope),
                "loglog_intercept": float(intercept),
                "fit_points": len(valid),
            }
        )
    slopes = pd.DataFrame(slope_rows)

    output_dir = root / "results" / "hyram" / "rare_trajectory_qae_scaling"
    output_dir.mkdir(parents=True, exist_ok=True)
    scaling.to_csv(output_dir / f"query_scaling_{args.mode}.csv", index=False)
    precision.to_csv(output_dir / f"precision_boundary_{args.mode}.csv", index=False)
    slopes.to_csv(output_dir / f"query_slopes_{args.mode}.csv", index=False)
    (output_dir / f"resource_models_{args.mode}.json").write_text(
        json.dumps(
            {
                metric: {"intercept": values[0], "slope_per_grover_power": values[1], "r_squared": values[2]}
                for metric, values in models.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nPrecision boundary")
    print(precision.to_string(index=False))
    print("\nQuery slopes")
    print(slopes.to_string(index=False))
    print("\nLargest QAE schedule")
    print(
        qae.sort_values("actual_query_budget").tail(1)[[
            "actual_query_budget", "maximum_power", "shots_per_power",
            "relative_rmse", "estimated_max_circuit_transpiled_depth",
            "estimated_max_circuit_transpiled_cx",
        ]].to_string(index=False)
    )
    print(f"\nresults: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
