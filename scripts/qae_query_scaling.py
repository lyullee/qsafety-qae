"""Compare ideal ML-QAE and classical Monte Carlo at equal query budgets.

This is an oracle-query comparison, not a wall-clock or hardware-speed claim.
The QAE budget counts (2m+1) state-preparation/oracle-equivalent calls for each
Grover power m and each shot. Circuit synthesis costs are reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


def batched_mle(
    counts: np.ndarray,
    shots: int,
    powers: list[int],
    grid_points: int,
    batch_size: int = 100,
) -> np.ndarray:
    theta = np.linspace(1e-10, math.pi / 2 - 1e-10, grid_points)
    probabilities = np.asarray(
        [np.sin((2 * power + 1) * theta) ** 2 for power in powers]
    )
    probabilities = np.clip(probabilities, 1e-14, 1 - 1e-14)
    log_probability = np.log(probabilities)
    log_complement = np.log1p(-probabilities)
    estimates = np.empty(counts.shape[0], dtype=float)
    for start in range(0, counts.shape[0], batch_size):
        stop = min(start + batch_size, counts.shape[0])
        batch = counts[start:stop]
        likelihood = batch @ log_probability
        likelihood += (shots - batch) @ log_complement
        best = np.argmax(likelihood, axis=1)
        estimates[start:stop] = np.sin(theta[best]) ** 2
    return estimates


def error_summary(estimates: np.ndarray, exact: float) -> dict[str, float]:
    errors = estimates - exact
    return {
        "mean_estimate": float(np.mean(estimates)),
        "bias": float(np.mean(errors)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "q025": float(np.quantile(estimates, 0.025)),
        "q975": float(np.quantile(estimates, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--levels", type=int, nargs="+")
    parser.add_argument("--shots-per-power", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--grid-points", type=int)
    parser.add_argument("--unit-miles", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument(
        "--tag",
        default="",
        help="Optional filename suffix using letters, numbers, underscore, or hyphen",
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        raise ValueError("--tag may contain only letters, numbers, underscore, or hyphen")

    if args.mode == "quick":
        levels = args.levels or list(range(1, 8))
        shots = args.shots_per_power or 32
        repeats = args.repeats or 1000
        grid_points = args.grid_points or 16385
    else:
        levels = args.levels or list(range(1, 10))
        shots = args.shots_per_power or 64
        repeats = args.repeats or 10000
        grid_points = args.grid_points or 65537

    root = args.project_root
    risk = json.loads(
        (root / "results" / "classical_risk" / "pooled_risk.json").read_text(
            encoding="utf-8"
        )
    )
    unit_probability = 1 - math.exp(
        -risk["serious_rate_per_1000_mile_year"] * args.unit_miles / 1000
    )
    rng = np.random.default_rng(args.seed)
    rows = []

    for size in args.sizes:
        exact = 1 - (1 - unit_probability) ** size
        amplitude_angle = math.asin(math.sqrt(exact))
        for level in levels:
            powers = [0] + [2**j for j in range(level - 1)]
            amplified = np.asarray(
                [math.sin((2 * power + 1) * amplitude_angle) ** 2 for power in powers]
            )
            qae_counts = np.column_stack(
                [rng.binomial(shots, probability, repeats) for probability in amplified]
            )
            qae_estimates = batched_mle(
                qae_counts, shots, powers, grid_points
            )
            query_budget = shots * sum(2 * power + 1 for power in powers)
            classical_estimates = rng.binomial(
                query_budget, exact, repeats
            ) / query_budget

            for method, estimates in (
                ("classical_mc", classical_estimates),
                ("ideal_mlqae", qae_estimates),
            ):
                rows.append(
                    {
                        "risk_qubits": size,
                        "binary_scenarios": 2**size,
                        "exact_probability": exact,
                        "method": method,
                        "level": level,
                        "powers": "|".join(map(str, powers)),
                        "shots_per_power": shots,
                        "query_budget": query_budget,
                        "repetitions": repeats,
                        **error_summary(estimates, exact),
                    }
                )
            print(
                f"finished K={size}, level={level}, query_budget={query_budget}"
            )

    frame = pd.DataFrame(rows)
    slope_rows = []
    for (size, method), group in frame.groupby(["risk_qubits", "method"]):
        valid = group[(group["rmse"] > 0) & (group["level"] >= 2)]
        slope, intercept = np.polyfit(
            np.log(valid["query_budget"]), np.log(valid["rmse"]), 1
        )
        slope_rows.append(
            {
                "risk_qubits": size,
                "binary_scenarios": 2**size,
                "method": method,
                "loglog_rmse_slope": float(slope),
                "loglog_intercept": float(intercept),
                "fit_levels": "level>=2",
            }
        )
    slopes = pd.DataFrame(slope_rows)

    paired = frame.pivot_table(
        index=["risk_qubits", "binary_scenarios", "query_budget"],
        columns="method",
        values="rmse",
    ).reset_index()
    paired["rmse_ratio_classical_over_qae"] = (
        paired["classical_mc"] / paired["ideal_mlqae"]
    )

    output_dir = root / "results" / "qae_risk" / "query_scaling"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    scaling_path = output_dir / f"query_scaling_{args.mode}{suffix}.csv"
    slopes_path = output_dir / f"query_slopes_{args.mode}{suffix}.csv"
    paired_path = output_dir / f"query_rmse_ratio_{args.mode}{suffix}.csv"
    frame.to_csv(scaling_path, index=False)
    slopes.to_csv(slopes_path, index=False)
    paired.to_csv(paired_path, index=False)
    print(slopes.to_string(index=False))
    print(f"scaling: {scaling_path}")
    print(f"slopes: {slopes_path}")
    print(f"RMSE ratios: {paired_path}")


if __name__ == "__main__":
    main()
