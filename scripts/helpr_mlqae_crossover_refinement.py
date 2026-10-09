"""Refine the ideal ML-QAE/importance crossover on a frozen dense budget grid."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from helpr_mlqae_importance_boundary import simulate_cell


PROBABILITIES = [
    0.006915260456630194,
    0.007970272865693958,
    0.009025285274757721,
]
TARGET_BUDGETS = [
    16_384,
    20_480,
    24_576,
    28_672,
    32_768,
    36_864,
    40_960,
    45_056,
    49_152,
    53_248,
    57_344,
    61_440,
    65_536,
]
PRECISIONS = [
    0.005,
    0.006,
    0.007,
    0.008,
    0.009,
    0.010,
    0.012,
    0.014,
    0.016,
    0.018,
    0.020,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5000)
    parser.add_argument("--grid-points", type=int, default=65_537)
    parser.add_argument("--seed", type=int, default=36001)
    parser.add_argument("--schedule-ratio", type=float, default=1.8)
    parser.add_argument("--probability-upper", type=float, default=0.02)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for probability in PROBABILITIES:
        for budget in TARGET_BUDGETS:
            row, estimates = simulate_cell(
                probability,
                args.schedule_ratio,
                budget,
                args.repeats,
                args.grid_points,
                args.probability_upper,
                rng,
            )
            squared_errors = np.square(estimates - probability)
            rmse = float(row["rmse"])
            mse_se = float(squared_errors.std(ddof=1) / math.sqrt(args.repeats))
            rows.append(
                {
                    **row,
                    "rmse_standard_error": mse_se / (2 * rmse) if rmse else 0.0,
                }
            )
            print(
                f"finished p={probability:.8g}, target={budget}, "
                f"actual={row['actual_query_budget']}",
                flush=True,
            )
    validation = pd.DataFrame(rows)
    critical_z = NormalDist().inv_cdf(1 - 0.05 / (2 * len(validation)))
    validation["familywise_relative_rmse_upper"] = (
        validation["rmse"] + critical_z * validation["rmse_standard_error"]
    ) / validation["probability"]

    robust = (
        validation.groupby("actual_query_budget", as_index=False)
        .agg(
            target_query_budget=("target_query_budget", "first"),
            maximum_power=("maximum_power", "max"),
            shots_per_power=("shots_per_power", "first"),
            maximum_relative_rmse=("relative_rmse", "max"),
            maximum_familywise_relative_rmse_upper=(
                "familywise_relative_rmse_upper",
                "max",
            ),
        )
        .sort_values("actual_query_budget")
    )

    importance_summary = json.loads(
        (
            args.project_root
            / "results"
            / "helpr"
            / "importance_sampling"
            / "importance_validation_summary.json"
        ).read_text(encoding="utf-8")
    )
    probability = float(importance_summary["probability_used_for_query_costing"])
    variance_reduction = float(importance_summary["estimated_variance_reduction_factor"])
    importance_variance = probability * (1 - probability) / variance_reduction
    boundary_rows = []
    for precision in PRECISIONS:
        passing = robust.loc[
            robust["maximum_familywise_relative_rmse_upper"].le(precision)
        ].sort_values("actual_query_budget")
        qae_queries = int(passing.iloc[0]["actual_query_budget"]) if len(passing) else np.nan
        importance_queries = math.ceil(
            importance_variance / (probability * precision) ** 2
        )
        boundary_rows.append(
            {
                "target_relative_rmse": precision,
                "validated_qae_queries": qae_queries,
                "strong_importance_queries": importance_queries,
                "importance_to_qae_query_ratio": (
                    importance_queries / qae_queries if np.isfinite(qae_queries) else np.nan
                ),
                "qae_query_advantage": bool(
                    np.isfinite(qae_queries) and qae_queries < importance_queries
                ),
            }
        )
    boundary = pd.DataFrame(boundary_rows)

    output_dir = args.project_root / "results" / "helpr" / "mlqae_crossover_refinement"
    output_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output_dir / "dense_validation_full.csv", index=False)
    robust.to_csv(output_dir / "robust_budget_envelope_full.csv", index=False)
    boundary.to_csv(output_dir / "refined_crossover_full.csv", index=False)
    metadata = {
        "schedule_ratio": args.schedule_ratio,
        "repeats": args.repeats,
        "grid_points": args.grid_points,
        "familywise_cells": len(validation),
        "familywise_critical_z": critical_z,
        "coarsest_relative_rmse_target_with_qae_query_advantage_on_tested_grid": (
            float(boundary.loc[boundary["qae_query_advantage"], "target_relative_rmse"].max())
            if boundary["qae_query_advantage"].any()
            else None
        ),
        "claim_limit": "Ideal oracle-query crossover only; circuit and state-preparation costs excluded.",
    }
    (output_dir / "metadata_full.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nRobust budget envelope")
    print(robust.to_string(index=False))
    print("\nRefined crossover")
    print(boundary.to_string(index=False))
    print("\nMetadata")
    print(json.dumps(metadata, indent=2))
    print(f"\nresults: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
