"""Test rare-event ML-QAE sensitivity to the predeclared search upper bound."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from hyram_rare_trajectory_qae_scaling import bounded_batched_mle
from research_schedule import schedule_for_ratio


PROBABILITY_UPPERS = [1e-5, 1e-4, 1e-3, 1e-2]
TARGET_BUDGETS = [131_072, 262_144, 524_288, 1_048_576]


def rmse_and_se(estimates: np.ndarray, exact: float) -> tuple[float, float]:
    squared = np.square(estimates - exact)
    mse = float(squared.mean())
    rmse = math.sqrt(mse)
    mse_se = float(squared.std(ddof=1) / math.sqrt(len(squared)))
    return rmse, mse_se / (2 * rmse) if rmse else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--repeats-per-seed", type=int)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--grid-points", type=int)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    repeats = args.repeats_per_seed or (500 if args.mode == "quick" else 2_000)
    seeds = args.seeds or (2 if args.mode == "quick" else 5)
    grid_points = args.grid_points or (32_769 if args.mode == "quick" else 65_537)
    root = args.project_root
    trajectory = json.loads(
        (
            root / "results" / "hyram" / "stochastic_trajectory_oracle"
            / "trajectory_summary.json"
        ).read_text(encoding="utf-8")
    )
    exact = float(trajectory["random_opportunity_hazard_probability"])
    if any(bound <= exact for bound in PROBABILITY_UPPERS):
        raise ValueError("Every tested upper bound must exceed the exact probability")

    rows: list[dict] = []
    for budget_index, target_budget in enumerate(TARGET_BUDGETS):
        powers, shots, actual_budget = schedule_for_ratio(1.5, target_budget)
        theta = math.asin(math.sqrt(exact))
        amplified = [
            math.sin((2 * power + 1) * theta) ** 2 for power in powers
        ]
        for seed_index in range(seeds):
            derived_seed = args.seed + 100_000 * budget_index + seed_index
            rng = np.random.default_rng(derived_seed)
            # Common count data across all prior bounds isolate the effect of
            # the likelihood search interval itself.
            counts = np.column_stack(
                [rng.binomial(shots, value, repeats) for value in amplified]
            )
            for upper in PROBABILITY_UPPERS:
                estimates = bounded_batched_mle(
                    counts,
                    shots,
                    powers,
                    upper,
                    grid_points,
                )
                rmse, rmse_se = rmse_and_se(estimates, exact)
                rows.append(
                    {
                        "probability_search_upper": upper,
                        "upper_over_exact_probability": upper / exact,
                        "target_query_budget": target_budget,
                        "actual_query_budget": actual_budget,
                        "powers": "|".join(map(str, powers)),
                        "maximum_power": max(powers),
                        "shots_per_power": shots,
                        "seed": derived_seed,
                        "repetitions": repeats,
                        "grid_points": grid_points,
                        "mean_estimate": float(estimates.mean()),
                        "bias": float((estimates - exact).mean()),
                        "rmse": rmse,
                        "rmse_standard_error": rmse_se,
                        "relative_rmse": rmse / exact,
                        "gross_alias_fraction": float(
                            np.mean(np.abs(estimates - exact) > 5 * exact)
                        ),
                        "upper_boundary_fraction": float(
                            np.mean(estimates >= 0.999 * upper)
                        ),
                    }
                )
            print(
                f"finished budget={actual_budget}, seed={seed_index + 1}/{seeds}",
                flush=True,
            )

    raw = pd.DataFrame(rows)
    cells = len(PROBABILITY_UPPERS) * len(TARGET_BUDGETS) * seeds
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * cells))
    raw["familywise_95pct_rmse_upper"] = raw["rmse"] + z * raw["rmse_standard_error"]
    raw["familywise_95pct_relative_rmse_upper"] = (
        raw["familywise_95pct_rmse_upper"] / exact
    )
    summary = (
        raw.groupby(["probability_search_upper", "actual_query_budget"], as_index=False)
        .agg(
            upper_over_exact_probability=("upper_over_exact_probability", "first"),
            maximum_power=("maximum_power", "first"),
            shots_per_power=("shots_per_power", "first"),
            maximum_relative_rmse=("relative_rmse", "max"),
            maximum_familywise_95pct_relative_rmse_upper=(
                "familywise_95pct_relative_rmse_upper", "max"
            ),
            maximum_gross_alias_fraction=("gross_alias_fraction", "max"),
            maximum_upper_boundary_fraction=("upper_boundary_fraction", "max"),
        )
        .sort_values(["probability_search_upper", "actual_query_budget"])
    )
    reference = summary.loc[
        np.isclose(summary["probability_search_upper"], 1e-4)
    ][["actual_query_budget", "maximum_relative_rmse"]].rename(
        columns={"maximum_relative_rmse": "reference_1e-4_maximum_relative_rmse"}
    )
    summary = summary.merge(reference, on="actual_query_budget", validate="many_to_one")
    summary["rmse_ratio_vs_1e-4_reference"] = (
        summary["maximum_relative_rmse"]
        / summary["reference_1e-4_maximum_relative_rmse"]
    )

    output_dir = root / "results" / "hyram" / "qae_prior_bound_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / f"prior_sensitivity_raw_{args.mode}.csv", index=False)
    summary.to_csv(output_dir / f"prior_sensitivity_summary_{args.mode}.csv", index=False)
    (output_dir / f"prior_sensitivity_metadata_{args.mode}.json").write_text(
        json.dumps(
            {
                "exact_probability": exact,
                "probability_search_uppers": PROBABILITY_UPPERS,
                "common_random_numbers_across_bounds": True,
                "familywise_cells": cells,
                "familywise_z": z,
                "gross_alias_definition": "absolute estimation error > 5 times exact probability",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nPrior-bound sensitivity")
    print(summary.to_string(index=False))
    print(f"\nresults: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
