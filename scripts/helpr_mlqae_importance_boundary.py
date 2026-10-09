"""Train and independently validate an ideal ML-QAE/importance query boundary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from hyram_rare_trajectory_qae_scaling import bounded_batched_mle, summarize
from research_schedule import schedule_for_ratio


PROBABILITIES = [
    0.006915260456630194,
    0.007970272865693958,
    0.009025285274757721,
]
PRECISIONS = [0.10, 0.05, 0.02, 0.01, 0.005]
RATIOS = [1.3, 1.5, 1.8, 2.0]
TARGET_BUDGETS = [
    512,
    1024,
    2048,
    4096,
    8192,
    16_384,
    32_768,
    65_536,
    131_072,
]


def simulate_cell(
    probability: float,
    ratio: float,
    target_budget: int,
    repeats: int,
    grid_points: int,
    probability_upper: float,
    rng: np.random.Generator,
) -> tuple[dict[str, object], np.ndarray]:
    powers, shots, actual_budget = schedule_for_ratio(ratio, target_budget)
    theta = math.asin(math.sqrt(probability))
    amplified = np.asarray(
        [math.sin((2 * power + 1) * theta) ** 2 for power in powers]
    )
    counts = np.column_stack(
        [rng.binomial(shots, value, repeats) for value in amplified]
    )
    estimates = bounded_batched_mle(
        counts,
        shots,
        powers,
        probability_upper,
        grid_points,
    )
    return (
        {
            "probability": probability,
            "schedule_ratio": ratio,
            "target_query_budget": target_budget,
            "actual_query_budget": actual_budget,
            "powers": "|".join(map(str, powers)),
            "maximum_power": max(powers),
            "shots_per_power": shots,
            "repeats": repeats,
            **summarize(estimates, probability),
        },
        estimates,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--training-repeats", type=int)
    parser.add_argument("--validation-repeats", type=int)
    parser.add_argument("--grid-points", type=int)
    parser.add_argument("--probability-upper", type=float, default=0.02)
    parser.add_argument("--training-seed", type=int, default=34001)
    parser.add_argument("--validation-seed", type=int, default=35001)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    training_repeats = args.training_repeats or (200 if args.mode == "quick" else 1000)
    validation_repeats = args.validation_repeats or (
        1000 if args.mode == "quick" else 5000
    )
    grid_points = args.grid_points or (16_385 if args.mode == "quick" else 65_537)
    if max(PROBABILITIES) >= args.probability_upper:
        raise ValueError("Probability interval must lie inside the MLE search interval")

    training_rng = np.random.default_rng(args.training_seed)
    training_rows = []
    for probability in PROBABILITIES:
        for ratio in RATIOS:
            for budget in TARGET_BUDGETS:
                row, _ = simulate_cell(
                    probability,
                    ratio,
                    budget,
                    training_repeats,
                    grid_points,
                    args.probability_upper,
                    training_rng,
                )
                training_rows.append(row)
                print(
                    f"training p={probability:.8g}, ratio={ratio:g}, "
                    f"budget={row['actual_query_budget']}",
                    flush=True,
                )
    training = pd.DataFrame(training_rows)

    selection_rows = []
    for ratio in RATIOS:
        robust_budgets = []
        complete = True
        for precision in PRECISIONS:
            point_budgets = []
            for probability in PROBABILITIES:
                eligible = training.loc[
                    training["schedule_ratio"].eq(ratio)
                    & np.isclose(training["probability"], probability)
                    & training["relative_rmse"].le(0.85 * precision)
                ].sort_values("actual_query_budget")
                if eligible.empty:
                    complete = False
                    point_budgets.append(np.nan)
                else:
                    point_budgets.append(int(eligible.iloc[0]["actual_query_budget"]))
            robust_bud = (
                int(np.nanmax(point_budgets))
                if point_budgets and np.isfinite(point_budgets).all()
                else np.nan
            )
            robust_budgets.append(robust_bud)
            selection_rows.append(
                {
                    "schedule_ratio": ratio,
                    "target_relative_rmse": precision,
                    "robust_training_query_budget": robust_bud,
                    "point_query_budgets": "|".join(
                        "NA" if not np.isfinite(value) else str(int(value))
                        for value in point_budgets
                    ),
                }
            )
        ratio_score = float(np.nansum(robust_budgets)) if complete else math.inf
        for row in selection_rows:
            if row["schedule_ratio"] == ratio:
                row["schedule_complete"] = complete
                row["ratio_score"] = ratio_score
    selection = pd.DataFrame(selection_rows)
    viable = selection.loc[selection["schedule_complete"]].copy()
    if viable.empty:
        raise RuntimeError("No schedule passed the training-margin rule")
    selected_ratio = float(
        viable.groupby("schedule_ratio")["ratio_score"].first().idxmin()
    )
    selected_budgets = {
        float(row.target_relative_rmse): int(row.robust_training_query_budget)
        for row in selection.loc[selection["schedule_ratio"].eq(selected_ratio)].itertuples()
    }

    validation_rng = np.random.default_rng(args.validation_seed)
    validation_rows = []
    validation_estimates = []
    for probability in PROBABILITIES:
        for precision in PRECISIONS:
            target_budget = selected_budgets[precision]
            row, estimates = simulate_cell(
                probability,
                selected_ratio,
                target_budget,
                validation_repeats,
                grid_points,
                args.probability_upper,
                validation_rng,
            )
            errors_squared = np.square(estimates - probability)
            rmse = float(row["rmse"])
            mse_se = float(errors_squared.std(ddof=1) / math.sqrt(validation_repeats))
            rmse_se = mse_se / (2 * rmse) if rmse else 0.0
            validation_rows.append(
                {
                    **row,
                    "target_relative_rmse": precision,
                    "target_absolute_rmse": probability * precision,
                    "rmse_standard_error": rmse_se,
                }
            )
            validation_estimates.append(estimates)
            print(
                f"validation p={probability:.8g}, precision={precision:g}, "
                f"budget={row['actual_query_budget']}",
                flush=True,
            )
    validation = pd.DataFrame(validation_rows)
    family_cells = len(validation)
    critical_z = NormalDist().inv_cdf(1 - 0.05 / (2 * family_cells))
    validation["familywise_95pct_rmse_upper"] = (
        validation["rmse"] + critical_z * validation["rmse_standard_error"]
    )
    validation["familywise_relative_rmse_upper"] = (
        validation["familywise_95pct_rmse_upper"] / validation["probability"]
    )
    validation["passes_familywise_target"] = validation[
        "familywise_relative_rmse_upper"
    ].le(validation["target_relative_rmse"])

    importance_path = (
        args.project_root
        / "results"
        / "helpr"
        / "importance_sampling"
        / "importance_query_budgets.csv"
    )
    importance = pd.read_csv(importance_path).set_index("target_relative_rmse")
    comparison_rows = []
    center_probability = PROBABILITIES[1]
    for precision in PRECISIONS:
        center = validation.loc[
            np.isclose(validation["probability"], center_probability)
            & np.isclose(validation["target_relative_rmse"], precision)
        ].iloc[0]
        qae_queries = int(center["actual_query_budget"])
        importance_queries = int(importance.loc[precision, "defensive_importance_queries"])
        crude_queries = int(importance.loc[precision, "crude_mc_queries"])
        comparison_rows.append(
            {
                "target_relative_rmse": precision,
                "validated_qae_queries": qae_queries,
                "strong_importance_queries": importance_queries,
                "crude_mc_queries": crude_queries,
                "importance_to_qae_query_ratio": importance_queries / qae_queries,
                "crude_to_qae_query_ratio": crude_queries / qae_queries,
                "maximum_oracle_cost_multiplier_for_qae_parity": (
                    importance_queries / qae_queries
                ),
                "qae_passes_all_probability_points": bool(
                    validation.loc[
                        np.isclose(validation["target_relative_rmse"], precision),
                        "passes_familywise_target",
                    ].all()
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    output_dir = args.project_root / "results" / "helpr" / "mlqae_boundary"
    output_dir.mkdir(parents=True, exist_ok=True)
    training.to_csv(output_dir / f"training_{args.mode}.csv", index=False)
    selection.to_csv(output_dir / f"schedule_selection_{args.mode}.csv", index=False)
    validation.to_csv(output_dir / f"heldout_validation_{args.mode}.csv", index=False)
    comparison.to_csv(output_dir / f"qae_vs_classical_{args.mode}.csv", index=False)
    metadata = {
        "mode": args.mode,
        "selected_schedule_ratio": selected_ratio,
        "selected_training_budgets": selected_budgets,
        "training_repeats": training_repeats,
        "validation_repeats": validation_repeats,
        "grid_points": grid_points,
        "familywise_cells": family_cells,
        "familywise_critical_z": critical_z,
        "all_validation_cells_pass": bool(validation["passes_familywise_target"].all()),
        "claim_limit": (
            "Query ratios assume ideal coherent access. State preparation, reversible "
            "HELPR evaluation, and fault-tolerant overhead are not included."
        ),
    }
    (output_dir / f"metadata_{args.mode}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print("\nSchedule selection")
    print(selection.to_string(index=False))
    print("\nHeld-out validation")
    print(
        validation[[
            "probability",
            "target_relative_rmse",
            "actual_query_budget",
            "relative_rmse",
            "familywise_relative_rmse_upper",
            "passes_familywise_target",
        ]].to_string(index=False)
    )
    print("\nQAE versus classical")
    print(comparison.to_string(index=False))
    print("\nMetadata")
    print(json.dumps(metadata, indent=2))
    print(f"\nresults: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
