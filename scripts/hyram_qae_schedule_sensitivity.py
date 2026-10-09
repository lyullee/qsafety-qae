"""Held-out HyRAM+ sensitivity analysis across ML-QAE power schedules.

Each schedule ratio receives an independent training scan and is then evaluated
with independent validation seeds.  The study tests robustness; it does not pick
the best ratio after seeing validation results.
"""

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


BASE_HYRAM_PROBABILITY = 4.600343517804214e-7
FULL_PROBABILITIES = [
    1e-8,
    BASE_HYRAM_PROBABILITY,
    5e-7,
    1e-6,
    2e-6,
    3e-6,
]
FULL_RATIOS = [1.3, 1.5, 1.8, 2.0]
TARGET_RELATIVE_RMSE = 0.02
CANDIDATE_BUDGETS = [
    500_000,
    650_000,
    800_000,
    1_000_000,
    1_200_000,
    1_500_000,
    1_800_000,
    2_100_000,
    2_500_000,
    3_000_000,
    3_500_000,
    4_200_000,
    5_000_000,
    6_000_000,
    7_000_000,
    8_000_000,
    10_000_000,
    12_500_000,
    15_000_000,
    18_000_000,
    21_000_000,
    25_000_000,
]


def simulate_rmse(
    probability: float,
    ratio: float,
    target_budget: int,
    repeats: int,
    grid_points: int,
    probability_upper: float,
    rng: np.random.Generator,
) -> tuple[float, float, list[int], int, int]:
    powers, shots, actual_budget = schedule_for_ratio(ratio, target_budget)
    theta = math.asin(math.sqrt(probability))
    amplified = [math.sin((2 * power + 1) * theta) ** 2 for power in powers]
    counts = np.column_stack(
        [rng.binomial(shots, value, repeats) for value in amplified]
    )
    estimates = bounded_batched_mle(
        counts, shots, powers, probability_upper, grid_points, batch_size=100
    )
    squared = np.square(estimates - probability)
    rmse = math.sqrt(float(squared.mean()))
    mse_se = float(squared.std(ddof=1) / math.sqrt(repeats))
    rmse_se = mse_se / (2 * rmse) if rmse else 0.0
    return rmse, rmse_se, powers, shots, actual_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--probabilities", type=float, nargs="+")
    parser.add_argument("--ratios", type=float, nargs="+")
    parser.add_argument("--training-repeats", type=int)
    parser.add_argument("--validation-repeats-per-seed", type=int)
    parser.add_argument("--validation-seeds", type=int)
    parser.add_argument("--grid-points", type=int)
    parser.add_argument("--output-tag")
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()

    probabilities = args.probabilities or (
        [BASE_HYRAM_PROBABILITY, 1e-6]
        if args.mode == "quick"
        else FULL_PROBABILITIES
    )
    ratios = args.ratios or ([1.3, 1.5] if args.mode == "quick" else FULL_RATIOS)
    training_repeats = args.training_repeats or (150 if args.mode == "quick" else 1_000)
    validation_repeats = args.validation_repeats_per_seed or (
        250 if args.mode == "quick" else 3_000
    )
    validation_seeds = args.validation_seeds or (2 if args.mode == "quick" else 5)
    grid_points = args.grid_points or (32_769 if args.mode == "quick" else 65_537)
    output_tag = args.output_tag or args.mode

    if any(not 0 < p < 1 for p in probabilities):
        raise ValueError("Every probability must lie between zero and one")
    if any(r <= 1 for r in ratios):
        raise ValueError("Every schedule ratio must exceed one")

    training_rows: list[dict] = []
    selected_rows: list[dict] = []
    for probability_index, probability in enumerate(probabilities):
        probability_upper = 100 * probability
        for ratio_index, ratio in enumerate(ratios):
            training_rng = np.random.default_rng(
                args.seed + 10_000 * probability_index + 100 * ratio_index
            )
            candidates: list[dict] = []
            chosen = None
            for target_budget in CANDIDATE_BUDGETS:
                rmse, rmse_se, powers, shots, actual = simulate_rmse(
                    probability,
                    ratio,
                    target_budget,
                    training_repeats,
                    grid_points,
                    probability_upper,
                    training_rng,
                )
                row = {
                    "probability": probability,
                    "schedule_ratio": ratio,
                    "target_relative_rmse": TARGET_RELATIVE_RMSE,
                    "probability_search_upper": probability_upper,
                    "target_query_budget": target_budget,
                    "actual_query_budget": actual,
                    "powers": "|".join(map(str, powers)),
                    "maximum_power": max(powers),
                    "shots_per_power": shots,
                    "training_rmse": rmse,
                    "training_rmse_standard_error": rmse_se,
                    "training_relative_rmse": rmse / probability,
                }
                training_rows.append(row)
                candidates.append(row)
                if rmse / probability <= 0.9 * TARGET_RELATIVE_RMSE:
                    chosen = row
                    break
            if chosen is None:
                chosen = min(candidates, key=lambda item: item["training_relative_rmse"])
                status = "no_margin_pass_selected_lowest_training_rmse"
            else:
                status = "first_training_budget_below_0.9x_target"
            selected_rows.append({**chosen, "selection_status": status})
            print(
                f"training p={probability:.6g}, ratio={ratio:g}, "
                f"selected={chosen['actual_query_budget']}",
                flush=True,
            )

    selected = pd.DataFrame(selected_rows)
    validation_rows: list[dict] = []
    for cell_index, item in selected.iterrows():
        for seed_index in range(validation_seeds):
            derived_seed = args.seed + 1_000_000 + 1000 * cell_index + seed_index
            rng = np.random.default_rng(derived_seed)
            rmse, rmse_se, powers, shots, actual = simulate_rmse(
                float(item["probability"]),
                float(item["schedule_ratio"]),
                int(item["target_query_budget"]),
                validation_repeats,
                grid_points,
                float(item["probability_search_upper"]),
                rng,
            )
            validation_rows.append(
                {
                    "probability": item["probability"],
                    "schedule_ratio": item["schedule_ratio"],
                    "target_relative_rmse": TARGET_RELATIVE_RMSE,
                    "seed": derived_seed,
                    "actual_query_budget": actual,
                    "powers": "|".join(map(str, powers)),
                    "maximum_power": max(powers),
                    "shots_per_power": shots,
                    "rmse": rmse,
                    "rmse_standard_error": rmse_se,
                    "relative_rmse": rmse / float(item["probability"]),
                }
            )
        print(
            f"validation p={item['probability']:.6g}, "
            f"ratio={item['schedule_ratio']:g}",
            flush=True,
        )

    validation = pd.DataFrame(validation_rows)
    familywise_cells = len(selected) * validation_seeds
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * familywise_cells))
    validation["global_familywise_relative_rmse_upper"] = (
        validation["rmse"] + z * validation["rmse_standard_error"]
    ) / validation["probability"]
    summary = (
        validation.groupby(["probability", "schedule_ratio"], as_index=False)
        .agg(
            qae_queries=("actual_query_budget", "first"),
            powers=("powers", "first"),
            maximum_power=("maximum_power", "first"),
            shots_per_power=("shots_per_power", "first"),
            maximum_relative_rmse=("relative_rmse", "max"),
            maximum_global_familywise_relative_rmse_upper=(
                "global_familywise_relative_rmse_upper",
                "max",
            ),
        )
        .sort_values(["probability", "schedule_ratio"])
    )
    summary["passes_global_familywise_target"] = (
        summary["maximum_global_familywise_relative_rmse_upper"]
        <= TARGET_RELATIVE_RMSE
    )

    # Persist expensive simulation outputs before lightweight table joins.
    # A later reporting error must not force the stochastic experiment to rerun.
    output_dir = (
        args.project_root / "results" / "hyram" / "qae_schedule_sensitivity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(training_rows).to_csv(
        output_dir / f"training_scan_{output_tag}.csv", index=False
    )
    selected.to_csv(output_dir / f"selected_budgets_{output_tag}.csv", index=False)
    validation.to_csv(output_dir / f"validation_raw_{output_tag}.csv", index=False)
    summary.to_csv(
        output_dir / f"validation_summary_checkpoint_{output_tag}.csv", index=False
    )

    final_comparison = pd.read_csv(
        args.project_root
        / "results"
        / "hyram"
        / "rarity_precision_validation"
        / "algorithm_comparison_dense_2pct_final.csv"
    )
    importance_queries = []
    for probability in summary["probability"]:
        matched = final_comparison.loc[
            np.isclose(
                final_comparison["probability"],
                probability,
                rtol=1e-12,
                atol=0.0,
            )
        ]
        if len(matched) != 1:
            raise RuntimeError(f"No unique classical comparison for p={probability:g}")
        importance_queries.append(int(matched.iloc[0]["oracle_importance_splitting_queries"]))
    summary["oracle_importance_splitting_queries"] = importance_queries
    summary["importance_to_qae_ratio"] = (
        summary["oracle_importance_splitting_queries"] / summary["qae_queries"]
    )
    summary["lowest_query_method"] = np.where(
        summary["importance_to_qae_ratio"] > 1,
        "validated_mlqae",
        "oracle_importance_splitting",
    )
    summary.loc[
        ~summary["passes_global_familywise_target"], "lowest_query_method"
    ] = "qae_validation_failed"

    ratio_15 = summary.loc[np.isclose(summary["schedule_ratio"], 1.5), [
        "probability", "qae_queries", "lowest_query_method"
    ]].rename(
        columns={
            "qae_queries": "ratio_1p5_qae_queries",
            "lowest_query_method": "ratio_1p5_lowest_query_method",
        }
    )
    summary = summary.merge(ratio_15, on="probability", how="left")
    summary["qae_queries_relative_to_ratio_1p5"] = (
        summary["qae_queries"] / summary["ratio_1p5_qae_queries"]
    )

    robustness_rows: list[dict] = []
    for probability, group in summary.groupby("probability"):
        all_pass = bool(group["passes_global_familywise_target"].all())
        winner_stable = bool(
            group["lowest_query_method"].nunique() == 1
            and not group["lowest_query_method"].eq("qae_validation_failed").any()
        )
        robustness_rows.append(
            {
                "probability": probability,
                "tested_schedule_ratios": "|".join(f"{x:g}" for x in group["schedule_ratio"]),
                "ratios_passing_global_target": int(
                    group["passes_global_familywise_target"].sum()
                ),
                "ratios_tested": len(group),
                "minimum_qae_queries": int(group["qae_queries"].min()),
                "maximum_qae_queries": int(group["qae_queries"].max()),
                "qae_query_spread_ratio": float(
                    group["qae_queries"].max() / group["qae_queries"].min()
                ),
                "all_ratios_pass": all_pass,
                "query_winner_stable_across_ratios": winner_stable,
                "ratio_1p5_winner": group["ratio_1p5_lowest_query_method"].iloc[0],
            }
        )
    robustness = pd.DataFrame(robustness_rows)

    summary.to_csv(output_dir / f"schedule_comparison_{output_tag}.csv", index=False)
    robustness.to_csv(
        output_dir / f"probability_robustness_{output_tag}.csv", index=False
    )
    metadata = {
        "analysis_role": "confirmatory schedule sensitivity, not post-validation ratio selection",
        "probabilities": probabilities,
        "schedule_ratios": ratios,
        "target_relative_rmse": TARGET_RELATIVE_RMSE,
        "training_margin_fraction": 0.9,
        "training_repeats": training_repeats,
        "validation_repeats_per_seed": validation_repeats,
        "validation_seeds": validation_seeds,
        "familywise_validation_cells": familywise_cells,
        "familywise_z": z,
        "interpretation_limit": (
            "Each ratio is independently budget-tuned using training replicates and assessed "
            "with held-out seeds. Results measure finite-schedule sensitivity under the same "
            "ideal probability-oracle model, not end-to-end QPU runtime."
        ),
    }
    (output_dir / f"metadata_{output_tag}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nHeld-out schedule comparison")
    print(summary.to_string(index=False))
    print("\nProbability-level robustness")
    print(robustness.to_string(index=False))
    print(f"\nresults: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
