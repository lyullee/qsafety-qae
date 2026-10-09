"""Joint precision and conditional-sampling-cost map for HyRAM+ QAE screening."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from hyram_rare_trajectory_qae_scaling import bounded_batched_mle
from hyram_strong_rare_event_baselines import optimized_product_allocation
from research_schedule import schedule_for_ratio


PRECISIONS = [0.10, 0.09, 0.08, 0.07, 0.06, 0.05]
COST_MULTIPLIERS = [1, 2, 5, 10, 20, 50, 100]
CANDIDATE_BUDGETS = [
    350_000, 400_000, 450_000, 500_000, 524_288, 550_000, 600_000,
    650_000, 700_000, 750_000, 800_000, 850_000, 900_000, 950_000,
    1_000_000, 1_048_576, 1_100_000, 1_150_000, 1_200_000,
]


def simulate_rmse(
    probability: float,
    target_budget: int,
    repeats: int,
    grid_points: int,
    probability_upper: float,
    rng: np.random.Generator,
) -> tuple[float, float, list[int], int, int]:
    powers, shots, actual_budget = schedule_for_ratio(1.5, target_budget)
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
    parser.add_argument("--training-repeats", type=int)
    parser.add_argument("--validation-repeats-per-seed", type=int)
    parser.add_argument("--validation-seeds", type=int)
    parser.add_argument("--grid-points", type=int)
    parser.add_argument("--probability-upper", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    training_repeats = args.training_repeats or (300 if args.mode == "quick" else 2_000)
    validation_repeats = args.validation_repeats_per_seed or (
        500 if args.mode == "quick" else 3_000
    )
    validation_seeds = args.validation_seeds or (2 if args.mode == "quick" else 5)
    grid_points = args.grid_points or (32_769 if args.mode == "quick" else 65_537)
    root = args.project_root

    inventory_dir = root / "results" / "hyram" / "system_scenario_inventory"
    trajectory = json.loads(
        (
            root / "results" / "hyram" / "stochastic_trajectory_oracle"
            / "trajectory_summary.json"
        ).read_text(encoding="utf-8")
    )
    frame = pd.read_csv(inventory_dir / "component_leak_scenarios.csv")
    exact = float(trajectory["random_opportunity_hazard_probability"])
    weights = frame["component_count"].to_numpy(float)
    weights /= weights.sum()
    leak = -np.expm1(-frame["median_leak_frequency_per_component_year"].to_numpy(float))
    ignition = (
        frame["immediate_ignition_probability_given_release"].to_numpy(float)
        + frame["delayed_ignition_probability_given_release"].to_numpy(float)
    )
    leak_probability = float(np.dot(weights, leak))
    conditional_weights = weights * leak / leak_probability
    means = np.asarray(
        [leak_probability, 0.1, float(np.dot(conditional_weights, ignition))]
    )
    variances = means * (1 - means)
    if not np.isclose(np.prod(means), exact):
        raise AssertionError("Conditional-stage product does not match trajectory amplitude")

    training_rng = np.random.default_rng(args.seed)
    training_rows: list[dict] = []
    for target_budget in CANDIDATE_BUDGETS:
        rmse, rmse_se, powers, shots, actual_budget = simulate_rmse(
            exact,
            target_budget,
            training_repeats,
            grid_points,
            args.probability_upper,
            training_rng,
        )
        training_rows.append(
            {
                "target_query_budget": target_budget,
                "actual_query_budget": actual_budget,
                "powers": "|".join(map(str, powers)),
                "maximum_power": max(powers),
                "shots_per_power": shots,
                "rmse": rmse,
                "rmse_standard_error": rmse_se,
                "relative_rmse": rmse / exact,
            }
        )
        print(f"training target_budget={target_budget}, actual={actual_budget}", flush=True)
    training = pd.DataFrame(training_rows).sort_values("actual_query_budget")

    selected_rows: list[dict] = []
    for precision in PRECISIONS:
        eligible = training.loc[training["relative_rmse"].le(0.9 * precision)]
        if len(eligible):
            chosen = eligible.iloc[0]
            status = "first_training_budget_below_0.9x_target"
        else:
            chosen = training.loc[training["relative_rmse"].idxmin()]
            status = "no_margin_pass_selected_lowest_training_rmse"
        selected_rows.append(
            {
                "target_relative_rmse": precision,
                "selected_target_query_budget": int(chosen["target_query_budget"]),
                "selected_actual_query_budget": int(chosen["actual_query_budget"]),
                "training_relative_rmse": float(chosen["relative_rmse"]),
                "selection_status": status,
            }
        )
    selected = pd.DataFrame(selected_rows)

    validation_rows: list[dict] = []
    for precision_index, item in selected.iterrows():
        for seed_index in range(validation_seeds):
            derived_seed = args.seed + 100_000 + 10_000 * precision_index + seed_index
            rng = np.random.default_rng(derived_seed)
            rmse, rmse_se, powers, shots, actual_budget = simulate_rmse(
                exact,
                int(item["selected_target_query_budget"]),
                validation_repeats,
                grid_points,
                args.probability_upper,
                rng,
            )
            validation_rows.append(
                {
                    "target_relative_rmse": item["target_relative_rmse"],
                    "seed": derived_seed,
                    "actual_query_budget": actual_budget,
                    "powers": "|".join(map(str, powers)),
                    "maximum_power": max(powers),
                    "shots_per_power": shots,
                    "rmse": rmse,
                    "rmse_standard_error": rmse_se,
                    "relative_rmse": rmse / exact,
                }
            )
            print(
                f"validation precision={item['target_relative_rmse']:.2f}, "
                f"seed={seed_index + 1}/{validation_seeds}",
                flush=True,
            )
    validation = pd.DataFrame(validation_rows)
    cells = len(PRECISIONS) * validation_seeds
    z = NormalDist().inv_cdf(1 - 0.05 / (2 * cells))
    validation["familywise_relative_rmse_upper"] = (
        validation["rmse"] + z * validation["rmse_standard_error"]
    ) / exact
    validation_summary = (
        validation.groupby("target_relative_rmse", as_index=False)
        .agg(
            qae_queries=("actual_query_budget", "first"),
            powers=("powers", "first"),
            maximum_power=("maximum_power", "first"),
            maximum_relative_rmse=("relative_rmse", "max"),
            maximum_familywise_relative_rmse_upper=(
                "familywise_relative_rmse_upper", "max"
            ),
        )
        .sort_values("target_relative_rmse", ascending=False)
    )
    validation_summary["passes_familywise_target"] = (
        validation_summary["maximum_familywise_relative_rmse_upper"]
        <= validation_summary["target_relative_rmse"]
    )

    map_rows: list[dict] = []
    for _, item in validation_summary.iterrows():
        precision = float(item["target_relative_rmse"])
        target = precision * exact
        allocation, theoretical_rmse = optimized_product_allocation(
            means, variances, target
        )
        unit_cost_budget = int(allocation.sum())
        qae_queries = int(item["qae_queries"])
        critical_multiplier = qae_queries / unit_cost_budget
        for multiplier in COST_MULTIPLIERS:
            effective_classical_cost = multiplier * unit_cost_budget
            map_rows.append(
                {
                    "target_relative_rmse": precision,
                    "passes_qae_familywise_target": item["passes_familywise_target"],
                    "qae_queries": qae_queries,
                    "maximum_qae_grover_power": int(item["maximum_power"]),
                    "conditional_stage_allocation": "|".join(map(str, allocation)),
                    "unit_cost_conditional_queries": unit_cost_budget,
                    "conditional_sample_cost_multiplier": multiplier,
                    "effective_classical_primitive_cost": effective_classical_cost,
                    "critical_cost_multiplier_for_parity": critical_multiplier,
                    "qae_lower_query_cost": bool(
                        item["passes_familywise_target"]
                        and qae_queries < effective_classical_cost
                    ),
                    "classical_cost_over_qae": effective_classical_cost / qae_queries,
                    "conditional_theoretical_rmse": theoretical_rmse,
                }
            )
    cost_map = pd.DataFrame(map_rows)

    output_dir = root / "results" / "hyram" / "precision_conditional_cost_map"
    output_dir.mkdir(parents=True, exist_ok=True)
    training.to_csv(output_dir / f"qae_training_scan_{args.mode}.csv", index=False)
    selected.to_csv(output_dir / f"qae_selected_budgets_{args.mode}.csv", index=False)
    validation.to_csv(output_dir / f"qae_validation_raw_{args.mode}.csv", index=False)
    validation_summary.to_csv(
        output_dir / f"qae_validation_summary_{args.mode}.csv", index=False
    )
    cost_map.to_csv(output_dir / f"precision_cost_map_{args.mode}.csv", index=False)
    (output_dir / f"metadata_{args.mode}.json").write_text(
        json.dumps(
            {
                "probability_search_upper": args.probability_upper,
                "training_margin_fraction": 0.9,
                "familywise_cells": cells,
                "familywise_z": z,
                "conditional_cost_interpretation": (
                    "Multiplier converts one ideal conditional draw into equivalent primitive "
                    "trajectory-oracle calls; it is a sensitivity parameter, not a measured runtime."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nQAE held-out validation")
    print(validation_summary.to_string(index=False))
    print("\nParity thresholds")
    print(
        cost_map.drop_duplicates("target_relative_rmse")[[
            "target_relative_rmse", "qae_queries", "unit_cost_conditional_queries",
            "critical_cost_multiplier_for_parity",
        ]].to_string(index=False)
    )
    print(f"\nresults: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
