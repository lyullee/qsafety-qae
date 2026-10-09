"""Strong classical rare-event baselines for the HyRAM+ trajectory model.

Conditional samplers are treated as unit-cost ideal primitives. Oracle-informed
importance proposals additionally use the true scenario probabilities. These
are optimistic classical ceilings, not directly implementable claims.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


PRECISIONS = [0.50, 0.25, 0.10, 0.05]


def product_variance(means: np.ndarray, per_sample_variances: np.ndarray, counts: np.ndarray) -> float:
    return float(
        np.prod(np.square(means) + per_sample_variances / counts)
        - np.prod(np.square(means))
    )


def optimized_product_allocation(
    means: np.ndarray,
    per_sample_variances: np.ndarray,
    target_rmse: float,
) -> tuple[np.ndarray, float]:
    coefficients = per_sample_variances / np.square(means)
    roots = np.sqrt(coefficients)
    approximate_scale = roots.sum() / (target_rmse / np.prod(means)) ** 2

    def allocation(scale: float) -> np.ndarray:
        return np.maximum(1, np.ceil(scale * roots)).astype(int)

    low, high = 0.0, max(1.0, approximate_scale)
    while product_variance(means, per_sample_variances, allocation(high)) > target_rmse**2:
        high *= 2
    for _ in range(80):
        middle = (low + high) / 2
        if product_variance(means, per_sample_variances, allocation(middle)) <= target_rmse**2:
            high = middle
        else:
            low = middle
    counts = allocation(high)
    return counts, math.sqrt(product_variance(means, per_sample_variances, counts))


def simulate_importance(
    base_weights: np.ndarray,
    probabilities: np.ndarray,
    proposal: np.ndarray,
    samples: int,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    allocations = rng.multinomial(samples, proposal, size=repeats)
    successes = rng.binomial(allocations, probabilities)
    return (successes * (base_weights / proposal)).sum(axis=1) / samples


def error_summary(estimates: np.ndarray, exact: float) -> dict[str, float]:
    errors = estimates - exact
    squared = np.square(errors)
    rmse = float(np.sqrt(np.mean(squared)))
    mse_standard_error = float(squared.std(ddof=1) / math.sqrt(len(squared)))
    return {
        "mean_estimate": float(estimates.mean()),
        "bias": float(errors.mean()),
        "rmse": rmse,
        "rmse_standard_error": mse_standard_error / (2 * rmse) if rmse else 0.0,
        "relative_rmse": rmse / exact,
        "q025": float(np.quantile(estimates, 0.025)),
        "q975": float(np.quantile(estimates, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    repeats = args.repeats or (2_000 if args.mode == "quick" else 20_000)
    root = args.project_root
    inventory_dir = root / "results" / "hyram" / "system_scenario_inventory"
    trajectory_dir = root / "results" / "hyram" / "stochastic_trajectory_oracle"
    qae_boundary_path = (
        root / "results" / "hyram" / "rare_trajectory_qae_scaling"
        / "precision_boundary_full.csv"
    )
    frame = pd.read_csv(inventory_dir / "component_leak_scenarios.csv")
    trajectory = json.loads(
        (trajectory_dir / "trajectory_summary.json").read_text(encoding="utf-8")
    )
    qae_boundary = pd.read_csv(qae_boundary_path).set_index("target_relative_rmse")

    weights = frame["component_count"].to_numpy(float)
    weights /= weights.sum()
    leak = -np.expm1(-frame["median_leak_frequency_per_component_year"].to_numpy(float))
    ignition = (
        frame["immediate_ignition_probability_given_release"].to_numpy(float)
        + frame["delayed_ignition_probability_given_release"].to_numpy(float)
    )
    isolation_failure = 0.1
    exact = float(np.dot(weights, leak * isolation_failure * ignition))
    if not np.isclose(exact, trajectory["random_opportunity_hazard_probability"]):
        raise AssertionError("Factorized trajectory probability does not match circuit result")

    leak_probability = float(np.dot(weights, leak))
    conditional_weights = weights * leak / leak_probability
    ignition_given_leak = float(np.dot(conditional_weights, ignition))
    means = np.asarray([leak_probability, isolation_failure, ignition_given_leak])
    if not np.isclose(np.prod(means), exact):
        raise AssertionError("Conditional probability product does not match exact result")

    # True-probability-informed proposals minimize the Bernoulli estimator's
    # second moment at the leak and conditional-ignition stages.
    leak_proposal = weights * np.sqrt(np.clip(leak, 1e-15, 1.0))
    leak_proposal /= leak_proposal.sum()
    leak_is_variance = float(
        np.sum(weights**2 * leak / leak_proposal) - leak_probability**2
    )
    ignition_proposal = conditional_weights * np.sqrt(np.clip(ignition, 1e-15, 1.0))
    ignition_proposal /= ignition_proposal.sum()
    ignition_is_variance = float(
        np.sum(conditional_weights**2 * ignition / ignition_proposal)
        - ignition_given_leak**2
    )

    direct_variances = means * (1 - means)
    importance_variances = np.asarray(
        [leak_is_variance, isolation_failure * (1 - isolation_failure), ignition_is_variance]
    )
    rng = np.random.default_rng(args.seed)
    budget_rows: list[dict] = []
    simulation_rows: list[dict] = []

    for precision in PRECISIONS:
        target = precision * exact
        qae_queries = int(qae_boundary.loc[precision, "minimum_observed_qae_queries"])
        crude_budget = math.ceil(exact * (1 - exact) / target**2)
        direct_counts, direct_rmse = optimized_product_allocation(
            means, direct_variances, target
        )
        importance_counts, importance_rmse = optimized_product_allocation(
            means, importance_variances, target
        )
        methods = {
            "crude_trajectory_mc": {
                "budget": crude_budget,
                "theoretical_rmse": math.sqrt(exact * (1 - exact) / crude_budget),
                "information": "unconditional trajectory samples only",
                "allocations": str(crude_budget),
            },
            "ideal_conditional_splitting": {
                "budget": int(direct_counts.sum()),
                "theoretical_rmse": direct_rmse,
                "information": "unit-cost exact conditional samplers; true stage structure",
                "allocations": "|".join(map(str, direct_counts)),
            },
            "oracle_importance_conditional_splitting": {
                "budget": int(importance_counts.sum()),
                "theoretical_rmse": importance_rmse,
                "information": "unit-cost conditional samplers and true scenario probabilities",
                "allocations": "|".join(map(str, importance_counts)),
            },
        }
        for method, values in methods.items():
            budget_rows.append(
                {
                    "target_relative_rmse": precision,
                    "target_absolute_rmse": target,
                    "method": method,
                    "information_level": values["information"],
                    "stage_allocation_leak_isolation_ignition": values["allocations"],
                    "minimum_classical_queries": values["budget"],
                    "theoretical_rmse": values["theoretical_rmse"],
                    "ideal_mlqae_queries": qae_queries,
                    "classical_to_qae_query_ratio": values["budget"] / qae_queries,
                }
            )

        # Simulations validate the formulae at their own first-passing budgets.
        crude = rng.binomial(crude_budget, exact, repeats) / crude_budget
        leak_direct = rng.binomial(direct_counts[0], means[0], repeats) / direct_counts[0]
        isolation_direct = rng.binomial(direct_counts[1], means[1], repeats) / direct_counts[1]
        ignition_direct = rng.binomial(direct_counts[2], means[2], repeats) / direct_counts[2]
        direct_product = leak_direct * isolation_direct * ignition_direct

        leak_importance = simulate_importance(
            weights, leak, leak_proposal, int(importance_counts[0]), repeats, rng
        )
        isolation_importance = (
            rng.binomial(importance_counts[1], means[1], repeats) / importance_counts[1]
        )
        ignition_importance = simulate_importance(
            conditional_weights,
            ignition,
            ignition_proposal,
            int(importance_counts[2]),
            repeats,
            rng,
        )
        importance_product = leak_importance * isolation_importance * ignition_importance
        for method, estimates, counts in (
            ("crude_trajectory_mc", crude, np.asarray([crude_budget])),
            ("ideal_conditional_splitting", direct_product, direct_counts),
            (
                "oracle_importance_conditional_splitting",
                importance_product,
                importance_counts,
            ),
        ):
            simulation_rows.append(
                {
                    "target_relative_rmse": precision,
                    "method": method,
                    "query_budget": int(counts.sum()),
                    "stage_allocation_leak_isolation_ignition": "|".join(map(str, counts)),
                    "repetitions": repeats,
                    **error_summary(estimates, exact),
                }
            )
        print(f"finished precision={precision:g}", flush=True)

    budgets = pd.DataFrame(budget_rows)
    simulations = pd.DataFrame(simulation_rows)
    stage_parameters = pd.DataFrame(
        {
            "stage": ["leak", "isolation_failure", "ignition_given_leak"],
            "conditional_probability": means,
            "direct_bernoulli_variance_per_sample": direct_variances,
            "oracle_importance_variance_per_sample": importance_variances,
        }
    )
    output_dir = root / "results" / "hyram" / "strong_rare_event_baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets.to_csv(output_dir / f"minimum_query_budgets_{args.mode}.csv", index=False)
    simulations.to_csv(output_dir / f"simulation_validation_{args.mode}.csv", index=False)
    stage_parameters.to_csv(output_dir / "conditional_stage_parameters.csv", index=False)

    print("\nConditional stages")
    print(stage_parameters.to_string(index=False))
    print("\nMinimum query budgets")
    print(
        budgets[[
            "target_relative_rmse", "method",
            "stage_allocation_leak_isolation_ignition", "minimum_classical_queries",
            "ideal_mlqae_queries", "classical_to_qae_query_ratio",
        ]].to_string(index=False)
    )
    print("\nSimulation validation")
    print(
        simulations[[
            "target_relative_rmse", "method", "query_budget", "relative_rmse", "bias"
        ]].to_string(index=False)
    )
    print(f"\nresults: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
