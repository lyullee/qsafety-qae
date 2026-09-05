"""Classical reference calculation for the PHMSA QAE experiment.

Quick mode is safe to run interactively. Full mode increases repetitions and
sample budgets and is intended to be run by the user before final figures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
        / denominator
    )
    return center - half_width, center + half_width


def poisson_rate_interval(count: int, exposure: float, z: float = 1.959963984540054) -> tuple[float, float]:
    """Large-sample interval used only as a lightweight diagnostic."""
    if count <= 0 or exposure <= 0:
        return 0.0, math.nan
    rate = count / exposure
    half_width = z * math.sqrt(count) / exposure
    return max(0.0, rate - half_width), rate + half_width


def estimate_pooled_risk(exposure: pd.DataFrame) -> dict:
    valid = exposure[exposure["total_miles"].gt(0)].copy()
    mile_years = float(valid["total_miles"].sum())
    incidents = int(valid["incident_count"].sum())
    serious = int(valid["serious_incident_count"].sum())

    incident_rate = incidents / mile_years
    serious_rate = serious / mile_years
    serious_given_incident = serious / incidents
    q_low, q_high = wilson_interval(serious, incidents)
    i_low, i_high = poisson_rate_interval(incidents, mile_years)
    s_low, s_high = poisson_rate_interval(serious, mile_years)

    expected = incident_rate * valid["total_miles"]
    positive_expected = expected.gt(0)
    pearson_dispersion = float(
        (
            (valid.loc[positive_expected, "incident_count"] - expected[positive_expected])
            ** 2
            / expected[positive_expected]
        ).sum()
        / max(int(positive_expected.sum()) - 1, 1)
    )
    observed_zero = float(valid["incident_count"].eq(0).mean())
    poisson_expected_zero = float(np.exp(-expected).mean())

    return {
        "operator_year_rows_positive_miles": int(len(valid)),
        "mile_years": mile_years,
        "incident_count": incidents,
        "serious_incident_count": serious,
        "incident_rate_per_1000_mile_year": 1000 * incident_rate,
        "incident_rate_per_1000_approx_95pct": [1000 * i_low, 1000 * i_high],
        "serious_rate_per_1000_mile_year": 1000 * serious_rate,
        "serious_rate_per_1000_approx_95pct": [1000 * s_low, 1000 * s_high],
        "serious_given_incident": serious_given_incident,
        "serious_given_incident_wilson_95pct": [q_low, q_high],
        "poisson_diagnostics": {
            "pearson_dispersion_intercept_only": pearson_dispersion,
            "observed_zero_fraction": observed_zero,
            "expected_zero_fraction": poisson_expected_zero,
        },
        "interval_note": (
            "Poisson rate intervals are lightweight large-sample diagnostics. "
            "The final model should use bootstrap or likelihood/profile intervals."
        ),
    }


def portfolio_rows(risk: dict, unit_miles: float, sizes: list[int]) -> list[dict]:
    rate_per_mile = risk["serious_rate_per_1000_mile_year"] / 1000
    unit_probability = 1 - math.exp(-rate_per_mile * unit_miles)
    rows = []
    for size in sizes:
        exact_probability = 1 - (1 - unit_probability) ** size
        rows.append(
            {
                "portfolio_units": size,
                "unit_miles": unit_miles,
                "total_miles": size * unit_miles,
                "binary_scenario_count": 2**size,
                "unit_serious_probability": unit_probability,
                "exact_probability_at_least_one_serious": exact_probability,
            }
        )
    return rows


def monte_carlo_scaling(
    portfolios: list[dict], budgets: list[int], repetitions: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for portfolio in portfolios:
        truth = portfolio["exact_probability_at_least_one_serious"]
        for sample_budget in budgets:
            # The number of at-least-one-serious outcomes in M independent
            # portfolio simulations is Binomial(M, truth). Sampling it directly
            # is distributionally identical to materializing the full M x K array.
            estimates = rng.binomial(sample_budget, truth, size=repetitions) / sample_budget
            errors = estimates - truth
            rows.append(
                {
                    "portfolio_units": portfolio["portfolio_units"],
                    "binary_scenario_count": str(portfolio["binary_scenario_count"]),
                    "sample_budget": sample_budget,
                    "repetitions": repetitions,
                    "exact_probability": truth,
                    "mean_estimate": float(estimates.mean()),
                    "bias": float(errors.mean()),
                    "mae": float(np.abs(errors).mean()),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "estimate_q025": float(np.quantile(estimates, 0.025)),
                    "estimate_q975": float(np.quantile(estimates, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--unit-miles", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()

    if args.mode == "quick":
        sizes = [2, 4, 8, 16]
        budgets = [64, 256, 1024, 4096]
        repetitions = 250
    else:
        sizes = [2, 4, 8, 16, 32, 64]
        budgets = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
        repetitions = 10000

    input_path = (
        args.project_root
        / "data"
        / "processed"
        / "phmsa"
        / "operator_year_exposure_2017_2025.csv"
    )
    output_dir = args.project_root / "results" / "classical_risk"
    output_dir.mkdir(parents=True, exist_ok=True)
    exposure = pd.read_csv(input_path)

    risk = estimate_pooled_risk(exposure)
    portfolios = portfolio_rows(risk, args.unit_miles, sizes)
    mc = monte_carlo_scaling(portfolios, budgets, repetitions, args.seed)

    risk_path = output_dir / "pooled_risk.json"
    portfolio_path = output_dir / f"portfolio_truth_{args.mode}.csv"
    mc_path = output_dir / f"mc_scaling_{args.mode}.csv"
    risk_path.write_text(json.dumps(risk, indent=2), encoding="utf-8")
    pd.DataFrame(portfolios).to_csv(portfolio_path, index=False)
    mc.to_csv(mc_path, index=False)

    print(json.dumps(risk, indent=2))
    print(f"portfolio: {portfolio_path}")
    print(f"monte carlo: {mc_path}")


if __name__ == "__main__":
    main()
