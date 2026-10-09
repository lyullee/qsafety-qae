"""Propagate HyRAM+ leak-frequency uncertainty into the QAE target amplitude.

The primary case mirrors the structure created by HyRAM+'s uncertainty module:
one lognormal variable for every physical component and leak size.  Additional
rank-correlation cases are sensitivity tests, not claims about the true
dependence structure.  HyRAM+ source code is neither imported nor redistributed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODELS = [
    "independent_physical_component_size",
    "shared_component_class_size",
    "shared_within_component_class",
    "shared_by_leak_size",
    "fully_rank_correlated",
]


def sample_amplitudes(
    rng: np.random.Generator,
    draws: int,
    batch_size: int,
    model: str,
    mus: np.ndarray,
    sigmas: np.ndarray,
    class_index: np.ndarray,
    ignition: np.ndarray,
    isolation_failure: float,
) -> np.ndarray:
    """Return hazardous-trajectory amplitudes for one dependence model."""
    n_components = len(class_index)
    n_classes, n_sizes = mus.shape
    mu_physical = mus[class_index]
    sigma_physical = sigmas[class_index]
    out = np.empty(draws, dtype=float)
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        size = stop - start
        if model == "independent_physical_component_size":
            z = rng.standard_normal((size, n_components, n_sizes))
            frequency = np.exp(mu_physical[None, :, :] + sigma_physical[None, :, :] * z)
        elif model == "shared_component_class_size":
            z = rng.standard_normal((size, n_classes, n_sizes))
            class_frequency = np.exp(mus[None, :, :] + sigmas[None, :, :] * z)
            frequency = class_frequency[:, class_index, :]
        elif model == "shared_within_component_class":
            z = rng.standard_normal((size, n_classes, 1))
            class_frequency = np.exp(mus[None, :, :] + sigmas[None, :, :] * z)
            frequency = class_frequency[:, class_index, :]
        elif model == "shared_by_leak_size":
            z = rng.standard_normal((size, 1, n_sizes))
            frequency = np.exp(mu_physical[None, :, :] + sigma_physical[None, :, :] * z)
        elif model == "fully_rank_correlated":
            z = rng.standard_normal((size, 1, 1))
            frequency = np.exp(mu_physical[None, :, :] + sigma_physical[None, :, :] * z)
        else:
            raise ValueError(f"Unknown dependence model: {model}")

        leak_probability = -np.expm1(-frequency)
        hazardous = leak_probability * isolation_failure * ignition[None, None, :]
        out[start:stop] = hazardous.mean(axis=(1, 2))
    return out


def summarize(values: np.ndarray, model: str, nominal: float) -> dict:
    quantiles = np.quantile(values, [0.005, 0.025, 0.05, 0.5, 0.95, 0.975, 0.995])
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    median = float(quantiles[3])
    return {
        "dependence_model": model,
        "draws": len(values),
        "nominal_median_parameter_amplitude": nominal,
        "mean_amplitude": mean,
        "median_amplitude": median,
        "standard_deviation": std,
        "coefficient_of_variation": std / mean,
        "q0_5": float(quantiles[0]),
        "q2_5": float(quantiles[1]),
        "q5": float(quantiles[2]),
        "q95": float(quantiles[4]),
        "q97_5": float(quantiles[5]),
        "q99_5": float(quantiles[6]),
        "relative_95pct_interval_width_vs_median": float(
            (quantiles[5] - quantiles[1]) / median
        ),
        "mean_to_nominal_ratio": mean / nominal,
        "probability_within_5pct_of_nominal": float(
            np.mean(np.abs(values - nominal) <= 0.05 * nominal)
        ),
        "probability_within_10pct_of_nominal": float(
            np.mean(np.abs(values - nominal) <= 0.10 * nominal)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--draws-per-seed", type=int)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    draws_per_seed = args.draws_per_seed or (20_000 if args.mode == "quick" else 200_000)
    seeds = args.seeds or (2 if args.mode == "quick" else 5)
    root = args.project_root

    manifest_path = root / "data" / "external" / "hyram" / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    system = manifest["default_gaseous_hydrogen_system"]
    counts_dict = system["component_counts"]
    params = system["lognormal_annual_leak_frequency_parameters"]
    categories = [name for name, count in counts_dict.items() if count > 0]
    counts = np.asarray([counts_dict[name] for name in categories], dtype=int)
    mus = np.asarray([[pair[0] for pair in params[name]] for name in categories])
    sigmas = np.asarray([[pair[1] for pair in params[name]] for name in categories])
    class_index = np.repeat(np.arange(len(categories)), counts)
    ignition = np.asarray([0.012, 0.012, 0.012, 0.08, 0.08], dtype=float)
    isolation_failure = 0.1

    nominal_frequency = np.exp(mus)[class_index]
    nominal = float(
        np.mean(-np.expm1(-nominal_frequency) * isolation_failure * ignition[None, :])
    )
    trajectory_path = (
        root / "results" / "hyram" / "stochastic_trajectory_oracle"
        / "trajectory_summary.json"
    )
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    prior_nominal = float(trajectory["random_opportunity_hazard_probability"])
    if not math.isclose(nominal, prior_nominal, rel_tol=1e-12, abs_tol=1e-16):
        raise AssertionError(f"Nominal amplitude mismatch: {nominal} vs {prior_nominal}")

    output_dir = root / "results" / "hyram" / "leak_frequency_uncertainty"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_rows: list[dict] = []
    pooled_by_model: dict[str, np.ndarray] = {}
    for model_index, model in enumerate(MODELS):
        chunks = []
        for seed_index in range(seeds):
            rng = np.random.default_rng(args.seed + 10_000 * model_index + seed_index)
            values = sample_amplitudes(
                rng, draws_per_seed, args.batch_size, model, mus, sigmas,
                class_index, ignition, isolation_failure,
            )
            chunks.append(values)
            row = summarize(values, model, nominal)
            row["seed_index"] = seed_index + 1
            row["random_seed"] = args.seed + 10_000 * model_index + seed_index
            seed_rows.append(row)
            print(f"finished model={model}, seed={seed_index + 1}/{seeds}")
        pooled_by_model[model] = np.concatenate(chunks)

    seed_frame = pd.DataFrame(seed_rows)
    summary = pd.DataFrame(
        [summarize(pooled_by_model[model], model, nominal) for model in MODELS]
    )
    summary.insert(2, "physical_components", len(class_index))
    summary.insert(3, "leak_sizes", len(ignition))

    cost_path = (
        root / "results" / "hyram" / "precision_conditional_cost_map"
        / "precision_cost_map_full.csv"
    )
    cost = (
        pd.read_csv(cost_path)
        .sort_values(["target_relative_rmse", "conditional_sample_cost_multiplier"])
        .drop_duplicates("target_relative_rmse")
    )
    primary = summary.loc[
        summary["dependence_model"].eq("independent_physical_component_size")
    ].iloc[0]
    comparison_rows = []
    for row in cost.itertuples(index=False):
        target_relative = float(row.target_relative_rmse)
        comparison_rows.append(
            {
                "target_relative_rmse": target_relative,
                "qae_queries": int(row.qae_queries),
                "qae_absolute_rmse_target_at_nominal": nominal * target_relative,
                "primary_input_standard_deviation": float(primary.standard_deviation),
                "input_sd_to_qae_rmse_target_ratio": float(
                    primary.standard_deviation / (nominal * target_relative)
                ),
                "interpretation": (
                    "input uncertainty exceeds computational RMSE target"
                    if primary.standard_deviation > nominal * target_relative
                    else "computational RMSE target exceeds input uncertainty"
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows)

    suffix = f"{args.mode}_s{seeds}_n{draws_per_seed}"
    seed_frame.to_csv(output_dir / f"uncertainty_by_seed_{suffix}.csv", index=False)
    summary.to_csv(output_dir / f"uncertainty_summary_{suffix}.csv", index=False)
    comparison.to_csv(output_dir / f"uncertainty_vs_qae_precision_{suffix}.csv", index=False)
    metadata = {
        "mode": args.mode,
        "draws_per_seed": draws_per_seed,
        "seeds": seeds,
        "pooled_draws_per_model": draws_per_seed * seeds,
        "nominal_amplitude": nominal,
        "primary_model": "independent_physical_component_size",
        "primary_model_basis": (
            "HyRAM+ creates one lognormal distribution variable per physical "
            "component and leak size; independent LHS columns are mirrored here."
        ),
        "sensitivity_models_note": (
            "Dependence alternatives bracket unspecified rank correlation and "
            "must not be interpreted as calibrated correlation models."
        ),
        "source_manifest": str(manifest_path),
        "audited_hyram_commit": manifest["audited_commit"],
    }
    (output_dir / f"uncertainty_metadata_{suffix}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    labels = [
        "Independent\nphysical components",
        "Shared class-size",
        "Shared within class",
        "Shared by leak size",
        "Fully correlated",
    ]
    medians = summary["median_amplitude"].to_numpy(float)
    lower = medians - summary["q2_5"].to_numpy(float)
    upper = summary["q97_5"].to_numpy(float) - medians
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.errorbar(
        np.arange(len(summary)), medians, yerr=np.vstack([lower, upper]),
        fmt="o", capsize=5, color="#1769aa", ecolor="#5c85b8", linewidth=2,
    )
    ax.axhline(nominal, color="#c62828", linestyle="--", label="Median-parameter target")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(summary)), labels)
    ax.set_ylabel("Hazardous-trajectory probability per random opportunity")
    ax.set_title("HyRAM+ leak-frequency uncertainty: dependence sensitivity")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"leak_frequency_uncertainty_{suffix}.{extension}", dpi=220)
    plt.close(fig)

    columns = [
        "dependence_model", "mean_amplitude", "median_amplitude",
        "coefficient_of_variation", "q2_5", "q97_5",
        "relative_95pct_interval_width_vs_median", "mean_to_nominal_ratio",
    ]
    print("\nLeak-frequency uncertainty")
    print(summary[columns].to_string(index=False))
    print("\nInput uncertainty versus QAE precision")
    print(comparison.to_string(index=False))
    print(f"\nresults: {output_dir}")


if __name__ == "__main__":
    main()
