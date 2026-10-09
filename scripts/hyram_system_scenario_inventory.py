"""Build a transparent HyRAM+-based hydrogen scenario inventory.

No HyRAM+ source code is imported or redistributed. Public default values in
``data/external/hyram/benchmark_manifest.json`` are independently combined
using equations documented in the cited HyRAM+ technical manual.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ignition_row(mass_flow: float, rows: list[dict]) -> dict:
    if mass_flow < 0.125:
        key = "<0.125"
    elif mass_flow <= 6.25:
        key = "0.125-6.25"
    else:
        key = ">6.25"
    return next(row for row in rows if row["release_rate_kg_per_s"] == key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    manifest_path = (
        args.project_root / "data" / "external" / "hyram" / "benchmark_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    system = manifest["default_gaseous_hydrogen_system"]
    event_tree = manifest["event_tree"]
    detection = float(event_tree["default_detection_and_isolation_probability"])
    leak_sizes = system["leak_sizes_percent_flow_area"]
    mass_flows = system["verified_mass_flow_rates_kg_per_s"]
    counts = system["component_counts"]
    parameters = system["lognormal_annual_leak_frequency_parameters"]

    rows: list[dict] = []
    for component, count in counts.items():
        if count <= 0:
            continue
        for index, (leak_size, mass_flow) in enumerate(zip(leak_sizes, mass_flows)):
            mu, sigma = parameters[component][index]
            median_frequency = math.exp(mu)
            mean_frequency = math.exp(mu + 0.5 * sigma**2)
            ignition = ignition_row(float(mass_flow), event_tree["hydrogen_ignition_probability_by_release_rate"])
            immediate = float(ignition["immediate_ignition"])
            delayed = float(ignition["delayed_ignition"])
            hazard_given_leak = (1 - detection) * (immediate + delayed)
            rows.append(
                {
                    "component": component,
                    "component_count": count,
                    "leak_size_percent": leak_size,
                    "mass_flow_kg_per_s": mass_flow,
                    "ignition_bin_kg_per_s": ignition["release_rate_kg_per_s"],
                    "mu_log_frequency": mu,
                    "sigma_log_frequency": sigma,
                    "median_leak_frequency_per_component_year": median_frequency,
                    "mean_leak_frequency_per_component_year": mean_frequency,
                    "immediate_ignition_probability_given_release": immediate,
                    "delayed_ignition_probability_given_release": delayed,
                    "detection_and_isolation_probability": detection,
                    "hazardous_outcome_probability_given_leak": hazard_given_leak,
                    "system_median_leak_frequency_per_year": count * median_frequency,
                    "system_mean_leak_frequency_per_year": count * mean_frequency,
                    "system_median_hazardous_frequency_per_year": count * median_frequency * hazard_given_leak,
                    "system_mean_hazardous_frequency_per_year": count * mean_frequency * hazard_given_leak,
                }
            )

    frame = pd.DataFrame(rows)
    total_median = frame["system_median_hazardous_frequency_per_year"].sum()
    total_mean = frame["system_mean_hazardous_frequency_per_year"].sum()
    frame["median_hazardous_frequency_share"] = (
        frame["system_median_hazardous_frequency_per_year"] / total_median
    )
    frame["mean_hazardous_frequency_share"] = (
        frame["system_mean_hazardous_frequency_per_year"] / total_mean
    )

    by_size = (
        frame.groupby(["leak_size_percent", "mass_flow_kg_per_s", "ignition_bin_kg_per_s"], as_index=False)
        .agg(
            median_leak_frequency_per_year=("system_median_leak_frequency_per_year", "sum"),
            mean_leak_frequency_per_year=("system_mean_leak_frequency_per_year", "sum"),
            median_hazardous_frequency_per_year=("system_median_hazardous_frequency_per_year", "sum"),
            mean_hazardous_frequency_per_year=("system_mean_hazardous_frequency_per_year", "sum"),
        )
        .sort_values("leak_size_percent")
    )
    by_component = (
        frame.groupby("component", as_index=False)
        .agg(
            component_count=("component_count", "first"),
            median_hazardous_frequency_per_year=("system_median_hazardous_frequency_per_year", "sum"),
            mean_hazardous_frequency_per_year=("system_mean_hazardous_frequency_per_year", "sum"),
        )
        .sort_values("median_hazardous_frequency_per_year", ascending=False)
    )

    weights = frame["median_hazardous_frequency_share"].to_numpy()
    effective_scenarios = float(1 / np.square(weights).sum())
    ordered_shares = np.sort(weights)[::-1]
    opportunities = sum(counts.values()) * len(leak_sizes)
    raw_payoff = frame["median_leak_frequency_per_component_year"] * frame["hazardous_outcome_probability_given_leak"]
    max_payoff = float(raw_payoff.max())
    normalized_amplitude = total_median / (opportunities * max_payoff)
    reconstructed_total = opportunities * max_payoff * normalized_amplitude

    summary = {
        "positive_component_classes": int((np.array(list(counts.values())) > 0).sum()),
        "physical_component_count": int(sum(counts.values())),
        "leak_sizes": len(leak_sizes),
        "component_leak_scenarios": len(frame),
        "median_total_leak_frequency_per_year": float(frame["system_median_leak_frequency_per_year"].sum()),
        "mean_total_leak_frequency_per_year": float(frame["system_mean_leak_frequency_per_year"].sum()),
        "median_hazardous_outcome_frequency_per_year": float(total_median),
        "mean_hazardous_outcome_frequency_per_year": float(total_mean),
        "median_frequency_implied_return_period_years": float(1 / total_median),
        "effective_scenario_count_inverse_simpson": effective_scenarios,
        "top_1_scenario_share": float(ordered_shares[:1].sum()),
        "top_5_scenario_share": float(ordered_shares[:5].sum()),
        "top_10_scenario_share": float(ordered_shares[:10].sum()),
        "amplitude_encoding": {
            "sampling_distribution": "uniform over 92 physical components and 5 leak sizes",
            "payoff": "median component leak frequency times hazardous-outcome probability, divided by the maximum payoff",
            "maximum_raw_payoff": max_payoff,
            "normalized_amplitude": float(normalized_amplitude),
            "reconstruction_multiplier": float(opportunities * max_payoff),
            "reconstructed_frequency_per_year": float(reconstructed_total),
            "absolute_reconstruction_error": float(abs(reconstructed_total - total_median)),
        },
        "interpretation": (
            "Frequency is the primary HyRAM-style quantity. It is not relabeled as an annual "
            "probability. QAE applies to the bounded normalized expectation and the frequency "
            "is recovered by a known multiplier."
        ),
    }

    output_dir = args.project_root / "results" / "hyram" / "system_scenario_inventory"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "component_leak_scenarios.csv", index=False)
    by_size.to_csv(output_dir / "summary_by_leak_size.csv", index=False)
    by_component.to_csv(output_dir / "summary_by_component.csv", index=False)
    (output_dir / "inventory_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].loglog(
        by_size["leak_size_percent"],
        by_size["median_hazardous_frequency_per_year"],
        marker="o",
        label="Median-based frequency",
    )
    axes[0].loglog(
        by_size["leak_size_percent"],
        by_size["mean_hazardous_frequency_per_year"],
        marker="s",
        label="Lognormal mean-based frequency",
    )
    axes[0].set_xlabel("Leak size (% of pipe flow area)")
    axes[0].set_ylabel("Hazardous outcome frequency (per year)")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    plot_components = by_component.sort_values("median_hazardous_frequency_per_year")
    axes[1].barh(
        plot_components["component"],
        plot_components["median_hazardous_frequency_per_year"],
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Median-based hazardous outcome frequency (per year)")
    axes[1].set_ylabel("Component class")
    axes[1].grid(True, axis="x", which="both", alpha=0.25)
    fig.suptitle("HyRAM+ default gaseous-hydrogen scenario inventory")
    fig.tight_layout()
    fig.savefig(output_dir / "scenario_inventory.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps(summary, indent=2))
    print("by leak size")
    print(by_size.to_string(index=False))
    print(f"results: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
