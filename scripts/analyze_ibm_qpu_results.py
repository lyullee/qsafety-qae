#!/usr/bin/env python
"""Consolidate IBM QPU runs and plot the observed MLQAE power response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from qae_heterogeneous_risk import OUTPUT_DIR


def split_numbers(value: str, cast=float) -> list:
    return [cast(item) for item in str(value).split("|")]


def load_results() -> pd.DataFrame:
    frames = []
    for path in sorted(OUTPUT_DIR.glob("ibm_results_*.csv")):
        frame = pd.read_csv(path)
        frame["result_path"] = str(path.resolve())
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No IBM result files found in {OUTPUT_DIR}")
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["job_id", "scenario_qubits"], keep="last"
    )


def add_submission_metadata(results: pd.DataFrame) -> pd.DataFrame:
    records = []
    for path in sorted(OUTPUT_DIR.glob("ibm_submission_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        records.append({
            "job_id": data.get("job_id"),
            "submitted_utc": data.get("submitted_utc"),
            "completed_utc": data.get("completed_utc"),
            "actual_usage_seconds": data.get("actual_usage"),
            "backend_properties_last_update": data.get("backend_properties_last_update"),
            "qiskit_version": data.get("qiskit_version"),
            "qiskit_ibm_runtime_version": data.get("qiskit_ibm_runtime_version"),
        })
    if not records:
        return results
    return results.merge(pd.DataFrame(records), on="job_id", how="left")


def power_response(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in results.iterrows():
        powers = split_numbers(row["powers"], int)
        goods = split_numbers(row["good_counts"], int)
        observed = split_numbers(row["good_fractions"], float)
        ideal = split_numbers(row["ideal_good_probabilities"], float)
        for power, count, obs, target in zip(powers, goods, observed, ideal):
            rows.append({
                "backend": row["backend"],
                "job_id": row["job_id"],
                "scenario_qubits": int(row["scenario_qubits"]),
                "risk_strata": int(row["risk_strata"]),
                "shots": int(row["shots_per_power"]),
                "grover_power": power,
                "good_count": count,
                "observed_good_probability": obs,
                "ideal_good_probability": target,
                "response_error": obs - target,
            })
    return pd.DataFrame(rows)


def plot_response(power: pd.DataFrame, shots: int, destination: Path) -> None:
    selected = power[power.shots == shots]
    strata_values = sorted(selected.risk_strata.unique())
    if not strata_values:
        raise SystemExit(f"No QPU results found for shots={shots}")
    fig, axes = plt.subplots(1, len(strata_values), figsize=(4.6 * len(strata_values), 4.2), sharey=True)
    if len(strata_values) == 1:
        axes = [axes]
    colors = {"ibm_fez": "#0072B2", "ibm_marrakesh": "#D55E00"}
    for axis, strata in zip(axes, strata_values):
        panel = selected[selected.risk_strata == strata]
        ideal = panel.groupby("grover_power", as_index=False).ideal_good_probability.first()
        axis.plot(
            ideal.grover_power, ideal.ideal_good_probability,
            color="black", marker="o", linestyle="--", label="Ideal",
        )
        for backend, group in panel.groupby("backend"):
            aggregate = group.groupby("grover_power", as_index=False).agg(
                mean_observed=("observed_good_probability", "mean"),
                std_observed=("observed_good_probability", "std"),
            ).sort_values("grover_power")
            axis.errorbar(
                aggregate.grover_power, aggregate.mean_observed,
                yerr=aggregate.std_observed.fillna(0.0),
                marker="o", linewidth=2, capsize=3, label=backend,
                color=colors.get(backend),
            )
        axis.set_title(f"{strata} risk strata")
        axis.set_xlabel("Grover power m")
        axis.set_xticks(sorted(panel.grover_power.unique()))
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Objective-qubit success probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"IBM QPU response versus ideal MLQAE trajectory ({shots} shots)", y=0.98)
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.92),
        ncol=len(labels), frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(destination, dpi=220, bbox_inches="tight")
    fig.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=4096)
    args = parser.parse_args()

    results = add_submission_metadata(load_results())
    power = power_response(results)
    comparison = results[results.shots_per_power == args.shots].sort_values(
        ["risk_strata", "backend"]
    )

    all_path = OUTPUT_DIR / "ibm_qpu_results_all.csv"
    comparison_path = OUTPUT_DIR / f"ibm_qpu_comparison_s{args.shots}.csv"
    summary_path = OUTPUT_DIR / f"ibm_qpu_repeat_summary_s{args.shots}.csv"
    power_path = OUTPUT_DIR / f"ibm_qpu_power_response_s{args.shots}.csv"
    figure_path = OUTPUT_DIR / f"ibm_qpu_power_response_s{args.shots}.png"
    results.to_csv(all_path, index=False)
    comparison.to_csv(comparison_path, index=False)
    repeat_summary = comparison.groupby(["backend", "risk_strata"], as_index=False).agg(
        independent_jobs=("job_id", "nunique"),
        mean_mle_probability=("mle_probability", "mean"),
        std_mle_probability=("mle_probability", "std"),
        mean_absolute_error=("absolute_error", "mean"),
        mean_relative_error=("relative_error", "mean"),
        mean_fit_rmse=("mle_fit_rmse", "mean"),
        model_mismatch_rate=("fit_status", lambda values: float((values == "model_mismatch").mean())),
    )
    repeat_summary.to_csv(summary_path, index=False)
    power[power.shots == args.shots].to_csv(power_path, index=False)
    plot_response(power, args.shots, figure_path)

    columns = [
        "backend", "risk_strata", "mle_probability", "relative_error",
        "mle_fit_rmse", "mle_fit_max_error", "fit_status", "actual_usage_seconds",
    ]
    print(comparison[columns].to_string(index=False))
    print(f"Comparison: {comparison_path}")
    print(f"Repeat summary: {summary_path}")
    print(f"Power response: {power_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
