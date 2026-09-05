#!/usr/bin/env python
"""Cluster bootstrap and data-driven QAE risk-strata construction.

Quick mode is a smoke/stability screen. Full mode is intentionally user-run and
checkpoints results so an interrupted calculation can resume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from psp_classical_pilot import (
    DATA,
    EPS,
    average_precision,
    binary_log_loss,
    calibration,
    category_map,
    fit_nb2,
    fit_ridge_logistic,
    poisson_deviance,
    predict_count,
    roc_auc,
    severity_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "psp_bootstrap"


def cluster_resample(frame: pd.DataFrame, cluster: str, rng: np.random.Generator) -> pd.DataFrame:
    ids = frame[cluster].dropna().unique()
    sampled = rng.choice(ids, size=len(ids), replace=True)
    groups = {key: value for key, value in frame.groupby(cluster, sort=False)}
    pieces = []
    for draw, key in enumerate(sampled):
        piece = groups[key].copy()
        piece["_bootstrap_cluster"] = draw
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def prepare_frequency(exposure: pd.DataFrame):
    frame = exposure[exposure.total_miles > 0].copy().sort_values(["operator_id", "year"])
    frame["hca_fraction"] = (frame.hca_miles / frame.total_miles).clip(0.0, 1.0)
    frame["lag_hca_fraction"] = frame.groupby("operator_id").hca_fraction.shift(1)
    frame["lag_year"] = frame.groupby("operator_id").year.shift(1)
    frame = frame[(frame.year - frame.lag_year == 1) & frame.lag_hca_fraction.notna()].copy()
    train = frame[frame.year <= 2022].copy()
    test = frame[frame.year >= 2023].copy()
    center = float(train.lag_hca_fraction.mean())
    scale = float(train.lag_hca_fraction.std()) or 1.0
    return train, test, center, scale


def frequency_matrix(frame: pd.DataFrame, center: float, scale: float):
    hca = (frame.lag_hca_fraction.to_numpy(float) - center) / scale
    time = (frame.year.to_numpy(float) - 2020.0) / 5.0
    x = np.column_stack([np.ones(len(frame)), hca, time])
    y = frame.incident_count.to_numpy(float)
    offset = np.log(frame.total_miles.to_numpy(float))
    return x, y, offset


def prepare_severity(scenarios: pd.DataFrame):
    development = scenarios[scenarios.year <= 2020].copy()
    final_train = scenarios[scenarios.year <= 2022].copy()
    test = scenarios[scenarios.year >= 2023].copy()
    features = ["facility_type", "on_offshore", "hca", "scada_in_place", "apparent_cause"]
    maps = {feature: category_map(development[feature]) for feature in features}
    x_train, columns = severity_matrix(final_train, maps)
    x_test, _ = severity_matrix(test, maps, columns)
    return final_train, test, maps, columns, x_train, x_test


def fixed_rank_strata(test: pd.DataFrame, probabilities: np.ndarray, scenario_qubits: list[int]):
    assignments = {}
    tables = []
    order = np.argsort(probabilities, kind="stable")
    ranks = np.empty(len(test), dtype=int)
    ranks[order] = np.arange(len(test))
    for qubits in scenario_qubits:
        n_strata = 2**qubits
        stratum = np.minimum((ranks * n_strata) // len(test), n_strata - 1)
        assignments[qubits] = stratum
        work = test[["report_number", "year", "serious_incident"]].copy()
        work["stratum"] = stratum
        work["probability"] = probabilities
        grouped = work.groupby("stratum", sort=True).agg(
            scenario_count=("report_number", "size"),
            observed_serious=("serious_incident", "sum"),
            probability_mean=("probability", "mean"),
            probability_min=("probability", "min"),
            probability_max=("probability", "max"),
        ).reset_index()
        grouped["scenario_qubits"] = qubits
        grouped["binary_scenarios"] = n_strata
        grouped["weight"] = grouped.scenario_count / len(test)
        grouped["expected_contribution"] = grouped.weight * grouped.probability_mean
        grouped["objective_ry_angle_rad"] = 2.0 * np.arcsin(np.sqrt(grouped.probability_mean.clip(0, 1)))
        tables.append(grouped)
    return assignments, pd.concat(tables, ignore_index=True)


def summarise_numeric(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in raw.select_dtypes(include=[np.number]).columns:
        if column == "repeat":
            continue
        values = raw[column].dropna().to_numpy(float)
        if not len(values):
            continue
        rows.append({
            "metric": column,
            "valid_repeats": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "q025": float(np.quantile(values, 0.025)),
            "median": float(np.median(values)),
            "q975": float(np.quantile(values, 0.975)),
        })
    return pd.DataFrame(rows)


def write_checkpoint(rows: list[dict], strata_rows: list[dict], raw_path: Path, strata_path: Path) -> None:
    pd.DataFrame(rows).sort_values("repeat").to_csv(raw_path, index=False)
    pd.DataFrame(strata_rows).sort_values(["repeat", "scenario_qubits", "stratum"]).to_csv(strata_path, index=False)


def bootstrap_once(
    repeat: int,
    seed: int,
    freq_train: pd.DataFrame,
    freq_test: pd.DataFrame,
    center: float,
    scale: float,
    sev_train: pd.DataFrame,
    sev_test: pd.DataFrame,
    maps: dict,
    columns: list[str],
    fixed_x_test: np.ndarray,
    assignments: dict[int, np.ndarray],
    penalty: float,
):
    rng = np.random.default_rng(seed + repeat)
    f_train = cluster_resample(freq_train, "operator_id", rng)
    f_test = cluster_resample(freq_test, "operator_id", rng)
    xtr, ytr, otr = frequency_matrix(f_train, center, scale)
    xte, yte, ote = frequency_matrix(f_test, center, scale)
    f_model = fit_nb2(xtr, ytr, otr)
    f_mu = predict_count(f_model, xte, ote)

    s_train = cluster_resample(sev_train, "operator_id", rng)
    s_test = cluster_resample(sev_test, "operator_id", rng)
    xs_train, _ = severity_matrix(s_train, maps, columns)
    xs_test, _ = severity_matrix(s_test, maps, columns)
    ys_train = s_train.serious_incident.to_numpy(float)
    ys_test = s_test.serious_incident.to_numpy(float)
    s_model = fit_ridge_logistic(xs_train, ys_train, penalty)
    s_prob = expit(xs_test @ s_model["beta"])
    fixed_prob = expit(fixed_x_test @ s_model["beta"])
    if 0 < ys_test.sum() < len(ys_test):
        cal_i, cal_s = calibration(ys_test, s_prob)
    else:
        cal_i, cal_s = np.nan, np.nan

    row = {
        "repeat": repeat,
        "frequency_fit_success": f_model["success"],
        "frequency_events": float(yte.sum()),
        "frequency_predicted_events": float(f_mu.sum()),
        "frequency_observed_to_predicted": float(yte.sum() / max(f_mu.sum(), EPS)),
        "frequency_mean_poisson_deviance": poisson_deviance(yte, f_mu),
        "frequency_alpha": float(f_model["alpha"]),
        "frequency_beta_intercept": float(f_model["beta"][0]),
        "frequency_beta_lag_hca": float(f_model["beta"][1]),
        "frequency_beta_time": float(f_model["beta"][2]),
        "severity_fit_success": s_model["success"],
        "severity_events": float(ys_test.sum()),
        "severity_predicted_events": float(s_prob.sum()),
        "severity_observed_to_predicted": float(ys_test.sum() / max(s_prob.sum(), EPS)),
        "severity_log_loss": binary_log_loss(ys_test, s_prob),
        "severity_brier_score": float(np.mean((ys_test - s_prob) ** 2)),
        "severity_roc_auc": roc_auc(ys_test, s_prob),
        "severity_average_precision": average_precision(ys_test, s_prob),
        "severity_calibration_intercept": cal_i,
        "severity_calibration_slope": cal_s,
        "severity_fixed_test_mean_probability": float(fixed_prob.mean()),
        "severity_beta_json": json.dumps(s_model["beta"].tolist()),
    }
    strata_rows = []
    for qubits, stratum in assignments.items():
        for level in range(2**qubits):
            mask = stratum == level
            strata_rows.append({
                "repeat": repeat,
                "scenario_qubits": qubits,
                "binary_scenarios": 2**qubits,
                "stratum": level,
                "probability_mean": float(fixed_prob[mask].mean()),
            })
    return row, strata_rows


def add_strata_intervals(base: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    stats = raw.groupby(["scenario_qubits", "binary_scenarios", "stratum"]).probability_mean.agg(
        bootstrap_mean="mean", bootstrap_std="std",
        bootstrap_q025=lambda x: x.quantile(0.025),
        bootstrap_median="median",
        bootstrap_q975=lambda x: x.quantile(0.975),
    ).reset_index()
    return base.merge(stats, on=["scenario_qubits", "binary_scenarios", "stratum"], how="left")


def report_text(mode: str, repeats: int, summary: pd.DataFrame, strata: pd.DataFrame) -> str:
    lookup = summary.set_index("metric")
    def interval(metric: str, digits: int = 4):
        row = lookup.loc[metric]
        return f"{row['median']:.{digits}f} [{row['q025']:.{digits}f}, {row['q975']:.{digits}f}]"
    base_means = strata.groupby("scenario_qubits").expected_contribution.sum()
    run_limit = (
        "Quick mode is only a pipeline check and its percentile intervals are too coarse for publication."
        if mode == "quick"
        else "Full mode completed the planned 1,000 resamples. The intervals reflect sampling variation under the specified models, not structural model misspecification."
    )
    return f"""# PSP cluster-bootstrap and QAE-strata report ({mode})

## Run status

- Completed cluster-bootstrap repeats: {repeats}.
- This run resampled operators, preserving each sampled operator's rows.
- Frequency model: NB2 with current mileage offset, lagged HCA fraction, and time.
- Conditional severity: ridge logistic with penalty 1 and fixed prespecified scenario fields.

## Stability screen

- Frequency observed/predicted, median [2.5%, 97.5%]: {interval('frequency_observed_to_predicted', 3)}.
- Frequency test deviance: {interval('frequency_mean_poisson_deviance', 4)}.
- Conditional-severity test log loss: {interval('severity_log_loss', 4)}.
- Conditional-severity test PR-AUC: {interval('severity_average_precision', 4)}.
- Fixed-test mean conditional probability: {interval('severity_fixed_test_mean_probability', 5)}.

## Data-driven QAE targets

- 4-stratum exact weighted mean: {base_means.loc[2]:.6f}.
- 8-stratum exact weighted mean: {base_means.loc[3]:.6f}.
- 16-stratum exact weighted mean: {base_means.loc[4]:.6f}.

The three weighted means agree by construction; increasing the register refines
heterogeneity rather than changing the estimand. `objective_ry_angle_rad` is the
controlled objective rotation for each risk stratum. Distribution-state preparation
cost must still be included in circuit resource accounting.

## Interpretation limit

{run_limit}
Apparent cause is post-incident context, so the severity model is not a prospective
pipeline-segment failure predictor.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    repeats = args.repeats or (20 if args.mode == "quick" else 1000)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    tag = args.mode
    raw_path = out / f"bootstrap_raw_{tag}.csv"
    strata_raw_path = out / f"risk_strata_bootstrap_raw_{tag}.csv"

    scenarios = pd.read_csv(DATA / "incident_scenarios_2010_2025.csv")
    exposure = pd.read_csv(DATA / "operator_year_exposure_2017_2025.csv")
    freq_train, freq_test, center, scale = prepare_frequency(exposure)
    sev_train, sev_test, maps, columns, xsev_train, xsev_test = prepare_severity(scenarios)
    base_severity = fit_ridge_logistic(xsev_train, sev_train.serious_incident.to_numpy(float), penalty=1.0)
    base_probability = expit(xsev_test @ base_severity["beta"])
    assignments, base_strata = fixed_rank_strata(sev_test, base_probability, [2, 3, 4])

    rows: list[dict] = []
    strata_rows: list[dict] = []
    if args.resume and raw_path.exists() and strata_raw_path.exists():
        rows = pd.read_csv(raw_path).to_dict("records")
        strata_rows = pd.read_csv(strata_raw_path).to_dict("records")
    completed = {int(row["repeat"]) for row in rows}
    for repeat in range(repeats):
        if repeat in completed:
            continue
        row, strata_part = bootstrap_once(
            repeat, args.seed, freq_train, freq_test, center, scale,
            sev_train, sev_test, maps, columns, xsev_test, assignments, 1.0,
        )
        rows.append(row)
        strata_rows.extend(strata_part)
        print(f"finished repeat={repeat + 1}/{repeats}", flush=True)
        if (repeat + 1) % args.checkpoint_every == 0:
            write_checkpoint(rows, strata_rows, raw_path, strata_raw_path)
    write_checkpoint(rows, strata_rows, raw_path, strata_raw_path)

    raw = pd.DataFrame(rows).sort_values("repeat")
    strata_raw = pd.DataFrame(strata_rows).sort_values(["repeat", "scenario_qubits", "stratum"])
    summary = summarise_numeric(raw)
    summary.to_csv(out / f"bootstrap_summary_{tag}.csv", index=False)
    strata = add_strata_intervals(base_strata, strata_raw)
    strata.to_csv(out / f"risk_strata_{tag}.csv", index=False)
    for qubits in [2, 3, 4]:
        strata[strata.scenario_qubits == qubits].to_csv(out / f"qae_input_n{qubits}_{tag}.csv", index=False)
    (out / f"BOOTSTRAP_REPORT_{tag}.md").write_text(report_text(tag, len(raw), summary, strata), encoding="utf-8")
    print("\nBootstrap summary")
    wanted = [
        "frequency_observed_to_predicted", "frequency_mean_poisson_deviance",
        "severity_log_loss", "severity_average_precision", "severity_fixed_test_mean_probability",
    ]
    print(summary[summary.metric.isin(wanted)].to_string(index=False))
    print(f"\nQAE strata: {out / f'risk_strata_{tag}.csv'}")


if __name__ == "__main__":
    main()
