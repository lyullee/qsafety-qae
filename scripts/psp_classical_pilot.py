#!/usr/bin/env python
"""Lightweight publication-screening models for the PHMSA PSP study.

This script intentionally uses only NumPy, pandas, and SciPy.  It is a pilot:
final inference requires clustered resampling or a hierarchical model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "phmsa"
DEFAULT_OUT = ROOT / "results" / "psp_pilot"
EPS = 1e-12


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    mu = np.clip(mu, EPS, None)
    terms = np.where(y > 0, y * np.log(np.clip(y, EPS, None) / mu) - (y - mu), mu)
    return float(2.0 * np.mean(terms))


def fit_poisson(x: np.ndarray, y: np.ndarray, offset: np.ndarray) -> dict:
    def objective(beta: np.ndarray) -> float:
        eta = np.clip(offset + x @ beta, -30.0, 30.0)
        mu = np.exp(eta)
        return float(np.sum(mu - y * eta + gammaln(y + 1.0)))

    init = np.zeros(x.shape[1])
    init[0] = np.log((y.sum() + 0.5) / (np.exp(offset).sum() + 0.5))
    result = minimize(objective, init, method="BFGS")
    return {"beta": result.x, "success": bool(result.success), "nll": float(result.fun)}


def fit_nb2(x: np.ndarray, y: np.ndarray, offset: np.ndarray) -> dict:
    def objective(theta: np.ndarray) -> float:
        beta, log_alpha = theta[:-1], theta[-1]
        alpha = np.exp(np.clip(log_alpha, -12.0, 8.0))
        mu = np.exp(np.clip(offset + x @ beta, -30.0, 30.0))
        size = 1.0 / alpha
        loglik = (
            gammaln(y + size)
            - gammaln(size)
            - gammaln(y + 1.0)
            + size * np.log(size / (size + mu))
            + y * np.log(mu / (size + mu) + EPS)
        )
        return float(-np.sum(loglik))

    pois = fit_poisson(x, y, offset)
    init = np.r_[pois["beta"], np.log(0.5)]
    result = minimize(objective, init, method="L-BFGS-B")
    return {
        "beta": result.x[:-1],
        "alpha": float(np.exp(result.x[-1])),
        "success": bool(result.success),
        "nll": float(result.fun),
    }


def predict_count(model: dict, x: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(offset + x @ model["beta"], -30.0, 30.0))


def binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(p)
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(y: np.ndarray, p: np.ndarray) -> float:
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    ys = y[order]
    precision = np.cumsum(ys) / np.arange(1, len(ys) + 1)
    return float(np.sum(precision * ys) / positives)


def fit_ridge_logistic(x: np.ndarray, y: np.ndarray, penalty: float) -> dict:
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(x @ beta, -30.0, 30.0)
        p = expit(eta)
        penalized = beta.copy()
        penalized[0] = 0.0
        value = np.sum(np.logaddexp(0.0, eta) - y * eta) + 0.5 * penalty * np.dot(penalized, penalized)
        grad = x.T @ (p - y) + penalty * penalized
        return float(value), grad

    prevalence = np.clip(y.mean(), 1e-5, 1 - 1e-5)
    init = np.zeros(x.shape[1])
    init[0] = np.log(prevalence / (1.0 - prevalence))
    result = minimize(lambda b: objective(b)[0], init, jac=lambda b: objective(b)[1], method="L-BFGS-B")
    return {"beta": result.x, "success": bool(result.success), "penalty": float(penalty)}


def calibration(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    score = np.log(p / (1.0 - p))
    x = np.column_stack([np.ones(len(p)), score])
    model = fit_ridge_logistic(x, y, penalty=1e-6)
    return float(model["beta"][0]), float(model["beta"][1])


def category_map(train: pd.Series, min_count: int = 50) -> set[str]:
    counts = train.fillna("UNKNOWN").astype(str).value_counts()
    return set(counts[counts >= min_count].index)


def severity_matrix(frame: pd.DataFrame, maps: dict[str, set[str]], columns: list[str] | None = None):
    features = ["facility_type", "on_offshore", "hca", "scada_in_place", "apparent_cause"]
    encoded = {}
    for feature in features:
        values = frame[feature].fillna("UNKNOWN").astype(str)
        encoded[feature] = values.where(values.isin(maps[feature]), "OTHER")
    design = pd.get_dummies(pd.DataFrame(encoded), prefix=features, dtype=float)
    if columns is None:
        columns = sorted(design.columns)
    design = design.reindex(columns=columns, fill_value=0.0)
    x = np.column_stack([np.ones(len(frame)), design.to_numpy(float)])
    return x, columns


def count_metrics(name: str, split: str, y: np.ndarray, mu: np.ndarray, n_parameters: int, nll: float | None = None):
    return {
        "component": "incident_frequency",
        "model": name,
        "split": split,
        "rows": int(len(y)),
        "events": int(y.sum()),
        "predicted_events": float(mu.sum()),
        "observed_to_predicted": float(y.sum() / max(mu.sum(), EPS)),
        "mean_poisson_deviance": poisson_deviance(y, mu),
        "parameters": int(n_parameters),
        "nll": nll,
    }


def binary_metrics(name: str, split: str, y: np.ndarray, p: np.ndarray, penalty: float | None = None):
    cal_i, cal_s = calibration(y, p)
    return {
        "component": "conditional_severity",
        "model": name,
        "split": split,
        "rows": int(len(y)),
        "events": int(y.sum()),
        "prevalence": float(y.mean()),
        "mean_prediction": float(p.mean()),
        "observed_to_predicted": float(y.sum() / max(p.sum(), EPS)),
        "log_loss": binary_log_loss(y, p),
        "brier_score": float(np.mean((y - p) ** 2)),
        "roc_auc": roc_auc(y, p),
        "average_precision": average_precision(y, p),
        "calibration_intercept": cal_i,
        "calibration_slope": cal_s,
        "penalty": penalty,
    }


def build_audit(scenarios: pd.DataFrame, exposure: pd.DataFrame, out: Path) -> None:
    positive = exposure[exposure["total_miles"] > 0].copy()
    flow = pd.DataFrame(
        [
            ["incident_scenarios_2010_2025", len(scenarios), int(scenarios.serious_incident.sum()), "conditional severity"],
            ["operator_year_all", len(exposure), int(exposure.serious_incident_count.sum()), "exposure table"],
            ["operator_year_positive_miles", len(positive), int(positive.serious_incident_count.sum()), "frequency denominator"],
        ],
        columns=["stage", "rows", "serious_incidents", "purpose"],
    )
    flow.to_csv(out / "data_flow.csv", index=False)

    missing = []
    for dataset, frame in [("incident_scenarios", scenarios), ("operator_year_exposure", exposure)]:
        for column in frame.columns:
            missing.append([dataset, column, len(frame), int(frame[column].isna().sum()), float(frame[column].isna().mean())])
    pd.DataFrame(missing, columns=["dataset", "variable", "rows", "missing", "missing_fraction"]).to_csv(
        out / "missingness.csv", index=False
    )

    annual_incident = scenarios.groupby("year").agg(
        incident_scenarios=("report_number", "size"), serious_incidents=("serious_incident", "sum")
    )
    annual_exposure = positive.groupby("year").agg(
        operator_years=("operator_id", "size"), miles=("total_miles", "sum"),
        matched_incidents=("incident_count", "sum"), matched_serious=("serious_incident_count", "sum")
    )
    annual_incident.join(annual_exposure, how="outer").reset_index().to_csv(out / "annual_outcomes.csv", index=False)

    support = []
    categorical = [
        "on_offshore", "system_part", "facility_type", "pipeline_function", "material",
        "class_location", "hca", "pressure_status", "pressure_restriction",
        "internal_inspection_capable", "scada_in_place", "apparent_cause",
    ]
    for feature in categorical:
        grouped = scenarios.groupby(feature, dropna=False).serious_incident.agg(["count", "sum"]).reset_index()
        for _, row in grouped.iterrows():
            support.append([
                feature, str(row[feature]), int(row["count"]), int(row["sum"]),
                float(row["sum"] / row["count"]), bool(row["count"] < 20), bool(row["sum"] == 0),
            ])
    pd.DataFrame(
        support,
        columns=["variable", "level", "incidents", "serious_incidents", "serious_fraction", "fewer_than_20", "zero_serious"],
    ).to_csv(out / "category_support.csv", index=False)


def run_frequency(exposure: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = exposure[exposure.total_miles > 0].copy().sort_values(["operator_id", "year"])
    frame["hca_fraction"] = (frame.hca_miles / frame.total_miles).clip(0.0, 1.0)
    frame["lag_hca_fraction"] = frame.groupby("operator_id").hca_fraction.shift(1)
    frame["lag_year"] = frame.groupby("operator_id").year.shift(1)
    frame = frame[(frame.year - frame.lag_year == 1) & frame.lag_hca_fraction.notna()].copy()
    train = frame[frame.year <= 2022].copy()
    test = frame[frame.year >= 2023].copy()
    center = float(train.lag_hca_fraction.mean())
    scale = float(train.lag_hca_fraction.std()) or 1.0

    def matrices(part: pd.DataFrame, covariates: bool):
        offset = np.log(part.total_miles.to_numpy(float))
        if covariates:
            hca = (part.lag_hca_fraction.to_numpy(float) - center) / scale
            time = (part.year.to_numpy(float) - 2020.0) / 5.0
            x = np.column_stack([np.ones(len(part)), hca, time])
        else:
            x = np.ones((len(part), 1))
        return x, part.incident_count.to_numpy(float), offset

    rows = []
    fitted = {}
    for name, family, covariates in [
        ("poisson_intercept", "poisson", False),
        ("poisson_lag_hca_time", "poisson", True),
        ("nb2_lag_hca_time", "nb2", True),
    ]:
        xtr, ytr, otr = matrices(train, covariates)
        model = fit_poisson(xtr, ytr, otr) if family == "poisson" else fit_nb2(xtr, ytr, otr)
        fitted[name] = model
        for split, part in [("train_2018_2022", train), ("test_2023_2025", test)]:
            x, y, offset = matrices(part, covariates)
            mu = predict_count(model, x, offset)
            rows.append(count_metrics(name, split, y, mu, len(model["beta"]), model["nll"] if split.startswith("train") else None))
    meta = {
        "train_rows": len(train), "test_rows": len(test), "hca_center": center, "hca_scale": scale,
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "beta"} | {"beta": v["beta"].tolist()} for k, v in fitted.items()},
        "warning": "Current-year mileage is the exposure offset; HCA fraction is lagged one year. Final inference needs operator-cluster uncertainty.",
    }
    return pd.DataFrame(rows), meta


def run_severity(scenarios: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train = scenarios[scenarios.year <= 2020].copy()
    validation = scenarios[scenarios.year.between(2021, 2022)].copy()
    test = scenarios[scenarios.year >= 2023].copy()
    features = ["facility_type", "on_offshore", "hca", "scada_in_place", "apparent_cause"]
    maps = {feature: category_map(train[feature]) for feature in features}
    xtr, columns = severity_matrix(train, maps)
    xval, _ = severity_matrix(validation, maps, columns)
    xtest, _ = severity_matrix(test, maps, columns)
    ytr = train.serious_incident.to_numpy(float)
    yval = validation.serious_incident.to_numpy(float)
    ytest = test.serious_incident.to_numpy(float)

    penalties = [0.1, 1.0, 10.0, 100.0]
    tuning = []
    candidates = {}
    for penalty in penalties:
        model = fit_ridge_logistic(xtr, ytr, penalty)
        candidates[penalty] = model
        pval = expit(xval @ model["beta"])
        tuning.append({"penalty": penalty, "validation_log_loss": binary_log_loss(yval, pval), "validation_brier": float(np.mean((yval-pval)**2))})
    tuning_df = pd.DataFrame(tuning).sort_values(["validation_log_loss", "penalty"])
    chosen = float(tuning_df.iloc[0].penalty)

    final_train = pd.concat([train, validation], ignore_index=True)
    xfinal, final_columns = severity_matrix(final_train, maps)
    xtest_final, _ = severity_matrix(test, maps, final_columns)
    yfinal = final_train.serious_incident.to_numpy(float)
    model = fit_ridge_logistic(xfinal, yfinal, chosen)

    rows = []
    baseline_p = np.full(len(test), yfinal.mean())
    model_p = expit(xtest_final @ model["beta"])
    rows.append(binary_metrics("pooled_prevalence", "test_2023_2025", ytest, baseline_p))
    rows.append(binary_metrics("ridge_scenario", "test_2023_2025", ytest, model_p, chosen))
    prediction = test[["report_number", "year", "serious_incident"]].copy()
    prediction["pooled_probability"] = baseline_p
    prediction["ridge_probability"] = model_p

    meta = {
        "development_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
        "development_events": int(ytr.sum()), "validation_events": int(yval.sum()), "test_events": int(ytest.sum()),
        "selected_penalty": chosen, "design_columns": final_columns, "category_maps": {k: sorted(v) for k, v in maps.items()},
        "fit_success": model["success"],
        "warning": "Apparent cause is post-incident scenario context. This model is conditional consequence analysis, not prospective failure prediction.",
    }
    return pd.DataFrame(rows), tuning_df, prediction, meta


def write_report(out: Path, frequency: pd.DataFrame, severity: pd.DataFrame, meta: dict) -> None:
    ftest = frequency[frequency.split == "test_2023_2025"].copy()
    stest = severity[severity.split == "test_2023_2025"].copy()
    best_frequency = ftest.sort_values("mean_poisson_deviance").iloc[0]
    baseline_s = stest[stest.model == "pooled_prevalence"].iloc[0]
    ridge_s = stest[stest.model == "ridge_scenario"].iloc[0]
    improvement = (baseline_s.log_loss - ridge_s.log_loss) / baseline_s.log_loss
    text = f"""# PSP classical pilot report

## Purpose

This is a lightweight screening run, not final inference. It tests whether the PHMSA
tables can support a frequency–consequence model before clustered bootstrap and QAE
state preparation are run.

## Incident-frequency pilot

- Training: operator-years through 2022 with a valid one-year lag.
- Test: 2023–2025.
- Current-year mileage is the exposure offset; HCA fraction is lagged one year.
- Lowest test mean Poisson deviance: `{best_frequency['model']}` = {best_frequency['mean_poisson_deviance']:.6f}.
- Its observed/predicted event ratio is {best_frequency['observed_to_predicted']:.3f}.

## Conditional-severity pilot

- Development/validation/test serious events: {meta['severity']['development_events']}/
  {meta['severity']['validation_events']}/{meta['severity']['test_events']}.
- Validation-selected ridge penalty: {meta['severity']['selected_penalty']:g}.
- Pooled test log loss: {baseline_s.log_loss:.6f}.
- Scenario-model test log loss: {ridge_s.log_loss:.6f}.
- Relative log-loss change versus pooled baseline: {improvement:+.1%}.
- Scenario-model test Brier score: {ridge_s.brier_score:.6f}; PR-AUC: {ridge_s.average_precision:.4f}.

`apparent_cause` is only a post-incident scenario condition. These probabilities must
not be presented as prospective segment-failure predictions.

## Decision rule

Proceed to the data-driven QAE prototype only if the statistical model yields stable,
calibrated heterogeneous probabilities after clustered resampling. If the scenario
model does not improve proper scoring rules over the pooled baseline, reduce model
complexity or redefine the safety estimand before scaling quantum experiments.
"""
    (out / "PILOT_REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    scenarios = pd.read_csv(DATA / "incident_scenarios_2010_2025.csv")
    exposure = pd.read_csv(DATA / "operator_year_exposure_2017_2025.csv")
    build_audit(scenarios, exposure, out)
    frequency, frequency_meta = run_frequency(exposure)
    severity, tuning, predictions, severity_meta = run_severity(scenarios)
    frequency.to_csv(out / "frequency_pilot_metrics.csv", index=False)
    severity.to_csv(out / "severity_pilot_metrics.csv", index=False)
    tuning.to_csv(out / "severity_penalty_tuning.csv", index=False)
    predictions.to_csv(out / "severity_test_predictions.csv", index=False)
    meta = {"frequency": frequency_meta, "severity": severity_meta}
    (out / "pilot_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(out, frequency, severity, meta)
    print("Frequency pilot")
    print(frequency.to_string(index=False))
    print("\nConditional-severity pilot")
    print(severity.to_string(index=False))
    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
