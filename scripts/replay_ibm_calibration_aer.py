#!/usr/bin/env python
"""Replay submitted IBM-QPU QAE jobs with historical calibration Aer noise models.

This is a local simulator only: it never creates a QPU job.  IBM's historic
BackendProperties currently omit a frequency field required by Aer thermal
relaxation, so the replay uses calibration-derived gate and readout errors while
explicitly disabling thermal relaxation.  It is therefore a calibration-informed
proxy, not a bit-for-bit digital twin of hardware.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from qae_heterogeneous_ibm import build_problem, service_open_only
from qae_heterogeneous_risk import OUTPUT_DIR
from qae_qiskit_runner import good_count, mle_amplitude, mle_fit_diagnostics


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def submission_records() -> dict[str, dict]:
    records = {}
    for path in OUTPUT_DIR.glob("ibm_submission_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("job_id"):
            records[data["job_id"]] = data
    return records


def hardware_rows(job_ids: list[str] | None) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in OUTPUT_DIR.glob("ibm_results_*.csv")]
    if not frames:
        raise SystemExit("No IBM result CSV exists under results/qae_heterogeneous")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["shots_per_power"] == 4096].copy()
    if job_ids:
        frame = frame[frame["job_id"].isin(job_ids)].copy()
    if frame.empty:
        raise SystemExit("No matching 4096-shot IBM result rows found")
    return frame.sort_values(["backend", "job_id", "scenario_qubits"])


def running_time(submission: dict) -> datetime:
    metrics = submission.get("job_metrics") or {}
    timestamp = (metrics.get("timestamps") or {}).get("running")
    return parse_timestamp(timestamp or submission["submitted_utc"])


def isa_resource_row(circuit, backend: str, n: int, power: int) -> dict:
    operations = {str(key): int(value) for key, value in circuit.count_ops().items()}
    return {
        "backend": backend,
        "scenario_qubits": n,
        "risk_strata": 2**n,
        "grover_power": power,
        "replay_isa_depth": int(circuit.depth()),
        "replay_isa_gate_count": int(circuit.size()),
        "replay_isa_two_qubit_gates": sum(
            value for gate, value in operations.items()
            if gate in {"cx", "cz", "ecr", "rzz", "iswap"}
        ),
        "replay_isa_operations": json.dumps(operations, sort_keys=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--job-ids", nargs="+")
    parser.add_argument("--input-tag", default="full")
    args = parser.parse_args()
    repeats = args.repeats if args.repeats is not None else (1 if args.mode == "quick" else 30)
    if repeats < 1:
        raise ValueError("--repeats must be positive")

    hardware = hardware_rows(args.job_ids)
    submissions = submission_records()
    service = service_open_only()
    raw_rows: list[dict] = []
    summary_rows: list[dict] = []
    resource_rows: list[dict] = []

    for job_id, job_rows in hardware.groupby("job_id", sort=False):
        submission = submissions.get(job_id)
        if submission is None:
            raise SystemExit(f"Missing submission JSON for job {job_id}")
        backend_name = str(job_rows.backend.iloc[0])
        timestamp = running_time(submission)
        backend = service.backend(backend_name)
        properties = backend.properties(datetime=timestamp)
        target = backend.target_history(datetime=timestamp)
        if properties is None:
            raise SystemExit(f"No historical BackendProperties for {backend_name} at {timestamp.isoformat()}")
        pass_manager = generate_preset_pass_manager(
            target=target,
            optimization_level=int(submission.get("optimization_level", 3)),
        )
        scenario_qubits = sorted(job_rows.scenario_qubits.astype(int).unique().tolist())
        powers = [int(value) for value in str(job_rows.powers.iloc[0]).split("|")]
        shots = int(job_rows.shots_per_power.iloc[0])
        circuits, keys, exact = build_problem(scenario_qubits, powers, args.input_tag)
        isa_circuits = pass_manager.run(circuits)
        for circuit, (n, power) in zip(isa_circuits, keys):
            resource_rows.append(isa_resource_row(circuit, backend_name, n, power))

        noise_model = NoiseModel.from_backend_properties(
            properties,
            gate_error=True,
            readout_error=True,
            thermal_relaxation=False,
        )
        simulator = AerSimulator(
            noise_model=noise_model,
            method="density_matrix",
            enable_truncation=True,
        )
        print(
            f"replaying job={job_id} backend={backend_name} repeats={repeats} "
            f"calibration={properties.last_update_date}"
        )
        for repeat in range(repeats):
            result = simulator.run(
                isa_circuits,
                shots=shots,
                seed_simulator=args.seed + repeat,
            ).result()
            counts = [result.get_counts(index) for index in range(len(isa_circuits))]
            for n in scenario_qubits:
                selected = [index for index, key in enumerate(keys) if key[0] == n]
                good_counts = [good_count(counts[index]) for index in selected]
                estimate = mle_amplitude(good_counts, shots, powers)
                raw_rows.append({
                    "job_id": job_id,
                    "backend": backend_name,
                    "hardware_running_utc": timestamp.isoformat(),
                    "calibration_last_update": properties.last_update_date.isoformat(),
                    "scenario_qubits": n,
                    "risk_strata": 2**n,
                    "repeat": repeat,
                    "shots_per_power": shots,
                    "powers": "|".join(map(str, powers)),
                    "good_counts": "|".join(map(str, good_counts)),
                    "mle_probability": estimate,
                    "exact_probability": exact[n],
                    "absolute_error": abs(estimate - exact[n]),
                    **mle_fit_diagnostics(estimate, good_counts, shots, powers),
                    "gate_error": True,
                    "readout_error": True,
                    "thermal_relaxation": False,
                })
        print(f"finished job={job_id}")

    raw = pd.DataFrame(raw_rows)
    resources = pd.DataFrame(resource_rows)
    for (job_id, backend, n, strata), group in raw.groupby(
        ["job_id", "backend", "scenario_qubits", "risk_strata"], sort=False
    ):
        observed = hardware[(hardware.job_id == job_id) & (hardware.scenario_qubits == n)].iloc[0]
        summary_rows.append({
            "job_id": job_id,
            "backend": backend,
            "scenario_qubits": n,
            "risk_strata": strata,
            "repeats": len(group),
            "hardware_mle_probability": float(observed.mle_probability),
            "hardware_fit_rmse": float(observed.mle_fit_rmse),
            "hardware_fit_status": observed.fit_status,
            "aer_mean_mle_probability": float(group.mle_probability.mean()),
            "aer_std_mle_probability": float(group.mle_probability.std(ddof=0)),
            "aer_mean_absolute_error": float(group.absolute_error.mean()),
            "aer_mean_fit_rmse": float(group.mle_fit_rmse.mean()),
            "aer_model_mismatch_rate": float((group.fit_status == "model_mismatch").mean()),
            "hardware_minus_aer_mle": float(observed.mle_probability - group.mle_probability.mean()),
            "gate_error": True,
            "readout_error": True,
            "thermal_relaxation": False,
            "interpretation_limit": "historical gate/readout calibration proxy; thermal relaxation omitted",
        })
    summary = pd.DataFrame(summary_rows)

    suffix = f"{args.mode}_r{repeats}"
    raw_path = OUTPUT_DIR / f"ibm_calibration_aer_raw_{suffix}.csv"
    summary_path = OUTPUT_DIR / f"ibm_calibration_aer_summary_{suffix}.csv"
    resource_path = OUTPUT_DIR / f"ibm_calibration_aer_resources_{suffix}.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    resources.to_csv(resource_path, index=False)
    print(summary.to_string(index=False))
    print(f"Raw: {raw_path}")
    print(f"Summary: {summary_path}")
    print(f"Resources: {resource_path}")
    print("No QPU workload was submitted.")


if __name__ == "__main__":
    main()
