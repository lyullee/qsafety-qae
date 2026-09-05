#!/usr/bin/env python
"""Safe IBM-QPU check, preview, and explicit submission for heterogeneous PHMSA QAE.

`check` and `preview` never submit a workload. `run` requires --confirm-qpu.
Open Plan auto-selection is enforced when loading the saved account.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

from qae_heterogeneous_risk import OUTPUT_DIR, load_strata, make_a_gate, make_qae_circuit
from qae_qiskit_runner import good_count, mle_amplitude, mle_fit_diagnostics


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def service_open_only() -> QiskitRuntimeService:
    return QiskitRuntimeService(
        channel="ibm_quantum_platform",
        plans_preference=["open"],
        region="us-east",
    )


def available_backend_rows(service: QiskitRuntimeService) -> list[dict]:
    rows = []
    for backend in service.backends(operational=True, simulator=False):
        try:
            pending = int(backend.status().pending_jobs)
        except Exception:
            pending = -1
        rows.append({
            "backend": backend.name,
            "qubits": int(backend.num_qubits),
            "pending_jobs": pending,
        })
    return sorted(rows, key=lambda row: (row["pending_jobs"] < 0, row["pending_jobs"], row["backend"]))


def build_problem(scenario_qubits: list[int], powers: list[int], input_tag: str):
    circuits = []
    keys = []
    exact = {}
    for n in scenario_qubits:
        strata = load_strata(n, input_tag)
        weights = strata.weight.to_numpy(float)
        risks = strata.probability_mean.to_numpy(float)
        exact[n] = float(np.dot(weights, risks))
        a_gate, _ = make_a_gate(weights, risks)
        for power in powers:
            circuits.append(make_qae_circuit(a_gate, n, power, measure=True))
            keys.append((n, power))
    return circuits, keys, exact


def select_backend(service: QiskitRuntimeService, name: str | None, min_qubits: int):
    if name:
        backend = service.backend(name)
        if backend.num_qubits < min_qubits:
            raise ValueError(f"{name} has fewer than {min_qubits} qubits")
        return backend
    return service.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)


def preview_rows(isa_circuits, keys, backend, shots: int) -> tuple[list[dict], dict]:
    rows = []
    duration_sum = 0.0
    duration_available = True
    for circuit, (n, power) in zip(isa_circuits, keys):
        operations = {str(k): int(v) for k, v in circuit.count_ops().items()}
        duration = None
        try:
            duration = float(circuit.estimate_duration(backend.target))
            duration_sum += duration
        except Exception:
            duration_available = False
        rows.append({
            "backend": backend.name,
            "scenario_qubits": n,
            "risk_strata": 2**n,
            "total_logical_qubits": n + 1,
            "grover_power": power,
            "shots": shots,
            "isa_depth": circuit.depth(),
            "isa_gate_count": circuit.size(),
            "isa_two_qubit_gates": sum(
                value for gate, value in operations.items()
                if gate in {"cx", "cz", "ecr", "rzz", "iswap"}
            ),
            "estimated_circuit_seconds": duration,
            "isa_operations": json.dumps(operations),
        })
    executions = len(isa_circuits) * shots
    estimate = {
        "backend": backend.name,
        "circuits": len(isa_circuits),
        "shots_per_circuit": shots,
        "executions": executions,
        "ibm_quick_formula_seconds": 2.0 + 0.00035 * executions,
        "duration_based_seconds_excluding_reset": (
            2.0 + shots * (duration_sum + len(isa_circuits) * float(backend.default_rep_delay))
            if duration_available else None
        ),
        "note": "Estimates exclude advanced mitigation and can differ from billed QPU usage.",
    }
    return rows, estimate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check", "preview", "run"], default="check")
    parser.add_argument("--scenario-qubits", type=int, nargs="+", default=[2])
    parser.add_argument("--powers", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--input-tag", default="full")
    parser.add_argument("--backend")
    parser.add_argument("--optimization", type=int, choices=[0, 1, 2, 3], default=3)
    parser.add_argument("--max-execution-time", type=int, default=300)
    parser.add_argument("--confirm-qpu", action="store_true")
    args = parser.parse_args()
    if args.mode == "run" and not args.confirm_qpu:
        raise SystemExit("QPU submission blocked: add --confirm-qpu after reviewing preview")

    service = service_open_only()
    backends = available_backend_rows(service)
    print("Saved account loaded with plans_preference=['open'], region='us-east'.")
    print(pd.DataFrame(backends).head(10).to_string(index=False))
    if args.mode == "check":
        print("No QPU workload was submitted.")
        return

    circuits, keys, exact = build_problem(args.scenario_qubits, args.powers, args.input_tag)
    backend = select_backend(service, args.backend, max(circuit.num_qubits for circuit in circuits))
    manager = generate_preset_pass_manager(backend=backend, optimization_level=args.optimization)
    isa_circuits = manager.run(circuits)
    rows, estimate = preview_rows(isa_circuits, keys, backend, args.shots)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    size_label = "-".join(map(str, args.scenario_qubits))
    preview_path = OUTPUT_DIR / f"ibm_preview_{backend.name}_n{size_label}_s{args.shots}.csv"
    estimate_path = OUTPUT_DIR / f"ibm_usage_estimate_{backend.name}_n{size_label}_s{args.shots}.json"
    pd.DataFrame(rows).to_csv(preview_path, index=False)
    estimate_path.write_text(json.dumps(estimate, indent=2), encoding="utf-8")
    print("\nBackend-specific ISA preview")
    print(pd.DataFrame(rows)[[
        "backend", "scenario_qubits", "risk_strata", "grover_power",
        "isa_depth", "isa_gate_count", "isa_two_qubit_gates",
    ]].to_string(index=False))
    print("\nUsage estimate")
    print(json.dumps(estimate, indent=2))
    print(f"Preview: {preview_path}")
    print("No QPU workload was submitted." if args.mode == "preview" else "Preparing confirmed submission.")
    if args.mode == "preview":
        return

    sampler = SamplerV2(mode=backend)
    sampler.options.max_execution_time = args.max_execution_time
    properties_last_update = None
    try:
        properties_last_update = backend.properties().last_update_date.isoformat()
    except Exception:
        pass
    job = sampler.run(isa_circuits, shots=args.shots)
    submitted = {
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend.name,
        "job_id": job.job_id(),
        "scenario_qubits": args.scenario_qubits,
        "powers": args.powers,
        "shots": args.shots,
        "max_execution_time": args.max_execution_time,
        "optimization_level": args.optimization,
        "backend_properties_last_update": properties_last_update,
        "qiskit_version": package_version("qiskit"),
        "qiskit_ibm_runtime_version": package_version("qiskit-ibm-runtime"),
    }
    submission_path = OUTPUT_DIR / f"ibm_submission_{job.job_id()}.json"
    submission_path.write_text(json.dumps(submitted, indent=2), encoding="utf-8")
    print(f"Submitted job {job.job_id()} to {backend.name}; waiting for result.")
    result = job.result()
    counts = [publication.data.meas.get_counts() for publication in result]
    result_rows = []
    for n in args.scenario_qubits:
        selected = [i for i, key in enumerate(keys) if key[0] == n]
        goods = [good_count(counts[i]) for i in selected]
        estimate_value = mle_amplitude(goods, args.shots, args.powers)
        theta = float(np.arcsin(np.sqrt(exact[n])))
        ideal_good_probabilities = [
            float(np.sin((2 * power + 1) * theta) ** 2) for power in args.powers
        ]
        result_rows.append({
            "backend": backend.name,
            "job_id": job.job_id(),
            "scenario_qubits": n,
            "risk_strata": 2**n,
            "shots_per_power": args.shots,
            "powers": "|".join(map(str, args.powers)),
            "good_counts": "|".join(map(str, goods)),
            "good_fractions": "|".join(f"{value / args.shots:.12g}" for value in goods),
            "ideal_good_probabilities": "|".join(
                f"{value:.12g}" for value in ideal_good_probabilities
            ),
            "raw_counts": json.dumps(
                [counts[i] for i in selected], sort_keys=True, separators=(",", ":")
            ),
            "exact_probability": exact[n],
            "mle_probability": estimate_value,
            "signed_error": estimate_value - exact[n],
            "absolute_error": abs(estimate_value - exact[n]),
            "relative_error": abs(estimate_value - exact[n]) / exact[n],
            **mle_fit_diagnostics(estimate_value, goods, args.shots, args.powers),
        })
    usage = None
    try:
        usage = job.usage()
    except Exception:
        pass
    metrics = None
    try:
        metrics = job.metrics()
    except Exception:
        pass
    submitted["completed_utc"] = datetime.now(timezone.utc).isoformat()
    submitted["actual_usage"] = usage
    submitted["actual_usage_unit"] = "seconds"
    submitted["job_metrics"] = metrics
    submission_path.write_text(json.dumps(submitted, indent=2, default=str), encoding="utf-8")
    hardware_path = OUTPUT_DIR / f"ibm_results_{job.job_id()}.csv"
    pd.DataFrame(result_rows).to_csv(hardware_path, index=False)
    print(pd.DataFrame(result_rows).to_string(index=False))
    print(f"Actual usage: {usage} seconds")
    print(f"Results: {hardware_path}")


if __name__ == "__main__":
    main()
