#!/usr/bin/env python
"""Data-driven heterogeneous QAE circuits for PHMSA risk strata.

The scenario register prepares empirical stratum weights. A uniformly controlled
Ry maps the fitted Serious-Incident probability for each stratum to an objective
qubit. All reported transpiled resources include both operations.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, UCRYGate
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator

from qae_qiskit_runner import custom_noise_model, good_count, mle_amplitude, mle_fit_diagnostics


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "results" / "psp_bootstrap"
OUTPUT_DIR = ROOT / "results" / "qae_heterogeneous"
BASIS = ["rz", "sx", "x", "cx"]


def load_strata(scenario_qubits: int, tag: str) -> pd.DataFrame:
    path = INPUT_DIR / f"qae_input_n{scenario_qubits}_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing QAE input: {path}; run psp_bootstrap_strata.py first")
    frame = pd.read_csv(path).sort_values("stratum").reset_index(drop=True)
    expected = 2**scenario_qubits
    if len(frame) != expected or frame.stratum.tolist() != list(range(expected)):
        raise ValueError(f"n={scenario_qubits} requires exactly {expected} ordered strata")
    if (frame.weight < 0).any() or (frame.probability_mean < 0).any() or (frame.probability_mean > 1).any():
        raise ValueError("Weights and probabilities must be valid probabilities")
    frame["weight"] = frame.weight / frame.weight.sum()
    return frame


def make_a_gate(weights: np.ndarray, probabilities: np.ndarray):
    n = int(round(math.log2(len(weights))))
    qc = QuantumCircuit(n + 1, name=f"A_heterogeneous_n{n}")
    qc.append(StatePreparation(np.sqrt(weights).astype(complex)), list(range(n)))
    angles = (2.0 * np.arcsin(np.sqrt(probabilities))).tolist()
    # UCRY qarg order is target first, followed by controls. Scenario qubit 0
    # is the least-significant control, matching the CSV stratum index.
    qc.append(UCRYGate(angles), [n, *range(n)])
    return qc.to_gate(label=f"Ahet{n}"), qc


def zero_reflection(qc: QuantumCircuit, n: int) -> None:
    qubits = list(range(n + 1))
    objective = n
    qc.x(qubits)
    qc.h(objective)
    qc.mcx(list(range(n)), objective)
    qc.h(objective)
    qc.x(qubits)


def make_qae_circuit(a_gate, n: int, power: int, measure: bool = True) -> QuantumCircuit:
    objective = n
    qc = QuantumCircuit(n + 1, name=f"heterogeneous_n{n}_m{power}")
    qc.append(a_gate, list(range(n + 1)))
    for _ in range(power):
        qc.z(objective)
        qc.append(a_gate.inverse(), list(range(n + 1)))
        zero_reflection(qc, n)
        qc.append(a_gate, list(range(n + 1)))
    if measure:
        meas = ClassicalRegister(1, "meas")
        qc.add_register(meas)
        qc.measure(objective, meas[0])
    return qc


def objective_probability(circuit: QuantumCircuit, objective: int) -> float:
    probabilities = Statevector.from_instruction(circuit).probabilities()
    indices = np.arange(len(probabilities))
    return float(probabilities[((indices >> objective) & 1) == 1].sum())


def preparation_check(a_circuit: QuantumCircuit, weights: np.ndarray, risks: np.ndarray) -> dict:
    n = a_circuit.num_qubits - 1
    probabilities = Statevector.from_instruction(a_circuit).probabilities()
    expected = np.r_[weights * (1.0 - risks), weights * risks]
    return {
        "prepared_probability_sum": float(probabilities.sum()),
        "max_basis_probability_error": float(np.max(np.abs(probabilities - expected))),
        "objective_probability": float(probabilities[2**n:].sum()),
        "exact_weighted_probability": float(np.dot(weights, risks)),
    }


def resource_record(label: str, circuit: QuantumCircuit, backend, optimization: int) -> dict:
    physical = transpile(circuit, basis_gates=BASIS, optimization_level=optimization)
    operations = {str(key): int(value) for key, value in physical.count_ops().items()}
    return {
        "component": label,
        "logical_qubits": circuit.num_qubits,
        "logical_depth": circuit.depth(),
        "logical_gate_count": circuit.size(),
        "logical_operations": json.dumps({str(k): int(v) for k, v in circuit.count_ops().items()}),
        "transpiled_depth": physical.depth(),
        "transpiled_gate_count": physical.size(),
        "transpiled_cx": operations.get("cx", 0),
        "transpiled_operations": json.dumps(operations),
    }, physical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["build", "ideal", "noisy"], default="ideal")
    parser.add_argument("--scenario-qubits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--powers", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--input-tag", default="full")
    parser.add_argument("--output-tag", default="main")
    parser.add_argument("--optimization", type=int, choices=[0, 1, 2, 3], default=2)
    parser.add_argument("--p1", type=float, default=0.0001)
    parser.add_argument("--p2", type=float, default=0.001)
    parser.add_argument("--readout-error", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    for value in [args.input_tag, args.output_tag]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("Tags may contain only letters, numbers, underscore, or hyphen")
    if min(args.scenario_qubits) < 1 or min(args.powers) < 0 or args.shots < 1:
        raise ValueError("Invalid qubit, power, or shot setting")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "noisy":
        noise = custom_noise_model(args.p1, args.p2, args.readout_error)
        backend = AerSimulator(method="density_matrix", noise_model=noise)
    else:
        backend = AerSimulator(method="statevector")

    validation_rows = []
    resource_rows = []
    result_rows = []
    for n in args.scenario_qubits:
        strata = load_strata(n, args.input_tag)
        weights = strata.weight.to_numpy(float)
        risks = strata.probability_mean.to_numpy(float)
        exact = float(np.dot(weights, risks))
        a_gate, a_circuit = make_a_gate(weights, risks)
        check = preparation_check(a_circuit, weights, risks)
        theta = math.asin(math.sqrt(exact))
        formula_errors = []
        circuits = []
        physical_circuits = []

        state_only = QuantumCircuit(n)
        state_only.append(StatePreparation(np.sqrt(weights).astype(complex)), range(n))
        objective_only = QuantumCircuit(n + 1)
        objective_only.append(UCRYGate((2 * np.arcsin(np.sqrt(risks))).tolist()), [n, *range(n)])
        for label, component in [("distribution_state_preparation", state_only), ("objective_rotation", objective_only), ("A_total", a_circuit)]:
            record, _ = resource_record(label, component, backend, args.optimization)
            record.update({"scenario_qubits": n, "risk_strata": 2**n, "grover_power": np.nan})
            resource_rows.append(record)

        for power in args.powers:
            unmeasured = make_qae_circuit(a_gate, n, power, measure=False)
            measured = make_qae_circuit(a_gate, n, power, measure=True)
            state_probability = objective_probability(unmeasured, n)
            formula_probability = math.sin((2 * power + 1) * theta) ** 2
            formula_errors.append(abs(state_probability - formula_probability))
            record, physical = resource_record(f"QAE_m{power}", measured, backend, args.optimization)
            record.update({"scenario_qubits": n, "risk_strata": 2**n, "grover_power": power})
            resource_rows.append(record)
            circuits.append(measured)
            physical_circuits.append(physical)
            qasm_path = OUTPUT_DIR / f"heterogeneous_n{n}_m{power}_{args.output_tag}.qasm"
            qasm_path.write_text(qasm2_dumps(physical), encoding="utf-8")

        validation_rows.append({
            "scenario_qubits": n,
            "risk_strata": 2**n,
            **check,
            "max_grover_formula_error": float(max(formula_errors)),
        })

        if args.mode != "build":
            run = backend.run(physical_circuits, shots=args.shots, seed_simulator=args.seed + n).result()
            counts = [run.get_counts(i) for i in range(len(physical_circuits))]
            goods = [good_count(value) for value in counts]
            estimate = mle_amplitude(goods, args.shots, args.powers)
            diagnostics = mle_fit_diagnostics(estimate, goods, args.shots, args.powers)
            result_rows.append({
                "mode": args.mode,
                "scenario_qubits": n,
                "risk_strata": 2**n,
                "total_qubits": n + 1,
                "shots_per_power": args.shots,
                "powers": "|".join(map(str, args.powers)),
                "good_counts": "|".join(map(str, goods)),
                "exact_probability": exact,
                "mle_probability": estimate,
                "signed_error": estimate - exact,
                "absolute_error": abs(estimate - exact),
                "relative_error": abs(estimate - exact) / exact,
                **diagnostics,
            })

    validation = pd.DataFrame(validation_rows)
    resources = pd.DataFrame(resource_rows)
    results = pd.DataFrame(result_rows)
    suffix = f"_{args.mode}_{args.output_tag}"
    validation.to_csv(OUTPUT_DIR / f"validation{suffix}.csv", index=False)
    resources.to_csv(OUTPUT_DIR / f"resources{suffix}.csv", index=False)
    if len(results):
        results.to_csv(OUTPUT_DIR / f"results{suffix}.csv", index=False)
    print("State preparation and Grover validation")
    print(validation.to_string(index=False))
    if len(results):
        print("\nML-QAE results")
        print(results.to_string(index=False))
    print("\nQAE circuit resources")
    print(resources[resources.component.str.startswith("QAE")][[
        "scenario_qubits", "risk_strata", "grover_power", "transpiled_depth", "transpiled_gate_count", "transpiled_cx"
    ]].to_string(index=False))
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
