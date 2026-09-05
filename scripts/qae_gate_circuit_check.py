"""Construct and verify small gate-level QAE building blocks without Qiskit.

The circuit models K independent PHMSA risk units.  Risk qubit i is prepared
with Ry(2 asin(sqrt(p))), and an objective qubit is set to one exactly when at
least one risk qubit is one.  The simulator then applies Grover powers and
checks their probabilities against sin^2((2m+1) theta).

Multi-controlled gates remain logical gates in the exported manifest.  Their
hardware-specific one- and two-qubit decomposition is deliberately deferred to
the later transpilation/hardware stage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def apply_ry(state: np.ndarray, qubit: int, theta: float) -> None:
    step = 1 << qubit
    view = state.reshape(-1, 2, step)
    zero = view[:, 0, :].copy()
    one = view[:, 1, :].copy()
    cosine = math.cos(theta / 2)
    sine = math.sin(theta / 2)
    view[:, 0, :] = cosine * zero - sine * one
    view[:, 1, :] = sine * zero + cosine * one


def apply_x(state: np.ndarray, qubit: int) -> None:
    step = 1 << qubit
    view = state.reshape(-1, 2, step)
    zero = view[:, 0, :].copy()
    view[:, 0, :] = view[:, 1, :]
    view[:, 1, :] = zero


def apply_mcx(state: np.ndarray, controls: list[int], target: int) -> None:
    indices = np.arange(state.size, dtype=np.int64)
    selected = (indices & (1 << target)) == 0
    for control in controls:
        selected &= (indices & (1 << control)) != 0
    low = indices[selected]
    high = low | (1 << target)
    temporary = state[low].copy()
    state[low] = state[high]
    state[high] = temporary


def mark_any_risk(state: np.ndarray, risk_qubits: int) -> None:
    """X(objective) iff OR(risk[0:K]); this reversible circuit is self-inverse."""
    controls = list(range(risk_qubits))
    objective = risk_qubits
    for qubit in controls:
        apply_x(state, qubit)
    apply_mcx(state, controls, objective)
    apply_x(state, objective)
    for qubit in reversed(controls):
        apply_x(state, qubit)


def apply_a(state: np.ndarray, risk_qubits: int, theta: float) -> None:
    for qubit in range(risk_qubits):
        apply_ry(state, qubit, theta)
    mark_any_risk(state, risk_qubits)


def apply_a_dagger(state: np.ndarray, risk_qubits: int, theta: float) -> None:
    mark_any_risk(state, risk_qubits)
    for qubit in reversed(range(risk_qubits)):
        apply_ry(state, qubit, -theta)


def apply_good_phase(state: np.ndarray, objective: int) -> None:
    indices = np.arange(state.size, dtype=np.int64)
    state[(indices & (1 << objective)) != 0] *= -1


def apply_zero_phase(state: np.ndarray) -> None:
    state[0] *= -1


def apply_grover(state: np.ndarray, risk_qubits: int, theta: float) -> None:
    # Q = A S0 A^dagger S_good, up to an irrelevant global phase.
    apply_good_phase(state, risk_qubits)
    apply_a_dagger(state, risk_qubits, theta)
    apply_zero_phase(state)
    apply_a(state, risk_qubits, theta)


def objective_probability(state: np.ndarray, objective: int) -> float:
    indices = np.arange(state.size, dtype=np.int64)
    return float(np.sum(np.abs(state[(indices & (1 << objective)) != 0]) ** 2))


def state_preparation_manifest(risk_qubits: int, theta: float) -> list[dict]:
    gates = [
        {"gate": "ry", "qubits": [qubit], "theta_radians": theta}
        for qubit in range(risk_qubits)
    ]
    gates.extend(
        {"gate": "x", "qubits": [qubit]} for qubit in range(risk_qubits)
    )
    gates.append(
        {
            "gate": "mcx",
            "controls": list(range(risk_qubits)),
            "target": risk_qubits,
        }
    )
    gates.append({"gate": "x", "qubits": [risk_qubits]})
    gates.extend(
        {"gate": "x", "qubits": [qubit]}
        for qubit in reversed(range(risk_qubits))
    )
    return gates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--powers", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--unit-miles", type=float, default=1000.0)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()

    root = args.project_root
    risk_path = root / "results" / "classical_risk" / "pooled_risk.json"
    risk = json.loads(risk_path.read_text(encoding="utf-8"))
    rate_per_mile = risk["serious_rate_per_1000_mile_year"] / 1000
    unit_probability = 1 - math.exp(-rate_per_mile * args.unit_miles)
    rotation = 2 * math.asin(math.sqrt(unit_probability))

    rows: list[dict] = []
    manifests: dict[str, dict] = {}
    for risk_qubits in args.sizes:
        total_qubits = risk_qubits + 1
        exact = 1 - (1 - unit_probability) ** risk_qubits
        amplitude_angle = math.asin(math.sqrt(exact))
        prepared = np.zeros(1 << total_qubits, dtype=np.complex128)
        prepared[0] = 1
        apply_a(prepared, risk_qubits, rotation)
        prepared_probability = objective_probability(prepared, risk_qubits)

        manifests[str(risk_qubits)] = {
            "risk_qubits": risk_qubits,
            "objective_qubit": risk_qubits,
            "total_qubits": total_qubits,
            "meaning": "objective=1 iff at least one risk qubit is 1",
            "state_preparation": state_preparation_manifest(risk_qubits, rotation),
            "grover_iteration": [
                {"gate": "z", "qubits": [risk_qubits], "role": "S_good"},
                {"gate": "A_dagger"},
                {"gate": "mcz_zero_reflection", "qubits": list(range(total_qubits))},
                {"gate": "A"},
            ],
            "logical_counts_state_preparation": {
                "ry": risk_qubits,
                "x": 2 * risk_qubits + 1,
                "mcx": 1,
            },
            "logical_counts_per_grover_iteration": {
                "ry": 2 * risk_qubits,
                "x": 4 * risk_qubits + 2,
                "mcx": 2,
                "z": 1,
                "multi_controlled_zero_reflection": 1,
            },
        }

        for power in args.powers:
            state = prepared.copy()
            for _ in range(power):
                apply_grover(state, risk_qubits, rotation)
            measured = objective_probability(state, risk_qubits)
            theoretical = math.sin((2 * power + 1) * amplitude_angle) ** 2
            rows.append(
                {
                    "risk_qubits": risk_qubits,
                    "objective_qubits": 1,
                    "total_qubits": total_qubits,
                    "binary_risk_scenarios": 2**risk_qubits,
                    "grover_power": power,
                    "unit_serious_probability": unit_probability,
                    "exact_at_least_one_probability": exact,
                    "prepared_objective_probability": prepared_probability,
                    "theoretical_amplified_probability": theoretical,
                    "circuit_amplified_probability": measured,
                    "absolute_error": abs(measured - theoretical),
                    "norm_error": abs(float(np.vdot(state, state).real) - 1),
                }
            )

    output_dir = root / "results" / "qae_risk"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gate_circuit_check.csv"
    json_path = output_dir / "gate_circuits.json"
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(manifests, indent=2), encoding="utf-8")

    print(frame.to_string(index=False))
    print(f"max circuit/formula error: {frame['absolute_error'].max():.3e}")
    print(f"max norm error: {frame['norm_error'].max():.3e}")
    print(f"circuit checks: {csv_path}")
    print(f"gate manifests: {json_path}")


if __name__ == "__main__":
    main()
