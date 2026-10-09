"""Construct small HyRAM+-anchored stochastic safety-trajectory oracles.

The script contrasts a collapsed payoff lookup with an explicit trajectory:
leak occurrence -> isolation failure -> ignition -> hazardous outcome. HyRAM+
source code is neither imported nor redistributed. No QPU job is submitted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation, UCRYGate, grover_operator
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit.quantum_info import Statevector


BASIS = ["rz", "sx", "x", "cx"]
COMPONENT_QUBITS = [0, 1, 2]
LEAK_SIZE_QUBITS = [3, 4, 5]
SCENARIO_QUBITS = COMPONENT_QUBITS + LEAK_SIZE_QUBITS


def ry_angles(probabilities: np.ndarray) -> list[float]:
    probabilities = np.clip(probabilities.astype(float), 0.0, 1.0)
    return (2 * np.arcsin(np.sqrt(probabilities))).tolist()


def scenario_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    components = list(dict.fromkeys(frame["component"].tolist()))
    leak_sizes = sorted(frame["leak_size_percent"].unique().tolist())
    if len(components) != 8 or len(leak_sizes) != 5:
        raise ValueError("Expected eight positive component classes and five leak sizes")

    component_weights = np.zeros(8)
    leak_weights = np.zeros(8)
    leak_weights[:5] = 1 / 5
    leak_probabilities = np.zeros(64)
    ignition_probabilities = np.zeros(8)

    for component_index, component in enumerate(components):
        subset = frame.loc[frame["component"].eq(component)]
        component_weights[component_index] = float(subset["component_count"].iloc[0])
        for leak_index, leak_size in enumerate(leak_sizes):
            row = subset.loc[np.isclose(subset["leak_size_percent"], leak_size)].iloc[0]
            scenario_index = component_index + 8 * leak_index
            frequency = float(row["median_leak_frequency_per_component_year"])
            # Explicit one-year Poisson conversion; frequency is not relabeled
            # directly as probability.
            leak_probabilities[scenario_index] = -math.expm1(-frequency)
            ignition_probabilities[leak_index] = float(
                row["immediate_ignition_probability_given_release"]
                + row["delayed_ignition_probability_given_release"]
            )
    component_weights /= component_weights.sum()
    return component_weights, leak_weights, leak_probabilities, ignition_probabilities


def prepare_scenario(
    component_weights: np.ndarray,
    leak_weights: np.ndarray,
    factorized: bool,
    total_qubits: int,
) -> QuantumCircuit:
    circuit = QuantumCircuit(total_qubits, name="scenario_state")
    if factorized:
        circuit.append(
            StatePreparation(np.sqrt(component_weights).astype(complex)),
            COMPONENT_QUBITS,
        )
        circuit.append(
            StatePreparation(np.sqrt(leak_weights).astype(complex)),
            LEAK_SIZE_QUBITS,
        )
    else:
        joint = np.zeros(64)
        for leak_index in range(8):
            for component_index in range(8):
                joint[component_index + 8 * leak_index] = (
                    component_weights[component_index] * leak_weights[leak_index]
                )
        circuit.append(
            StatePreparation(np.sqrt(joint).astype(complex)), SCENARIO_QUBITS
        )
    return circuit


def collapsed_oracle(
    component_weights: np.ndarray,
    leak_weights: np.ndarray,
    leak_probabilities: np.ndarray,
    ignition_probabilities: np.ndarray,
    isolation_failure: float,
    factorized: bool,
) -> tuple[QuantumCircuit, int]:
    objective = 6
    circuit = prepare_scenario(
        component_weights, leak_weights, factorized, total_qubits=7
    )
    payoff = np.zeros(64)
    for scenario_index in range(64):
        leak_index = scenario_index // 8
        payoff[scenario_index] = (
            leak_probabilities[scenario_index]
            * isolation_failure
            * ignition_probabilities[leak_index]
        )
    circuit.append(UCRYGate(ry_angles(payoff)), [objective, *SCENARIO_QUBITS])
    return circuit, objective


def trajectory_oracle(
    component_weights: np.ndarray,
    leak_weights: np.ndarray,
    leak_probabilities: np.ndarray,
    ignition_probabilities: np.ndarray,
    isolation_failure: float,
    factorized: bool,
) -> tuple[QuantumCircuit, int]:
    leak_event, isolation_event, ignition_event, objective = 6, 7, 8, 9
    circuit = prepare_scenario(
        component_weights, leak_weights, factorized, total_qubits=10
    )
    circuit.append(
        UCRYGate(ry_angles(leak_probabilities)),
        [leak_event, *SCENARIO_QUBITS],
    )
    circuit.ry(2 * math.asin(math.sqrt(isolation_failure)), isolation_event)
    circuit.append(
        UCRYGate(ry_angles(ignition_probabilities)),
        [ignition_event, *LEAK_SIZE_QUBITS],
    )
    circuit.mcx([leak_event, isolation_event, ignition_event], objective)
    return circuit, objective


def objective_probability(circuit: QuantumCircuit, objective: int) -> float:
    probabilities = Statevector.from_instruction(circuit).probabilities()
    indices = np.arange(len(probabilities))
    return float(probabilities[((indices >> objective) & 1) == 1].sum())


def qae_circuit(a_operator: QuantumCircuit, objective: int, power: int) -> QuantumCircuit:
    circuit = a_operator.copy(name=f"A_Q{power}")
    if power:
        phase_oracle = QuantumCircuit(a_operator.num_qubits, name="S_chi")
        phase_oracle.z(objective)
        grover = grover_operator(
            oracle=phase_oracle,
            state_preparation=a_operator,
            reflection_qubits=list(range(a_operator.num_qubits)),
        )
        for _ in range(power):
            circuit.compose(grover, inplace=True)
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimization", type=int, choices=[0, 1, 2, 3], default=2)
    parser.add_argument("--powers", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    root = args.project_root
    input_path = (
        root / "results" / "hyram" / "system_scenario_inventory"
        / "component_leak_scenarios.csv"
    )
    frame = pd.read_csv(input_path)
    component_weights, leak_weights, leak_probabilities, ignition_probabilities = (
        scenario_arrays(frame)
    )
    isolation_failure = 0.1
    joint_weights = np.asarray(
        [
            component_weights[component_index] * leak_weights[leak_index]
            for leak_index in range(8)
            for component_index in range(8)
        ]
    )
    payoff = np.asarray(
        [
            leak_probabilities[index]
            * isolation_failure
            * ignition_probabilities[index // 8]
            for index in range(64)
        ]
    )
    exact = float(np.dot(joint_weights, payoff))
    physical_opportunities = int(frame["component_count"].sum())
    expected_system_hazardous_opportunities = exact * physical_opportunities

    methods: dict[str, tuple[QuantumCircuit, int]] = {}
    for factorized in (False, True):
        state_name = "factorized_state" if factorized else "generic_joint_state"
        methods[f"collapsed_payoff__{state_name}"] = collapsed_oracle(
            component_weights,
            leak_weights,
            leak_probabilities,
            ignition_probabilities,
            isolation_failure,
            factorized,
        )
        methods[f"explicit_trajectory__{state_name}"] = trajectory_oracle(
            component_weights,
            leak_weights,
            leak_probabilities,
            ignition_probabilities,
            isolation_failure,
            factorized,
        )

    output_dir = root / "results" / "hyram" / "stochastic_trajectory_oracle"
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_rows: list[dict] = []
    resource_rows: list[dict] = []
    theta = math.asin(math.sqrt(exact))
    for method, (a_operator, objective) in methods.items():
        a_probability = objective_probability(a_operator, objective)
        validation_rows.append(
            {
                "method": method,
                "total_qubits": a_operator.num_qubits,
                "exact_random_opportunity_hazard_probability": exact,
                "statevector_probability": a_probability,
                "absolute_error": abs(a_probability - exact),
                "passes_1e-12": abs(a_probability - exact) < 1e-12,
            }
        )
        for power in args.powers:
            circuit = qae_circuit(a_operator, objective, power)
            observed = objective_probability(circuit, objective)
            expected = math.sin((2 * power + 1) * theta) ** 2
            physical = transpile(
                circuit,
                basis_gates=BASIS,
                optimization_level=args.optimization,
                seed_transpiler=20260907,
            )
            operations = {
                str(key): int(value) for key, value in physical.count_ops().items()
            }
            safe_method = method.replace("__", "_")
            qasm_path = output_dir / f"{safe_method}_m{power}.qasm"
            qasm_path.write_text(qasm2_dumps(physical), encoding="utf-8")
            resource_rows.append(
                {
                    "method": method,
                    "grover_power": power,
                    "total_qubits": circuit.num_qubits,
                    "exact_base_probability": exact,
                    "expected_amplified_probability": expected,
                    "statevector_amplified_probability": observed,
                    "absolute_amplification_error": abs(observed - expected),
                    "logical_depth": circuit.depth(),
                    "logical_gate_count": circuit.size(),
                    "transpiled_depth": physical.depth(),
                    "transpiled_gate_count": physical.size(),
                    "transpiled_cx": operations.get("cx", 0),
                    "transpiled_operations": json.dumps(operations),
                    "qasm_path": str(qasm_path.resolve()),
                }
            )

    validation = pd.DataFrame(validation_rows)
    resources = pd.DataFrame(resource_rows)
    validation.to_csv(output_dir / "validation.csv", index=False)
    resources.to_csv(output_dir / "circuit_resources.csv", index=False)
    summary = {
        "poisson_conversion": "annual occurrence probability = 1 - exp(-median annual frequency)",
        "random_opportunity_definition": "one of 92 physical components and one of five leak sizes",
        "physical_opportunity_count": physical_opportunities,
        "random_opportunity_hazard_probability": exact,
        "expected_system_hazardous_opportunities_per_year": expected_system_hazardous_opportunities,
        "at_least_one_system_event_probability_if_independent_poisson": -math.expm1(
            -expected_system_hazardous_opportunities
        ),
        "independence_warning": (
            "The at-least-one value is sensitivity-only. HyRAM+ sums mutually exclusive "
            "fault frequencies under shutdown and does not require this independence claim."
        ),
    }
    (output_dir / "trajectory_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("\nValidation")
    print(validation.to_string(index=False))
    print("\nCircuit resources")
    print(
        resources[[
            "method", "grover_power", "total_qubits",
            "transpiled_depth", "transpiled_gate_count", "transpiled_cx",
            "absolute_amplification_error",
        ]].to_string(index=False)
    )
    print(f"\nresults: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
