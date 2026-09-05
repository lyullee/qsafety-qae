"""Optional Qiskit circuit construction and Aer execution helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .core import RiskDistribution
from .estimators import MLQAEEstimate, estimate_amplitude


def _qiskit():
    try:
        from qiskit import ClassicalRegister, QuantumCircuit, transpile
        from qiskit.circuit.library import StatePreparation, UCRYGate
    except ImportError as exc:
        raise ImportError(
            "Circuit features require: pip install 'qsafety-qae[qiskit]'"
        ) from exc
    return ClassicalRegister, QuantumCircuit, StatePreparation, UCRYGate, transpile


@dataclass(frozen=True)
class CircuitBundle:
    circuits: tuple[object, ...]
    powers: tuple[int, ...]
    exact_probability: float
    objective_qubit: int


@dataclass(frozen=True)
class AerResult:
    estimate: MLQAEEstimate
    good_counts: tuple[int, ...]
    raw_counts: tuple[dict[str, int], ...]
    circuit_resources: tuple[dict[str, object], ...]


def build_state_preparation(model: RiskDistribution):
    """Build the study's distribution preparation and objective rotation gate."""
    _, QuantumCircuit, StatePreparation, UCRYGate, _ = _qiskit()
    n = model.scenario_qubits
    circuit = QuantumCircuit(n + 1, name=f"A_qsafety_n{n}")
    circuit.append(StatePreparation(np.sqrt(model.weights).astype(complex)), range(n))
    angles = (2 * np.arcsin(np.sqrt(model.probabilities))).tolist()
    circuit.append(UCRYGate(angles), [n, *range(n)])
    return circuit.to_gate(label=f"Aqs{n}"), circuit


def _zero_reflection(circuit, scenario_qubits: int) -> None:
    qubits = list(range(scenario_qubits + 1))
    objective = scenario_qubits
    circuit.x(qubits)
    circuit.h(objective)
    circuit.mcx(list(range(scenario_qubits)), objective)
    circuit.h(objective)
    circuit.x(qubits)


def build_qae_circuits(
    model: RiskDistribution,
    powers: tuple[int, ...] | list[int] = (0, 1, 2),
    *,
    measure: bool = True,
) -> CircuitBundle:
    """Create one ML-QAE circuit for each Grover power."""
    ClassicalRegister, QuantumCircuit, _, _, _ = _qiskit()
    schedule = tuple(int(power) for power in powers)
    if not schedule or any(power < 0 for power in schedule):
        raise ValueError("powers must be non-empty and non-negative")
    n = model.scenario_qubits
    objective = n
    a_gate, _ = build_state_preparation(model)
    circuits = []
    for power in schedule:
        circuit = QuantumCircuit(n + 1, name=f"qsafety_n{n}_m{power}")
        circuit.append(a_gate, range(n + 1))
        for _ in range(power):
            circuit.z(objective)
            circuit.append(a_gate.inverse(), range(n + 1))
            _zero_reflection(circuit, n)
            circuit.append(a_gate, range(n + 1))
        if measure:
            register = ClassicalRegister(1, "objective")
            circuit.add_register(register)
            circuit.measure(objective, register[0])
        circuits.append(circuit)
    return CircuitBundle(tuple(circuits), schedule, model.probability, objective)


def objective_probability(circuit, objective_qubit: int) -> float:
    """Calculate the objective-qubit probability from an unmeasured statevector."""
    try:
        from qiskit.quantum_info import Statevector
    except ImportError as exc:
        raise ImportError("Install qsafety-qae[qiskit]") from exc
    probabilities = Statevector.from_instruction(circuit).probabilities()
    indices = np.arange(len(probabilities))
    return float(probabilities[((indices >> objective_qubit) & 1) == 1].sum())


def validate_bundle(model: RiskDistribution, powers=(0, 1, 2)) -> dict[str, float]:
    """Check state preparation and Grover amplification against analytic values."""
    bundle = build_qae_circuits(model, powers, measure=False)
    theta = math.asin(math.sqrt(model.probability))
    errors = []
    for circuit, power in zip(bundle.circuits, bundle.powers):
        observed = objective_probability(circuit, bundle.objective_qubit)
        expected = math.sin((2 * power + 1) * theta) ** 2
        errors.append(abs(observed - expected))
    return {
        "exact_probability": model.probability,
        "max_grover_formula_error": float(max(errors)),
    }


def _good_count(counts: dict[str, int]) -> int:
    return int(sum(value for key, value in counts.items() if key.replace(" ", "")[-1] == "1"))


def run_aer(
    model: RiskDistribution,
    powers=(0, 1, 2),
    *,
    shots: int = 4_096,
    seed: int | None = None,
    optimization_level: int = 2,
    one_qubit_error: float = 0.0,
    two_qubit_error: float = 0.0,
    readout_error: float = 0.0,
) -> AerResult:
    """Run ideal or depolarizing-noise ML-QAE circuits with Qiskit Aer."""
    _, _, _, _, transpile = _qiskit()
    try:
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
    except ImportError as exc:
        raise ImportError("Aer execution requires: pip install 'qsafety-qae[qiskit]'") from exc
    for value in (one_qubit_error, two_qubit_error, readout_error):
        if not 0 <= value < 1:
            raise ValueError("error probabilities must lie in [0, 1)")
    noise = None
    if any((one_qubit_error, two_qubit_error, readout_error)):
        noise = NoiseModel()
        if one_qubit_error:
            noise.add_all_qubit_quantum_error(depolarizing_error(one_qubit_error, 1), ["x", "sx"])
        if two_qubit_error:
            noise.add_all_qubit_quantum_error(depolarizing_error(two_qubit_error, 2), ["cx"])
        if readout_error:
            noise.add_all_qubit_readout_error(
                ReadoutError([[1 - readout_error, readout_error], [readout_error, 1 - readout_error]])
            )
    backend = AerSimulator(method="density_matrix" if noise else "statevector", noise_model=noise)
    bundle = build_qae_circuits(model, powers, measure=True)
    physical = transpile(bundle.circuits, backend=backend, optimization_level=optimization_level)
    result = backend.run(physical, shots=shots, seed_simulator=seed).result()
    raw = tuple(result.get_counts(index) for index in range(len(physical)))
    good = tuple(_good_count(counts) for counts in raw)
    estimate = estimate_amplitude(good, shots, bundle.powers)
    resources = tuple(
        {
            "power": power,
            "depth": circuit.depth(),
            "gate_count": circuit.size(),
            "operations": {str(key): int(value) for key, value in circuit.count_ops().items()},
        }
        for power, circuit in zip(bundle.powers, physical)
    )
    return AerResult(estimate, good, raw, resources)

