"""Build a compressed HyRAM+ uncertainty oracle for the mean hazard amplitude.

The full stochastic model has 92 physical components x 5 leak sizes (460
log-frequency variables).  For the mean random-opportunity amplitude, an
independent factorized model can be represented by three registers: component
class, leak size, and a discretized standard-normal latent variable.  This is
not a claim that the 460-variable joint distribution has been loaded into a
quantum state; it is a controlled compression that preserves the expectation
under the stated independence model and is directly checked against a
classical Gauss--Hermite quadrature calculation.
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
from qiskit.quantum_info import Statevector


BASIS = ["rz", "sx", "x", "cx"]


def ry_angles(probabilities: np.ndarray) -> list[float]:
    probabilities = np.clip(probabilities.astype(float), 0.0, 1.0)
    return (2.0 * np.arcsin(np.sqrt(probabilities))).tolist()


def normal_quadrature(points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return standard-normal Gauss--Hermite nodes and normalized weights."""
    if points < 2 or points & (points - 1):
        raise ValueError("latent points must be a power of two and >= 2")
    nodes, weights = np.polynomial.hermite.hermgauss(points)
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


def load_parameters(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = (
        root
        / "results"
        / "hyram"
        / "system_scenario_inventory"
        / "component_leak_scenarios.csv"
    )
    frame = pd.read_csv(path)
    components = list(dict.fromkeys(frame["component"].tolist()))
    leak_sizes = sorted(frame["leak_size_percent"].unique().tolist())
    if len(components) != 8 or len(leak_sizes) != 5:
        raise ValueError("Expected eight component classes and five leak sizes")
    counts = np.zeros(8)
    mu = np.zeros((8, 5))
    sigma = np.zeros((8, 5))
    ignition = np.zeros(5)
    for ci, component in enumerate(components):
        subset = frame.loc[frame["component"].eq(component)]
        counts[ci] = float(subset["component_count"].iloc[0])
        for li, leak_size in enumerate(leak_sizes):
            row = subset.loc[np.isclose(subset["leak_size_percent"], leak_size)].iloc[0]
            mu[ci, li] = float(row["mu_log_frequency"])
            sigma[ci, li] = float(row["sigma_log_frequency"])
            ignition[li] = float(
                row["immediate_ignition_probability_given_release"]
                + row["delayed_ignition_probability_given_release"]
            )
    return counts / counts.sum(), mu, sigma, ignition, counts


def payoff_tensor(
    component_weights: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    ignition: np.ndarray,
    latent_z: np.ndarray,
    isolation_failure: float,
) -> tuple[np.ndarray, float]:
    """Return 8 x 5 x L payoff tensor and its classical expectation."""
    frequency = np.exp(mu[:, :, None] + sigma[:, :, None] * latent_z[None, None, :])
    leak_probability = -np.expm1(-frequency)
    payoff = leak_probability * isolation_failure * ignition[None, :, None]
    latent_weights = normal_quadrature(len(latent_z))[1]
    expectation = float(
        np.sum(
            component_weights[:, None, None]
            * (1.0 / 5.0)
            * latent_weights[None, None, :]
            * payoff
        )
    )
    return payoff, expectation


def compressed_oracle(
    component_weights: np.ndarray,
    payoff: np.ndarray,
    latent_weights: np.ndarray,
) -> tuple[QuantumCircuit, int, int]:
    """Prepare factorized class/size/latent registers and apply payoff rotation."""
    n_class = 3
    n_size = 3
    n_latent = int(math.log2(len(latent_weights)))
    class_q = list(range(0, n_class))
    size_q = list(range(n_class, n_class + n_size))
    latent_q = list(range(n_class + n_size, n_class + n_size + n_latent))
    data_q = class_q + size_q + latent_q
    objective = len(data_q)
    circuit = QuantumCircuit(objective + 1, name="A_hyram_compressed")
    circuit.append(StatePreparation(np.sqrt(component_weights).astype(complex)), class_q)
    size_weights = np.zeros(2**n_size)
    size_weights[:5] = 1.0 / 5.0
    circuit.append(StatePreparation(np.sqrt(size_weights).astype(complex)), size_q)
    circuit.append(StatePreparation(np.sqrt(latent_weights).astype(complex)), latent_q)
    flat = np.zeros(2 ** len(data_q))
    for li in range(5):
        for ci in range(8):
            for zi in range(len(latent_weights)):
                index = ci + (2**n_class) * li + (2**n_class) * (2**n_size) * zi
                flat[index] = payoff[ci, li, zi]
    circuit.append(UCRYGate(ry_angles(flat)), [objective, *data_q])
    return circuit, objective, objective + 1


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
    parser.add_argument("--latent-points", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--powers", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--optimization", type=int, choices=[0, 1, 2, 3], default=2)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root
    component_weights, mu, sigma, ignition, counts = load_parameters(root)
    isolation_failure = 0.1
    output_dir = root / "results" / "hyram" / "uncertainty_compressed_oracle"
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_rows: list[dict] = []
    resource_rows: list[dict] = []
    convergence_rows: list[dict] = []
    for points in args.latent_points:
        z, latent_weights = normal_quadrature(points)
        payoff, classical = payoff_tensor(
            component_weights, mu, sigma, ignition, z, isolation_failure
        )
        oracle, objective, total_qubits = compressed_oracle(
            component_weights, payoff, latent_weights
        )
        quantum = objective_probability(oracle, objective)
        validation_rows.append(
            {
                "latent_points": points,
                "class_register_qubits": 3,
                "leak_size_register_qubits": 3,
                "latent_register_qubits": int(math.log2(points)),
                "total_qubits": total_qubits,
                "classical_quadrature_probability": classical,
                "statevector_probability": quantum,
                "absolute_error": abs(quantum - classical),
                "passes_1e-12": abs(quantum - classical) < 1e-12,
            }
        )
        convergence_rows.append(
            {
                "latent_points": points,
                "classical_quadrature_probability": classical,
                "nominal_median_parameter_amplitude": 4.600343517804337e-07,
            }
        )
        for power in args.powers:
            circuit = qae_circuit(oracle, objective, power)
            observed = objective_probability(circuit, objective)
            theta = math.asin(math.sqrt(classical))
            expected = math.sin((2 * power + 1) * theta) ** 2
            physical = transpile(
                circuit,
                basis_gates=BASIS,
                optimization_level=args.optimization,
                seed_transpiler=20260922,
            )
            resource_rows.append(
                {
                    "latent_points": points,
                    "grover_power": power,
                    "total_qubits": circuit.num_qubits,
                    "statevector_amplified_probability": observed,
                    "expected_amplified_probability": expected,
                    "amplified_absolute_error": abs(observed - expected),
                    "depth_transpiled": physical.depth(),
                    "size_transpiled": physical.size(),
                    "cx_count": int(physical.count_ops().get("cx", 0)),
                    "rz_count": int(physical.count_ops().get("rz", 0)),
                    "sx_count": int(physical.count_ops().get("sx", 0)),
                    "x_count": int(physical.count_ops().get("x", 0)),
                }
            )

    validation = pd.DataFrame(validation_rows)
    resources = pd.DataFrame(resource_rows)
    convergence = pd.DataFrame(convergence_rows)
    mc_summary_path = (
        root
        / "results"
        / "hyram"
        / "leak_frequency_uncertainty"
        / "uncertainty_summary_full_s5_n200000.csv"
    )
    mc_reference = float(
        pd.read_csv(mc_summary_path)
        .loc[lambda frame: frame["dependence_model"].eq("independent_physical_component_size"), "mean_amplitude"]
        .iloc[0]
    )
    eight_point = float(
        validation.loc[validation["latent_points"].eq(8), "classical_quadrature_probability"].iloc[0]
    )
    validation.to_csv(output_dir / "validation.csv", index=False)
    resources.to_csv(output_dir / "circuit_resources.csv", index=False)
    convergence.to_csv(output_dir / "quadrature_convergence.csv", index=False)
    summary = {
        "model": "HyRAM+ independent physical-component x leak-size lognormal uncertainty",
        "physical_components": int(counts.sum()),
        "full_uncertainty_variables": int(counts.sum() * 5),
        "compression_registers": {
            "component_class": "3 qubits; 8 classes weighted by physical component count",
            "leak_size": "3 qubits; 5 leak sizes with uniform weights",
            "latent": "ceil(log2(L)) qubits; L-point Gauss-Hermite standard-normal quadrature",
        },
        "isolation_failure_probability": isolation_failure,
        "stored_independent_model_mc_mean": mc_reference,
        "eight_point_quadrature_relative_difference_to_stored_mc_mean": abs(eight_point - mc_reference) / mc_reference,
        "passes_all_statevector_checks": bool(validation["passes_1e-12"].all()),
        "validation_file": "validation.csv",
        "resource_file": "circuit_resources.csv",
        "quadrature_file": "quadrature_convergence.csv",
        "interpretation": "This is an expectation-preserving compressed oracle; it does not claim coherent preparation of the full 460-variable joint distribution.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
