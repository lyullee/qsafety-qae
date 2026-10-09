"""Construct and validate the finite-ensemble HELPR QROM oracle.

This is deliberately an empirical 6,000-entry oracle, not the population
oracle used for the ideal ML-QAE crossover.  It makes the previously logical
QROM proxy concrete: a uniform state over the stored sample indices followed
by sparse multi-controlled-X membership marking.  The distinction from the
population oracle is recorded in the output metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector


def objective_probability(circuit: QuantumCircuit, objective: int) -> float:
    probabilities = Statevector.from_instruction(circuit).probabilities()
    indices = np.arange(len(probabilities))
    return float(probabilities[((indices >> objective) & 1) == 1].sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission-cycles", type=int, default=30_000)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root
    pilot_dir = root / "results" / "helpr" / "candidate_pilot"
    frames = []
    for seed in [32001, 32002, 32003, 32004, 32005, 32006]:
        frames.append(
            pd.read_csv(
                pilot_dir / f"helpr_pilot_samples_api_n1000_seed{seed}_smys72_conditioned.csv"
            )
        )
    empirical = pd.concat(frames, ignore_index=True)
    stored_scenarios = len(empirical)
    index_qubits = math.ceil(math.log2(stored_scenarios))
    event_indices = empirical.index[
        empirical["cycles_to_fad_line"].le(args.mission_cycles)
    ].to_list()
    objective = index_qubits
    circuit = QuantumCircuit(objective + 1, name="A_empirical_helpr_qrom")
    amplitudes = np.zeros(2**index_qubits, dtype=complex)
    amplitudes[:stored_scenarios] = 1.0 / math.sqrt(stored_scenarios)
    circuit.append(StatePreparation(amplitudes), list(range(index_qubits)))
    for index in event_indices:
        zero_controls = [q for q in range(index_qubits) if not ((index >> q) & 1)]
        for q in zero_controls:
            circuit.x(q)
        circuit.mcx(list(range(index_qubits)), objective)
        for q in zero_controls:
            circuit.x(q)

    observed = objective_probability(circuit, objective)
    expected = len(event_indices) / stored_scenarios
    output_dir = root / "results" / "helpr" / "empirical_qrom_oracle"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "oracle_type": "finite empirical QROM",
        "stored_scenarios": stored_scenarios,
        "event_flags": len(event_indices),
        "mission_cycles": args.mission_cycles,
        "index_qubits": index_qubits,
        "objective_qubits": 1,
        "total_qubits": circuit.num_qubits,
        "logical_mcx_count": len(event_indices),
        "logical_zero_control_x_count": int(
            2
            * sum(index_qubits - int(index).bit_count() for index in event_indices)
        ),
        "expected_empirical_probability": expected,
        "statevector_probability": observed,
        "absolute_error": abs(observed - expected),
        "passes_1e-12": abs(observed - expected) < 1e-12,
        "population_probability_used_in_mlqae_boundary": 0.007970272865693958,
        "population_oracle_equivalence": False,
        "excluded_costs": [
            "continuous input distribution preparation",
            "reversible HELPR fracture-mechanics evaluation",
            "native decomposition and fault-tolerant error correction",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
