"""Run ideal statevector ML-QAE on the finite HELPR empirical QROM.

The target is deliberately the executable finite ensemble (42 events among
6000 stored samples), not the continuous population probability used by the
earlier ideal crossover.  This closes the finite-QROM loop: state preparation,
membership oracle, Grover amplification, sampled objective counts, and the
same grid maximum-likelihood estimator used elsewhere in the study.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, grover_operator
from qiskit.quantum_info import Statevector

from qsafety.estimators import estimate_amplitude


def load_empirical_frames(root: Path) -> pd.DataFrame:
    pilot_dir = root / "results" / "helpr" / "candidate_pilot"
    frames = [
        pd.read_csv(
            pilot_dir / f"helpr_pilot_samples_api_n1000_seed{seed}_smys72_conditioned.csv"
        )
        for seed in [32001, 32002, 32003, 32004, 32005, 32006]
    ]
    return pd.concat(frames, ignore_index=True)


def build_empirical_qrom(root: Path, mission_cycles: int) -> tuple[QuantumCircuit, int, int, int]:
    empirical = load_empirical_frames(root)
    events = empirical["cycles_to_fad_line"].le(mission_cycles).to_numpy()
    return build_qrom_from_events(events)


def build_qrom_from_events(events: np.ndarray) -> tuple[QuantumCircuit, int, int, int]:
    """Build a finite-table oracle from caller-supplied binary event flags.

    No HELPR implementation or standards table is required. Padding addresses
    receive zero amplitude, including when the table length is not a power of two.
    """
    flags = np.asarray(events)
    if flags.ndim != 1 or flags.size < 2 or not np.isin(flags, [0, 1]).all():
        raise ValueError("events must be a one-dimensional binary array of length >= 2")
    stored_scenarios = len(flags)
    index_qubits = math.ceil(math.log2(stored_scenarios))
    event_indices = np.flatnonzero(flags).tolist()
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
    return circuit, objective, stored_scenarios, len(event_indices)


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
    parser.add_argument("--mission-cycles", type=int, default=30_000)
    parser.add_argument("--powers", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--grid-points", type=int, default=65537)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if args.shots < 1 or args.repeats < 1:
        raise ValueError("shots and repeats must be positive")
    if len(set(args.powers)) != len(args.powers) or any(power < 0 for power in args.powers):
        raise ValueError("powers must be distinct non-negative integers")

    root = args.project_root
    a_operator, objective, stored_scenarios, event_flags = build_empirical_qrom(
        root, args.mission_cycles
    )
    exact = event_flags / stored_scenarios
    circuits = [qae_circuit(a_operator, objective, power) for power in args.powers]
    amplified = []
    resource_rows = []
    validation_rows = []
    for circuit, power in zip(circuits, args.powers):
        observed = objective_probability(circuit, objective)
        expected = math.sin((2 * power + 1) * math.asin(math.sqrt(exact))) ** 2
        amplified.append(observed)
        validation_rows.append(
            {
                "grover_power": power,
                "expected_amplified_probability": expected,
                "statevector_amplified_probability": observed,
                "absolute_error": abs(observed - expected),
            }
        )
        resource_rows.append(
            {
                "grover_power": power,
                "total_qubits": circuit.num_qubits,
                "logical_depth": circuit.depth(),
                "logical_size": circuit.size(),
                "logical_operations": {
                    str(key): int(value) for key, value in circuit.count_ops().items()
                },
            }
        )

    rng = np.random.default_rng(args.seed)
    counts = [int(rng.binomial(args.shots, probability)) for probability in amplified]
    estimate = estimate_amplitude(
        counts, args.shots, args.powers, grid_points=args.grid_points
    )
    repeat_rows = []
    for repeat in range(args.repeats):
        repeat_counts = [int(rng.binomial(args.shots, probability)) for probability in amplified]
        repeat_estimate = estimate_amplitude(
            repeat_counts, args.shots, args.powers, grid_points=args.grid_points
        )
        repeat_rows.append(
            {
                "repeat": repeat,
                "estimate": repeat_estimate.probability,
                "absolute_error": abs(repeat_estimate.probability - exact),
                "fit_rmse": repeat_estimate.fit_rmse,
                "query_budget": repeat_estimate.query_budget,
            }
        )
    repeat_frame = pd.DataFrame(repeat_rows)
    validation = pd.DataFrame(validation_rows)
    resources = pd.DataFrame(resource_rows)
    output_dir = root / "results" / "helpr" / "empirical_qrom_mlqae"
    output_dir.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output_dir / "amplification_validation.csv", index=False)
    resources.to_json(output_dir / "circuit_resources.json", orient="records", indent=2)
    repeat_frame.to_csv(output_dir / "repeat_estimates.csv", index=False)
    summary = {
        "oracle_type": "finite empirical HELPR QROM with ideal statevector ML-QAE",
        "mission_cycles": args.mission_cycles,
        "stored_scenarios": stored_scenarios,
        "event_flags": event_flags,
        "total_qubits": a_operator.num_qubits,
        "population_oracle_equivalence": False,
        "exact_finite_ensemble_probability": exact,
        "powers": args.powers,
        "shots_per_power": args.shots,
        "query_budget": estimate.query_budget,
        "seed": args.seed,
        "single_run_counts": counts,
        "single_run_estimate": estimate.to_dict(),
        "statevector_max_amplification_error": float(validation["absolute_error"].max()),
        "repeat_count": args.repeats,
        "repeat_mean_estimate": float(repeat_frame["estimate"].mean()),
        "repeat_rmse": float(np.sqrt(np.mean(np.square(repeat_frame["estimate"] - exact)))),
        "repeat_mae": float(repeat_frame["absolute_error"].mean()),
        "repeat_q025": float(repeat_frame["estimate"].quantile(0.025)),
        "repeat_q975": float(repeat_frame["estimate"].quantile(0.975)),
        "state_preparation_and_native_decomposition_included": False,
        "interpretation": "End-to-end ideal ML-QAE is validated only for the finite empirical QROM target p=42/6000; it does not implement the continuous HELPR population oracle used for the 0.0079703 crossover.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("\nAmplification validation")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
