"""Build or run PHMSA risk-amplitude circuits with Qiskit.

The default ``build`` mode only exports circuits and resource counts.  The
``ideal``, ``noisy``, and ``ibm`` modes execute jobs and must be selected
explicitly.  This file is intended for the isolated environment documented in
RUN_NEXT.md; it is not imported by the package-free gate checker.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.qasm2 import dumps as qasm2_dumps
from qiskit.synthesis import synth_mcx_1_clean_kg24


def append_mcx(
    qc: QuantumCircuit,
    controls: list[int],
    target: int,
    ancilla: int | None,
    synthesis: str,
) -> None:
    if synthesis == "one-clean" and len(controls) > 2:
        if ancilla is None:
            raise ValueError("one-clean MCX synthesis requires one ancilla qubit")
        qc.compose(
            synth_mcx_1_clean_kg24(len(controls)),
            qubits=[*controls, target, ancilla],
            inplace=True,
        )
    else:
        qc.mcx(controls, target)


def mark_any_risk(
    qc: QuantumCircuit, risk_qubits: int, ancilla: int | None, synthesis: str
) -> None:
    controls = list(range(risk_qubits))
    objective = risk_qubits
    qc.x(controls)
    append_mcx(qc, controls, objective, ancilla, synthesis)
    qc.x(objective)
    qc.x(controls)


def apply_a(
    qc: QuantumCircuit,
    risk_qubits: int,
    rotation: float,
    ancilla: int | None,
    synthesis: str,
) -> None:
    for qubit in range(risk_qubits):
        qc.ry(rotation, qubit)
    mark_any_risk(qc, risk_qubits, ancilla, synthesis)


def apply_a_dagger(
    qc: QuantumCircuit,
    risk_qubits: int,
    rotation: float,
    ancilla: int | None,
    synthesis: str,
) -> None:
    mark_any_risk(qc, risk_qubits, ancilla, synthesis)
    for qubit in reversed(range(risk_qubits)):
        qc.ry(-rotation, qubit)


def apply_zero_phase(
    qc: QuantumCircuit,
    computational_qubits: int,
    ancilla: int | None,
    synthesis: str,
) -> None:
    qubits = list(range(computational_qubits))
    target = qubits[-1]
    controls = qubits[:-1]
    qc.x(qubits)
    qc.h(target)
    append_mcx(qc, controls, target, ancilla, synthesis)
    qc.h(target)
    qc.x(qubits)


def apply_grover(
    qc: QuantumCircuit,
    risk_qubits: int,
    rotation: float,
    ancilla: int | None,
    synthesis: str,
) -> None:
    qc.z(risk_qubits)
    apply_a_dagger(qc, risk_qubits, rotation, ancilla, synthesis)
    apply_zero_phase(qc, risk_qubits + 1, ancilla, synthesis)
    apply_a(qc, risk_qubits, rotation, ancilla, synthesis)


def make_circuit(
    risk_qubits: int,
    rotation: float,
    grover_power: int,
    synthesis: str,
    measure: bool = True,
) -> QuantumCircuit:
    ancilla = risk_qubits + 1 if synthesis == "one-clean" and risk_qubits > 2 else None
    total_qubits = risk_qubits + 1 + (ancilla is not None)
    qc = QuantumCircuit(total_qubits, name=f"risk_k{risk_qubits}_m{grover_power}")
    apply_a(qc, risk_qubits, rotation, ancilla, synthesis)
    for _ in range(grover_power):
        apply_grover(qc, risk_qubits, rotation, ancilla, synthesis)
    if measure:
        meas = ClassicalRegister(1, "meas")
        qc.add_register(meas)
        qc.measure(risk_qubits, meas[0])
    return qc


def mle_amplitude(counts: list[int], shots: int, powers: list[int]) -> float:
    theta = np.linspace(1e-12, math.pi / 2 - 1e-12, 500_001)
    likelihood = np.zeros_like(theta)
    for good_count, power in zip(counts, powers):
        probability = np.sin((2 * power + 1) * theta) ** 2
        probability = np.clip(probability, 1e-15, 1 - 1e-15)
        likelihood += good_count * np.log(probability)
        likelihood += (shots - good_count) * np.log1p(-probability)
    return float(np.sin(theta[int(np.argmax(likelihood))]) ** 2)


def mle_fit_diagnostics(
    estimate: float, counts: list[int], shots: int, powers: list[int]
) -> dict[str, float | str]:
    theta = math.asin(math.sqrt(estimate))
    predicted = np.asarray(
        [math.sin((2 * power + 1) * theta) ** 2 for power in powers]
    )
    observed = np.asarray(counts, dtype=float) / shots
    residuals = observed - predicted
    rmse = float(np.sqrt(np.mean(residuals**2)))
    max_error = float(np.max(np.abs(residuals)))
    # A deliberately conservative screen: three-binomial-standard-error scale
    # with a 5 percentage-point floor. This flags gross ideal-model mismatch;
    # it is not a formal hypothesis test.
    threshold = max(3 / math.sqrt(shots), 0.05)
    return {
        "mle_fit_rmse": rmse,
        "mle_fit_max_error": max_error,
        "mle_fit_threshold": threshold,
        "fit_status": "model_mismatch" if rmse > threshold else "acceptable",
    }


def good_count(counts: dict[str, int]) -> int:
    return int(sum(value for key, value in counts.items() if key.replace(" ", "")[-1] == "1"))


def custom_noise_model(one_qubit_error: float, two_qubit_error: float, readout_error: float):
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    noise = NoiseModel()
    noise.add_all_qubit_quantum_error(
        depolarizing_error(one_qubit_error, 1), ["x", "sx"]
    )
    noise.add_all_qubit_quantum_error(
        depolarizing_error(two_qubit_error, 2), ["cx"]
    )
    noise.add_all_qubit_readout_error(
        ReadoutError([[1 - readout_error, readout_error],
                      [readout_error, 1 - readout_error]])
    )
    return noise


def run_aer(circuits: list[QuantumCircuit], shots: int, noisy: bool, args) -> tuple[list[dict], dict]:
    from qiskit_aer import AerSimulator

    if noisy:
        noise = custom_noise_model(args.p1, args.p2, args.readout_error)
        backend = AerSimulator(method="density_matrix", noise_model=noise)
    else:
        backend = AerSimulator(method="statevector")
    transpiled = transpile(circuits, backend=backend, optimization_level=args.optimization)
    result = backend.run(transpiled, shots=shots, seed_simulator=args.seed).result()
    return [result.get_counts(i) for i in range(len(circuits))], {
        "backend": backend.name,
        "job_id": None,
        "transpiled_resources": [
            {
                "transpiled_depth": circuit.depth(),
                "transpiled_gate_count": circuit.size(),
                "transpiled_operations": json.dumps(dict(circuit.count_ops())),
            }
            for circuit in transpiled
        ],
    }


def prepare_ibm(circuits: list[QuantumCircuit], args):
    from qiskit.transpiler import generate_preset_pass_manager
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService(
        channel=args.ibm_channel,
        plans_preference=[args.ibm_plan],
        region="us-east" if args.ibm_plan == "open" else None,
    )
    if args.ibm_backend:
        backend = service.backend(args.ibm_backend)
    else:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=max(circuit.num_qubits for circuit in circuits),
        )
    manager = generate_preset_pass_manager(
        backend=backend, optimization_level=args.optimization
    )
    isa_circuits = [manager.run(circuit) for circuit in circuits]
    return backend, isa_circuits


def run_ibm(circuits: list[QuantumCircuit], shots: int, args) -> tuple[list[dict], dict]:
    from qiskit_ibm_runtime import SamplerV2

    backend, isa_circuits = prepare_ibm(circuits, args)
    sampler = SamplerV2(mode=backend)
    job = sampler.run(isa_circuits, shots=shots)
    result = job.result()
    counts = [pub_result.data.meas.get_counts() for pub_result in result]
    return counts, {
        "backend": backend.name,
        "job_id": job.job_id(),
        "transpiled_resources": [
            {
                "transpiled_depth": circuit.depth(),
                "transpiled_gate_count": circuit.size(),
                "transpiled_operations": json.dumps(dict(circuit.count_ops())),
            }
            for circuit in isa_circuits
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["build", "transpile", "ideal", "noisy", "ibm-preview", "ibm"],
        default="build",
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--powers", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--unit-miles", type=float, default=1000.0)
    parser.add_argument("--optimization", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--p1", type=float, default=0.001)
    parser.add_argument("--p2", type=float, default=0.01)
    parser.add_argument("--readout-error", type=float, default=0.02)
    parser.add_argument(
        "--mcx-synthesis",
        choices=["noancilla", "one-clean"],
        default="noancilla",
    )
    parser.add_argument("--ibm-backend")
    parser.add_argument("--ibm-channel", default="ibm_quantum_platform")
    parser.add_argument(
        "--ibm-plan",
        choices=["open", "pay-as-you-go"],
        default="open",
    )
    parser.add_argument(
        "--confirm-hardware",
        action="store_true",
        help="Required with --mode ibm to prevent accidental QPU submission",
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()

    root = args.project_root
    risk = json.loads(
        (root / "results" / "classical_risk" / "pooled_risk.json").read_text(
            encoding="utf-8"
        )
    )
    unit_probability = 1 - math.exp(
        -risk["serious_rate_per_1000_mile_year"] * args.unit_miles / 1000
    )
    rotation = 2 * math.asin(math.sqrt(unit_probability))
    circuits: list[QuantumCircuit] = []
    keys: list[tuple[int, int]] = []
    output_dir = root / "results" / "qae_risk" / "qiskit"
    output_dir.mkdir(parents=True, exist_ok=True)

    build_rows = []
    for size in args.sizes:
        for power in args.powers:
            circuit = make_circuit(size, rotation, power, args.mcx_synthesis)
            circuits.append(circuit)
            keys.append((size, power))
            qasm_path = output_dir / (
                f"risk_{args.mcx_synthesis}_k{size}_m{power}.qasm"
            )
            qasm_path.write_text(qasm2_dumps(circuit), encoding="utf-8")
            build_rows.append(
                {
                    "risk_qubits": size,
                    "objective_qubits": 1,
                    "ancilla_qubits": circuit.num_qubits - size - 1,
                    "mcx_synthesis": args.mcx_synthesis,
                    "grover_power": power,
                    "logical_depth": circuit.depth(),
                    "logical_gate_count": circuit.size(),
                    "logical_operations": json.dumps(circuit.count_ops()),
                    "qasm_path": str(qasm_path),
                }
            )
    build_frame = pd.DataFrame(build_rows)
    build_frame.to_csv(output_dir / "circuit_resources_logical.csv", index=False)
    print(build_frame.to_string(index=False))
    if args.mode == "build":
        print(f"build-only output: {output_dir}")
        return

    if args.mode == "transpile":
        from qiskit_aer import AerSimulator

        noise = custom_noise_model(args.p1, args.p2, args.readout_error)
        backend = AerSimulator(method="density_matrix", noise_model=noise)
        transpiled = transpile(
            circuits, backend=backend, optimization_level=args.optimization
        )
        resource_rows = []
        for (size, power), circuit in zip(keys, transpiled):
            resource_rows.append(
                {
                    "risk_qubits": size,
                    "objective_qubits": 1,
                    "ancilla_qubits": circuit.num_qubits - size - 1,
                    "mcx_synthesis": args.mcx_synthesis,
                    "grover_power": power,
                    "transpiled_depth": circuit.depth(),
                    "transpiled_gate_count": circuit.size(),
                    "transpiled_operations": json.dumps(dict(circuit.count_ops())),
                }
            )
        resource_path = output_dir / (
            f"circuit_resources_transpiled_{args.mcx_synthesis}.csv"
        )
        resource_frame = pd.DataFrame(resource_rows)
        resource_frame.to_csv(resource_path, index=False)
        print(resource_frame.to_string(index=False))
        print(f"transpile-only output: {resource_path}")
        return

    if args.mode == "ibm-preview":
        backend, isa_circuits = prepare_ibm(circuits, args)
        preview_rows = []
        for (size, power), circuit in zip(keys, isa_circuits):
            preview_rows.append(
                {
                    "backend": backend.name,
                    "plan_restriction": args.ibm_plan,
                    "risk_qubits": size,
                    "total_qubits": circuit.num_qubits,
                    "grover_power": power,
                    "isa_depth": circuit.depth(),
                    "isa_gate_count": circuit.size(),
                    "isa_operations": json.dumps(dict(circuit.count_ops())),
                }
            )
        preview_path = output_dir / f"ibm_preview_{backend.name}.csv"
        preview = pd.DataFrame(preview_rows)
        preview.to_csv(preview_path, index=False)
        print(preview.to_string(index=False))
        print(f"preview only; no QPU job submitted: {preview_path}")
        return

    if args.mode == "ibm" and not args.confirm_hardware:
        raise ValueError(
            "Hardware submission requires the explicit --confirm-hardware flag"
        )

    if args.mode in {"ideal", "noisy"}:
        all_counts, backend_info = run_aer(
            circuits, args.shots, args.mode == "noisy", args
        )
    else:
        all_counts, backend_info = run_ibm(circuits, args.shots, args)

    result_rows = []
    for size in args.sizes:
        selected = [i for i, key in enumerate(keys) if key[0] == size]
        goods = [good_count(all_counts[i]) for i in selected]
        estimate = mle_amplitude(goods, args.shots, args.powers)
        diagnostics = mle_fit_diagnostics(estimate, goods, args.shots, args.powers)
        exact = 1 - (1 - unit_probability) ** size
        for i, good in zip(selected, goods):
            _, power = keys[i]
            resources = backend_info["transpiled_resources"][i]
            result_rows.append(
                {
                    "mode": args.mode,
                    "backend": backend_info["backend"],
                    "job_id": backend_info["job_id"],
                    "risk_qubits": size,
                    "objective_qubits": 1,
                    "ancilla_qubits": circuits[i].num_qubits - size - 1,
                    "mcx_synthesis": args.mcx_synthesis,
                    "grover_power": power,
                    "shots": args.shots,
                    "one_qubit_error": args.p1 if args.mode == "noisy" else None,
                    "two_qubit_error": args.p2 if args.mode == "noisy" else None,
                    "readout_error": args.readout_error if args.mode == "noisy" else None,
                    "good_counts": good,
                    "observed_amplified_probability": good / args.shots,
                    "exact_probability": exact,
                    "mle_probability": estimate,
                    "absolute_error": abs(estimate - exact),
                    **diagnostics,
                    **resources,
                }
            )
    noise_tag = ""
    if args.mode == "noisy":
        noise_tag = (
            f"_p1-{args.p1:g}_p2-{args.p2:g}_ro-{args.readout_error:g}"
        ).replace(".", "p")
    result_path = output_dir / (
        f"qae_{args.mode}_{args.mcx_synthesis}{noise_tag}_results.csv"
    )
    pd.DataFrame(result_rows).to_csv(result_path, index=False)
    print(f"result: {result_path}")


if __name__ == "__main__":
    main()
