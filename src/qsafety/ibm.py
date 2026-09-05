"""Opt-in IBM Quantum execution helpers with a mandatory confirmation flag."""

from __future__ import annotations

from dataclasses import dataclass

from .circuits import CircuitBundle, build_qae_circuits
from .core import RiskDistribution
from .estimators import MLQAEEstimate, estimate_amplitude


def _runtime():
    try:
        from qiskit.transpiler import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as exc:
        raise ImportError("IBM execution requires: pip install 'qsafety-qae[ibm]'") from exc
    return generate_preset_pass_manager, QiskitRuntimeService, SamplerV2


@dataclass(frozen=True)
class IBMPreview:
    backend: str
    circuits: tuple[object, ...]
    powers: tuple[int, ...]
    shots: int
    resources: tuple[dict[str, int], ...]

    @property
    def executions(self) -> int:
        return len(self.circuits) * self.shots


@dataclass(frozen=True)
class IBMResult:
    backend: str
    job_id: str
    good_counts: tuple[int, ...]
    estimate: MLQAEEstimate


def _service(open_plan_only: bool = True):
    _, Service, _ = _runtime()
    kwargs = {"channel": "ibm_quantum_platform"}
    if open_plan_only:
        kwargs.update(plans_preference=["open"], region="us-east")
    return Service(**kwargs)


def preview_ibm(
    model: RiskDistribution,
    powers=(0, 1, 2),
    *,
    shots: int = 512,
    backend_name: str | None = None,
    optimization_level: int = 3,
    open_plan_only: bool = True,
) -> IBMPreview:
    """Transpile for an IBM backend without submitting a workload."""
    manager_factory, _, _ = _runtime()
    service = _service(open_plan_only)
    bundle: CircuitBundle = build_qae_circuits(model, powers, measure=True)
    backend = (
        service.backend(backend_name)
        if backend_name
        else service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=max(circuit.num_qubits for circuit in bundle.circuits),
        )
    )
    manager = manager_factory(backend=backend, optimization_level=optimization_level)
    isa = tuple(manager.run(circuit) for circuit in bundle.circuits)
    resources = tuple(
        {
            "power": power,
            "depth": circuit.depth(),
            "gate_count": circuit.size(),
            "two_qubit_gates": int(
                sum(circuit.count_ops().get(gate, 0) for gate in ("cx", "cz", "ecr", "rzz", "iswap"))
            ),
        }
        for power, circuit in zip(bundle.powers, isa)
    )
    return IBMPreview(backend.name, isa, bundle.powers, shots, resources)


def run_ibm(preview: IBMPreview, *, confirm: bool = False) -> IBMResult:
    """Submit a previously reviewed preview. ``confirm=True`` is mandatory."""
    if not confirm:
        raise PermissionError("QPU submission blocked; pass confirm=True after reviewing the preview")
    _, _, SamplerV2 = _runtime()
    service = _service(True)
    backend = service.backend(preview.backend)
    job = SamplerV2(mode=backend).run(preview.circuits, shots=preview.shots)
    result = job.result()
    raw = tuple(publication.data.objective.get_counts() for publication in result)
    good = tuple(int(counts.get("1", 0)) for counts in raw)
    estimate = estimate_amplitude(good, preview.shots, preview.powers)
    return IBMResult(backend.name, job.job_id(), good, estimate)

