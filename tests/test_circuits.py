import pytest

pytest.importorskip("qiskit")

from qsafety import RiskDistribution
from qsafety.circuits import build_qae_circuits, validate_bundle


def test_qiskit_bundle_matches_formula():
    model = RiskDistribution([1, 1, 1, 1], [0.01, 0.02, 0.03, 0.08])
    validation = validate_bundle(model, [0, 1, 2])
    assert validation["max_grover_formula_error"] < 1e-12


def test_measured_bundle_shape():
    model = RiskDistribution([1, 1, 1, 1], [0.01, 0.02, 0.03, 0.08])
    bundle = build_qae_circuits(model, [0, 1])
    assert len(bundle.circuits) == 2
    assert all(circuit.num_qubits == 3 for circuit in bundle.circuits)
    assert all(circuit.num_clbits == 1 for circuit in bundle.circuits)

