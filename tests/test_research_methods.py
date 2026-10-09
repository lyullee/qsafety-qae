"""Synthetic tests that do not load third-party data or connect to a QPU."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")
pytest.importorskip("qiskit")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from helpr_empirical_qrom_mlqae import (
    build_qrom_from_events,
    objective_probability,
    qae_circuit,
)
from helpr_mlqae_importance_boundary import simulate_cell
from hyram_leak_frequency_uncertainty import sample_amplitudes
from hyram_rare_trajectory_qae_scaling import bounded_batched_mle
from hyram_strong_rare_event_baselines import (
    optimized_product_allocation,
    product_variance,
)
from hyram_uncertainty_compressed_oracle import normal_quadrature
from research_schedule import schedule_for_ratio


@pytest.mark.parametrize("ratio", [1.3, 1.5, 1.8, 2.0])
@pytest.mark.parametrize("budget", [1, 63, 64, 512, 4096, 32768])
def test_schedule_accounts_for_queries(ratio, budget):
    powers, shots, actual = schedule_for_ratio(ratio, budget)
    assert powers[0] == 0
    assert powers == sorted(set(powers))
    assert actual == shots * sum(2 * power + 1 for power in powers)
    assert 0 < actual <= budget


@pytest.mark.parametrize("ratio,budget", [(1.0, 64), (math.inf, 64), (1.5, 0), (1.5, True)])
def test_invalid_schedule_inputs_are_rejected(ratio, budget):
    with pytest.raises(ValueError):
        schedule_for_ratio(ratio, budget)


def test_bounded_mle_recovers_synthetic_counts():
    exact = 0.003
    powers = [0, 1, 2]
    shots = 1_000_000
    theta = math.asin(math.sqrt(exact))
    counts = np.array([[round(shots * math.sin((2 * m + 1) * theta) ** 2) for m in powers]])
    estimate = bounded_batched_mle(counts, shots, powers, 0.02, 8193)[0]
    assert estimate == pytest.approx(exact, abs=3e-6)


def test_classical_product_allocation_meets_target():
    means = np.array([0.1, 0.2, 0.3])
    variances = means * (1 - means)
    counts, rmse = optimized_product_allocation(means, variances, 0.001)
    assert np.all(counts >= 1)
    assert rmse <= 0.001
    assert rmse**2 == pytest.approx(product_variance(means, variances, counts))


def test_zero_lognormal_spread_has_exact_constant_amplitude():
    frequency = 0.01
    for model in (
        "independent_physical_component_size", "shared_component_class_size",
        "shared_within_component_class", "shared_by_leak_size", "fully_rank_correlated",
    ):
        values = sample_amplitudes(
            np.random.default_rng(4), 20, 7, model,
            np.full((2, 2), math.log(frequency)), np.zeros((2, 2)),
            np.array([0, 0, 1]), np.array([0.2, 0.4]), 0.1,
        )
        assert np.allclose(values, -math.expm1(-frequency) * 0.1 * 0.3)


def test_normal_quadrature_preserves_first_two_moments():
    nodes, weights = normal_quadrature(8)
    assert weights.sum() == pytest.approx(1.0)
    assert np.dot(nodes, weights) == pytest.approx(0.0, abs=1e-14)
    assert np.dot(nodes**2, weights) == pytest.approx(1.0)


@pytest.mark.parametrize("events", [[0] * 7, [1] * 7, [0, 1, 0, 0, 1, 0, 0]])
def test_finite_qrom_and_grover_use_only_caller_events(events):
    circuit, objective, rows, successes = build_qrom_from_events(np.array(events))
    exact = successes / rows
    assert rows == 7
    assert circuit.num_qubits == 4
    for power in (0, 1, 2):
        amplified = qae_circuit(circuit, objective, power)
        expected = math.sin((2 * power + 1) * math.asin(math.sqrt(exact))) ** 2
        assert objective_probability(amplified, objective) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("events", [[], [1], [0, 2], [[0, 1]], [0, math.nan]])
def test_invalid_qrom_events_are_rejected(events):
    with pytest.raises(ValueError):
        build_qrom_from_events(np.array(events))


def test_ideal_comparison_is_reproducible_on_synthetic_probability():
    first, first_values = simulate_cell(0.003, 1.5, 512, 12, 1025, 0.02, np.random.default_rng(9))
    second, second_values = simulate_cell(0.003, 1.5, 512, 12, 1025, 0.02, np.random.default_rng(9))
    assert first == second
    assert np.array_equal(first_values, second_values)
    assert first["actual_query_budget"] <= 512
