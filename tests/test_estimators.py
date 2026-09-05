import math

import pytest

from qsafety import RiskDistribution, compare_estimators, estimate_amplitude
from qsafety.estimators import amplified_probabilities


def test_mle_recovers_deterministic_counts():
    exact = 0.08
    powers = [0, 1, 2]
    shots = 100_000
    counts = [round(shots * value) for value in amplified_probabilities(exact, powers)]
    result = estimate_amplitude(counts, shots, powers, grid_points=100_001)
    assert result.probability == pytest.approx(exact, abs=2e-4)
    assert result.query_budget == shots * 9
    assert result.fit_rmse < 2e-4


def test_equal_query_budget_comparison():
    model = RiskDistribution([0.5, 0.5], [0.01, 0.09])
    mc, qae = compare_estimators(
        model, powers=[0, 1, 2], shots=64, repeats=50, seed=7, grid_points=4097
    )
    assert mc.query_budget == qae.query_budget == 576
    assert mc.exact_probability == qae.exact_probability == pytest.approx(0.05)
    assert all(math.isfinite(value) for value in [mc.rmse, qae.rmse])


def test_bad_mle_inputs_are_rejected():
    with pytest.raises(ValueError):
        estimate_amplitude([1, 2], 10, [0])
    with pytest.raises(ValueError):
        estimate_amplitude([11], 10, [0])

