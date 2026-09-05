import numpy as np
import pytest

from qsafety import RiskDistribution


def test_normalizes_weights_and_computes_probability():
    model = RiskDistribution([1, 3], [0.1, 0.3])
    assert np.allclose(model.weights, [0.25, 0.75])
    assert model.probability == pytest.approx(0.25)


@pytest.mark.parametrize(
    "weights, probabilities",
    [([], []), ([1], []), ([-1, 2], [0.1, 0.2]), ([0, 0], [0.1, 0.2]), ([1], [1.1])],
)
def test_rejects_invalid_models(weights, probabilities):
    with pytest.raises(ValueError):
        RiskDistribution(weights, probabilities)


def test_qubit_count_requires_power_of_two():
    assert RiskDistribution([1, 1, 1, 1], [0.1] * 4).scenario_qubits == 2
    with pytest.raises(ValueError):
        _ = RiskDistribution([1, 1, 1], [0.1] * 3).scenario_qubits
    with pytest.raises(ValueError, match="at least two strata"):
        _ = RiskDistribution([1], [0.1]).scenario_qubits


def test_csv_loader(tmp_path):
    path = tmp_path / "risk.csv"
    path.write_text("name,count,risk\na,2,0.1\nb,1,0.4\n", encoding="utf-8")
    model = RiskDistribution.from_csv(
        path, weight_column="count", probability_column="risk", label_column="name"
    )
    assert model.labels == ("a", "b")
    assert model.probability == pytest.approx(0.2)
