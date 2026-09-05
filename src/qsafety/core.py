"""Validated probability models used by the classical and quantum estimators."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RiskDistribution:
    """A finite mixture of conditional event probabilities.

    ``weights[j]`` is the probability of scenario or stratum ``j`` and
    ``probabilities[j]`` is the conditional probability of the event of
    interest in that scenario. The overall event probability is their weighted
    mean. Weights are normalized by default so ordinary count columns can also
    be supplied.
    """

    weights: np.ndarray
    probabilities: np.ndarray
    labels: tuple[str, ...] | None = None

    def __init__(
        self,
        weights: Iterable[float],
        probabilities: Iterable[float],
        labels: Iterable[str] | None = None,
        *,
        normalize: bool = True,
    ) -> None:
        weight_array = np.asarray(tuple(weights), dtype=float)
        probability_array = np.asarray(tuple(probabilities), dtype=float)
        if weight_array.ndim != 1 or probability_array.ndim != 1:
            raise ValueError("weights and probabilities must be one-dimensional")
        if len(weight_array) == 0 or len(weight_array) != len(probability_array):
            raise ValueError("weights and probabilities must have the same non-zero length")
        if not np.all(np.isfinite(weight_array)) or not np.all(np.isfinite(probability_array)):
            raise ValueError("weights and probabilities must be finite")
        if np.any(weight_array < 0) or float(weight_array.sum()) <= 0:
            raise ValueError("weights must be non-negative and have a positive sum")
        if np.any((probability_array < 0) | (probability_array > 1)):
            raise ValueError("probabilities must lie in [0, 1]")
        total = float(weight_array.sum())
        if normalize:
            weight_array = weight_array / total
        elif not np.isclose(total, 1.0, atol=1e-12):
            raise ValueError("weights must sum to one when normalize=False")
        label_tuple = tuple(labels) if labels is not None else None
        if label_tuple is not None and len(label_tuple) != len(weight_array):
            raise ValueError("labels must have the same length as weights")
        weight_array.setflags(write=False)
        probability_array.setflags(write=False)
        object.__setattr__(self, "weights", weight_array)
        object.__setattr__(self, "probabilities", probability_array)
        object.__setattr__(self, "labels", label_tuple)

    @property
    def size(self) -> int:
        return len(self.weights)

    @property
    def probability(self) -> float:
        """Return the exact weighted event probability represented by the model."""
        return float(np.dot(self.weights, self.probabilities))

    @property
    def scenario_qubits(self) -> int:
        """Return the required scenario-register size for power-of-two models."""
        if self.size < 2:
            raise ValueError("Qiskit circuit construction requires at least two strata")
        if self.size & (self.size - 1):
            raise ValueError("Qiskit state preparation requires a power-of-two number of strata")
        return self.size.bit_length() - 1

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        weight_column: str = "weight",
        probability_column: str = "probability",
        label_column: str | None = None,
        normalize: bool = True,
    ) -> RiskDistribution:
        """Read a risk distribution from a UTF-8 CSV file."""
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("input CSV has no data rows")
        required = {weight_column, probability_column}
        missing = required.difference(rows[0])
        if missing:
            raise ValueError(f"missing CSV columns: {', '.join(sorted(missing))}")
        weights = [float(row[weight_column]) for row in rows]
        probabilities = [float(row[probability_column]) for row in rows]
        labels = [row[label_column] for row in rows] if label_column else None
        return cls(weights, probabilities, labels, normalize=normalize)

    def as_records(self) -> list[dict[str, float | str | int]]:
        labels = self.labels or tuple(str(index) for index in range(self.size))
        return [
            {
                "stratum": index,
                "label": labels[index],
                "weight": float(self.weights[index]),
                "probability": float(self.probabilities[index]),
            }
            for index in range(self.size)
        ]
