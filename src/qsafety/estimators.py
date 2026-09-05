"""Classical Monte Carlo and maximum-likelihood amplitude estimation."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np

from .core import RiskDistribution


def _validate_schedule(powers: Iterable[int], shots: int) -> tuple[int, ...]:
    schedule = tuple(int(power) for power in powers)
    if not schedule or any(power < 0 for power in schedule):
        raise ValueError("powers must contain one or more non-negative integers")
    if len(set(schedule)) != len(schedule):
        raise ValueError("powers must not contain duplicates")
    if int(shots) != shots or shots < 1:
        raise ValueError("shots must be a positive integer")
    return schedule


@dataclass(frozen=True)
class MLQAEEstimate:
    probability: float
    theta: float
    log_likelihood: float
    observed_probabilities: tuple[float, ...]
    fitted_probabilities: tuple[float, ...]
    fit_rmse: float
    fit_max_error: float
    query_budget: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EstimateSummary:
    method: str
    exact_probability: float
    mean_estimate: float
    bias: float
    mae: float
    rmse: float
    q025: float
    q975: float
    repeats: int
    query_budget: int

    def to_dict(self) -> dict:
        return asdict(self)


def amplified_probabilities(probability: float, powers: Iterable[int]) -> np.ndarray:
    if not 0 <= probability <= 1:
        raise ValueError("probability must lie in [0, 1]")
    schedule = tuple(powers)
    theta = math.asin(math.sqrt(probability))
    return np.asarray([math.sin((2 * power + 1) * theta) ** 2 for power in schedule])


def estimate_amplitude(
    counts: Iterable[int],
    shots: int,
    powers: Iterable[int] = (0, 1, 2),
    *,
    grid_points: int = 500_001,
) -> MLQAEEstimate:
    """Estimate an event probability from objective-qubit success counts.

    This implements the grid maximum-likelihood estimator used in the study.
    Each count corresponds to the same-position Grover power.
    """
    schedule = _validate_schedule(powers, shots)
    observed_counts = np.asarray(tuple(counts), dtype=int)
    if len(observed_counts) != len(schedule):
        raise ValueError("counts and powers must have the same length")
    if np.any((observed_counts < 0) | (observed_counts > shots)):
        raise ValueError("each count must lie between zero and shots")
    if grid_points < 1_001:
        raise ValueError("grid_points must be at least 1001")
    theta_grid = np.linspace(1e-12, math.pi / 2 - 1e-12, int(grid_points))
    log_likelihood = np.zeros_like(theta_grid)
    for count, power in zip(observed_counts, schedule):
        probability = np.sin((2 * power + 1) * theta_grid) ** 2
        probability = np.clip(probability, 1e-15, 1 - 1e-15)
        log_likelihood += count * np.log(probability)
        log_likelihood += (shots - count) * np.log1p(-probability)
    best = int(np.argmax(log_likelihood))
    theta = float(theta_grid[best])
    estimate = float(math.sin(theta) ** 2)
    fitted = amplified_probabilities(estimate, schedule)
    observed = observed_counts.astype(float) / shots
    residual = observed - fitted
    return MLQAEEstimate(
        probability=estimate,
        theta=theta,
        log_likelihood=float(log_likelihood[best]),
        observed_probabilities=tuple(map(float, observed)),
        fitted_probabilities=tuple(map(float, fitted)),
        fit_rmse=float(np.sqrt(np.mean(residual**2))),
        fit_max_error=float(np.max(np.abs(residual))),
        query_budget=int(shots * sum(2 * power + 1 for power in schedule)),
    )


def _summary(method: str, estimates: np.ndarray, exact: float, budget: int) -> EstimateSummary:
    errors = estimates - exact
    return EstimateSummary(
        method=method,
        exact_probability=exact,
        mean_estimate=float(np.mean(estimates)),
        bias=float(np.mean(errors)),
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors**2))),
        q025=float(np.quantile(estimates, 0.025)),
        q975=float(np.quantile(estimates, 0.975)),
        repeats=len(estimates),
        query_budget=budget,
    )


def classical_monte_carlo(
    model: RiskDistribution,
    samples: int,
    *,
    repeats: int = 1_000,
    seed: int | None = None,
) -> EstimateSummary:
    """Repeated Bernoulli Monte Carlo for the model's overall event probability."""
    if samples < 1 or repeats < 1:
        raise ValueError("samples and repeats must be positive")
    rng = np.random.default_rng(seed)
    estimates = rng.binomial(samples, model.probability, repeats) / samples
    return _summary("classical_mc", estimates, model.probability, samples)


def simulate_mlqae(
    model: RiskDistribution,
    powers: Iterable[int] = (0, 1, 2),
    *,
    shots: int = 1_024,
    repeats: int = 1_000,
    seed: int | None = None,
    grid_points: int = 65_537,
) -> EstimateSummary:
    """Simulate ideal ML-QAE measurements and summarize estimation error."""
    schedule = _validate_schedule(powers, shots)
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(seed)
    amplified = amplified_probabilities(model.probability, schedule)
    counts = np.column_stack(
        [rng.binomial(shots, probability, repeats) for probability in amplified]
    )
    theta = np.linspace(1e-10, math.pi / 2 - 1e-10, grid_points)
    candidate = np.asarray([np.sin((2 * power + 1) * theta) ** 2 for power in schedule])
    candidate = np.clip(candidate, 1e-14, 1 - 1e-14)
    estimates = np.empty(repeats)
    for start in range(0, repeats, 128):
        block = counts[start : start + 128]
        likelihood = block @ np.log(candidate)
        likelihood += (shots - block) @ np.log1p(-candidate)
        estimates[start : start + 128] = np.sin(theta[np.argmax(likelihood, axis=1)]) ** 2
    budget = shots * sum(2 * power + 1 for power in schedule)
    return _summary("ideal_mlqae", estimates, model.probability, budget)


def compare_estimators(
    model: RiskDistribution,
    powers: Iterable[int] = (0, 1, 2),
    *,
    shots: int = 1_024,
    repeats: int = 1_000,
    seed: int | None = None,
    grid_points: int = 65_537,
) -> tuple[EstimateSummary, EstimateSummary]:
    """Compare MC and ideal ML-QAE at the same equivalent oracle-query budget."""
    schedule = _validate_schedule(powers, shots)
    budget = shots * sum(2 * power + 1 for power in schedule)
    mc = classical_monte_carlo(model, budget, repeats=repeats, seed=seed)
    qae = simulate_mlqae(
        model,
        schedule,
        shots=shots,
        repeats=repeats,
        seed=None if seed is None else seed + 1,
        grid_points=grid_points,
    )
    return mc, qae
