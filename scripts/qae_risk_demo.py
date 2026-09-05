"""GPU-capable statevector prototype for PHMSA risk amplitude estimation.

The model uses K independent 1,000 mile-year risk units.  Each unit is a
Bernoulli event with probability derived from the pooled PHMSA Serious Incident
rate.  The good subspace contains every state except the all-zero state, so its
amplitude is the probability of at least one Serious Incident.

This is an ideal statevector and shot-noise experiment, not a hardware runtime
claim.  Quick mode is intentionally small.  Full mode should be run by the user.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False


def select_backend(name: str):
    if name == "cpu":
        return np, "cpu"
    if name == "gpu":
        if not CUPY_AVAILABLE:
            raise RuntimeError("GPU requested, but CuPy is not importable")
        # Force initialization here so a broken CUDA runtime fails early.
        cp.cuda.Device(0).use()
        cp.cuda.runtime.memGetInfo()
        return cp, "gpu"
    if CUPY_AVAILABLE:
        try:
            cp.cuda.Device(0).use()
            cp.cuda.runtime.memGetInfo()
            return cp, "gpu"
        except Exception:
            pass
    return np, "cpu"


def backend_info(xp, backend_name: str) -> dict:
    if backend_name == "cpu":
        return {"backend": "cpu", "library": "numpy"}
    props = cp.cuda.runtime.getDeviceProperties(0)
    free, total = cp.cuda.runtime.memGetInfo()
    raw_name = props["name"]
    device_name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
    return {
        "backend": "gpu",
        "library": "cupy",
        "device": device_name,
        "vram_total_gib": total / 2**30,
        "vram_free_gib": free / 2**30,
    }


def available_bytes(xp, backend_name: str) -> int:
    if backend_name == "gpu":
        free, _ = cp.cuda.runtime.memGetInfo()
        return int(free)
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return 8 * 2**30


def product_bernoulli_state(qubits: int, unit_probability: float, xp):
    one_qubit = xp.asarray(
        [math.sqrt(1 - unit_probability), math.sqrt(unit_probability)],
        dtype=xp.complex128,
    )
    state = xp.asarray([1.0 + 0.0j], dtype=xp.complex128)
    for _ in range(qubits):
        state = xp.kron(state, one_qubit)
    return state


def good_probability(state, xp) -> float:
    # All basis states except |00...0> contain at least one Serious Incident.
    value = xp.sum(xp.abs(state[1:]) ** 2)
    return float(value.item() if hasattr(value, "item") else value)


def grover_step(state, prepared_state, xp):
    # Q = (2|psi><psi| - I) S_good.  S_good changes the phase of all good states.
    marked = state.copy()
    marked[1:] *= -1
    overlap = xp.vdot(prepared_state, marked)
    return 2 * prepared_state * overlap - marked


def amplified_probabilities(
    qubits: int,
    unit_probability: float,
    schedule: list[int],
    xp,
    available_memory: int,
) -> tuple[list[float], float]:
    estimated_bytes = 3 * (2**qubits) * np.dtype(np.complex128).itemsize
    if estimated_bytes > 0.70 * available_memory:
        raise MemoryError(
            f"{qubits} qubits need about {estimated_bytes / 2**30:.2f} GiB "
            "for three state buffers, above the 70% safety limit"
        )

    prepared = product_bernoulli_state(qubits, unit_probability, xp)
    probabilities = []
    max_formula_error = 0.0
    theta = math.asin(math.sqrt(1 - (1 - unit_probability) ** qubits))
    for power in schedule:
        state = prepared.copy()
        for _ in range(power):
            state = grover_step(state, prepared, xp)
        measured = good_probability(state, xp)
        theoretical = math.sin((2 * power + 1) * theta) ** 2
        probabilities.append(measured)
        max_formula_error = max(max_formula_error, abs(measured - theoretical))
    return probabilities, max_formula_error


def maximum_likelihood_estimate(
    counts: np.ndarray,
    shots: int,
    schedule: list[int],
    grid_points: int,
) -> float:
    theta = np.linspace(1e-12, math.pi / 2 - 1e-12, grid_points)
    log_likelihood = np.zeros_like(theta)
    for good_count, power in zip(counts, schedule):
        probability = np.sin((2 * power + 1) * theta) ** 2
        probability = np.clip(probability, 1e-15, 1 - 1e-15)
        log_likelihood += good_count * np.log(probability)
        log_likelihood += (shots - good_count) * np.log1p(-probability)
    best_theta = float(theta[int(np.argmax(log_likelihood))])
    return math.sin(best_theta) ** 2


def run_experiment(
    risk: dict,
    sizes: list[int],
    levels_list: list[int],
    shots: int,
    grid_points: int,
    unit_miles: float,
    xp,
    backend_name: str,
    seed: int,
) -> pd.DataFrame:
    rate_per_mile = risk["serious_rate_per_1000_mile_year"] / 1000
    unit_probability = 1 - math.exp(-rate_per_mile * unit_miles)
    memory = available_bytes(xp, backend_name)
    rng = np.random.default_rng(seed)
    rows = []

    for qubits in sizes:
        exact = 1 - (1 - unit_probability) ** qubits
        for levels in levels_list:
            schedule = [0] + [2**j for j in range(levels - 1)]
            started = time.perf_counter()
            try:
                amplified, formula_error = amplified_probabilities(
                    qubits, unit_probability, schedule, xp, memory
                )
                counts = np.asarray(
                    [rng.binomial(shots, probability) for probability in amplified],
                    dtype=np.int64,
                )
                estimate = maximum_likelihood_estimate(
                    counts, shots, schedule, grid_points
                )
                status = "ok"
                message = ""
            except MemoryError as exc:
                amplified = []
                formula_error = math.nan
                counts = np.asarray([], dtype=np.int64)
                estimate = math.nan
                status = "skipped_memory"
                message = str(exc)
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "backend": backend_name,
                    "risk_qubits": qubits,
                    "objective_qubits": 1,
                    "binary_scenario_count": str(2**qubits),
                    "unit_miles": unit_miles,
                    "unit_serious_probability": unit_probability,
                    "exact_probability": exact,
                    "schedule": "|".join(map(str, schedule)),
                    "shots_per_power": shots,
                    "grover_applications": shots * sum(schedule),
                    "amplification_power_cost": shots
                    * sum(2 * power + 1 for power in schedule),
                    "mle_estimate": estimate,
                    "absolute_error": abs(estimate - exact)
                    if math.isfinite(estimate)
                    else math.nan,
                    "statevector_formula_max_error": formula_error,
                    "good_counts": "|".join(map(str, counts.tolist())),
                    "runtime_seconds": elapsed,
                    "status": status,
                    "message": message,
                }
            )
    if backend_name == "gpu":
        cp.cuda.Stream.null.synchronize()
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--backend", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--levels", type=int, nargs="+")
    parser.add_argument("--shots", type=int)
    parser.add_argument("--grid-points", type=int)
    parser.add_argument("--unit-miles", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()

    if args.mode == "quick":
        default_sizes = [2, 4, 8]
        default_levels = [2, 3, 4]
        default_shots = 256
        default_grid = 100001
    else:
        default_sizes = [2, 4, 8, 12, 16, 20, 24, 26]
        default_levels = [2, 3, 4, 5, 6, 7, 8]
        default_shots = 4096
        default_grid = 1000001

    sizes = args.sizes or default_sizes
    levels = args.levels or default_levels
    shots = args.shots or default_shots
    grid_points = args.grid_points or default_grid
    if min(sizes) < 1 or min(levels) < 1 or shots < 1 or grid_points < 1001:
        raise ValueError("sizes, levels, shots, and grid-points must be positive")

    xp, backend_name = select_backend(args.backend)
    root = args.project_root
    risk_path = root / "results" / "classical_risk" / "pooled_risk.json"
    if not risk_path.exists():
        raise FileNotFoundError(
            f"{risk_path} does not exist; run classical_risk_baseline.py first"
        )
    risk = json.loads(risk_path.read_text(encoding="utf-8"))

    print(json.dumps(backend_info(xp, backend_name), indent=2))
    results = run_experiment(
        risk,
        sizes,
        levels,
        shots,
        grid_points,
        args.unit_miles,
        xp,
        backend_name,
        args.seed,
    )
    output_dir = root / "results" / "qae_risk"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"qae_{args.mode}_{backend_name}.csv"
    results.to_csv(output_path, index=False)
    print(results.to_string(index=False))
    print(f"result: {output_path}")


if __name__ == "__main__":
    main()
