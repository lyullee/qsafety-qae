"""Repeated QAE noise-sensitivity sweep for user-run experiments.

Quick mode is a small screening run. Full mode is intentionally expensive and
is never invoked by the development workflow. All circuits use the one-clean
ancilla MCX synthesis validated by qae_qiskit_runner.py.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit_aer import AerSimulator

from qae_qiskit_runner import (
    custom_noise_model,
    good_count,
    make_circuit,
    mle_amplitude,
    mle_fit_diagnostics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--powers", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--scales", type=float, nargs="+")
    parser.add_argument("--shots", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--base-p1", type=float, default=0.001)
    parser.add_argument("--base-p2", type=float, default=0.01)
    parser.add_argument("--base-readout", type=float, default=0.02)
    parser.add_argument("--unit-miles", type=float, default=1000.0)
    parser.add_argument("--optimization", type=int, choices=[0, 1, 2, 3], default=1)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--tag",
        default="",
        help="Optional filename suffix using letters, numbers, underscore, or hyphen",
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.tag):
        raise ValueError("--tag may contain only letters, numbers, underscore, or hyphen")

    if args.mode == "quick":
        sizes = args.sizes or [2, 4, 8]
        scales = args.scales or [0.0, 0.1, 0.25, 0.5, 1.0]
        shots = args.shots or 1024
        repeats = args.repeats or 3
    else:
        sizes = args.sizes or [2, 4, 8]
        scales = args.scales or [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
        shots = args.shots or 4096
        repeats = args.repeats or 30

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

    circuits = []
    keys = []
    for size in sizes:
        for power in args.powers:
            circuits.append(make_circuit(size, rotation, power, "one-clean"))
            keys.append((size, power))

    # Transpile once to a fixed basis so every noise level compares the same
    # physical circuit rather than a different compiler outcome.
    compile_noise = custom_noise_model(
        args.base_p1, args.base_p2, args.base_readout
    )
    compile_backend = AerSimulator(method="density_matrix", noise_model=compile_noise)
    physical_circuits = transpile(
        circuits, backend=compile_backend, optimization_level=args.optimization
    )

    raw_rows = []
    for scale_index, scale in enumerate(scales):
        if scale == 0:
            backend = AerSimulator(method="density_matrix")
        else:
            noise = custom_noise_model(
                args.base_p1 * scale,
                args.base_p2 * scale,
                args.base_readout * scale,
            )
            backend = AerSimulator(method="density_matrix", noise_model=noise)
        for repeat in range(repeats):
            run_seed = args.seed + 1000 * scale_index + repeat
            result = backend.run(
                physical_circuits, shots=shots, seed_simulator=run_seed
            ).result()
            all_counts = [result.get_counts(i) for i in range(len(circuits))]
            for size in sizes:
                selected = [i for i, key in enumerate(keys) if key[0] == size]
                goods = [good_count(all_counts[i]) for i in selected]
                estimate = mle_amplitude(goods, shots, args.powers)
                diagnostics = mle_fit_diagnostics(
                    estimate, goods, shots, args.powers
                )
                exact = 1 - (1 - unit_probability) ** size
                raw_rows.append(
                    {
                        "mode": args.mode,
                        "risk_qubits": size,
                        "objective_qubits": 1,
                        "ancilla_qubits": 0 if size <= 2 else 1,
                        "binary_scenarios": 2**size,
                        "noise_scale": scale,
                        "one_qubit_error": args.base_p1 * scale,
                        "two_qubit_error": args.base_p2 * scale,
                        "readout_error": args.base_readout * scale,
                        "repeat": repeat,
                        "seed": run_seed,
                        "shots_per_power": shots,
                        "powers": "|".join(map(str, args.powers)),
                        "good_counts": "|".join(map(str, goods)),
                        "exact_probability": exact,
                        "mle_probability": estimate,
                        "signed_error": estimate - exact,
                        "absolute_error": abs(estimate - exact),
                        **diagnostics,
                    }
                )
            print(f"finished noise_scale={scale:g}, repeat={repeat + 1}/{repeats}")

    raw = pd.DataFrame(raw_rows)
    summary = (
        raw.groupby(["risk_qubits", "noise_scale"], as_index=False)
        .agg(
            binary_scenarios=("binary_scenarios", "first"),
            exact_probability=("exact_probability", "first"),
            mean_mle_probability=("mle_probability", "mean"),
            std_mle_probability=("mle_probability", "std"),
            mean_bias=("signed_error", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            rmse=("signed_error", lambda values: float(np.sqrt(np.mean(values**2)))),
            mean_fit_rmse=("mle_fit_rmse", "mean"),
            model_mismatch_rate=(
                "fit_status", lambda values: float(np.mean(values == "model_mismatch"))
            ),
        )
        .sort_values(["risk_qubits", "noise_scale"])
    )
    summary["relative_rmse"] = summary["rmse"] / summary["exact_probability"]
    summary["usable_10pct"] = (
        (summary["relative_rmse"] <= 0.10)
        & (summary["model_mismatch_rate"] == 0)
    )

    output_dir = root / "results" / "qae_risk" / "qiskit"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    raw_path = output_dir / f"noise_sweep_raw_{args.mode}{suffix}.csv"
    summary_path = output_dir / f"noise_sweep_summary_{args.mode}{suffix}.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"raw results: {raw_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
