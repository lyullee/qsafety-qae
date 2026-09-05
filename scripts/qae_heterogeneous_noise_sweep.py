#!/usr/bin/env python
"""Repeated noise sweep for the PHMSA heterogeneous-risk QAE circuits."""

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

from qae_heterogeneous_risk import BASIS, INPUT_DIR, OUTPUT_DIR, load_strata, make_a_gate, make_qae_circuit
from qae_qiskit_runner import custom_noise_model, good_count, mle_amplitude, mle_fit_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--scenario-qubits", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--powers", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--scales", type=float, nargs="+")
    parser.add_argument("--shots", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--base-p1", type=float, default=0.001)
    parser.add_argument("--base-p2", type=float, default=0.01)
    parser.add_argument("--base-readout", type=float, default=0.02)
    parser.add_argument("--input-tag", default="full")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--optimization", type=int, choices=[0, 1, 2, 3], default=2)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    for value in [args.input_tag, args.tag]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("Tags may contain only letters, numbers, underscore, or hyphen")
    if args.mode == "quick":
        scales = args.scales or [0.0, 0.025, 0.05, 0.075, 0.1]
        shots = args.shots or 4096
        repeats = args.repeats or 5
    else:
        scales = args.scales or [0.0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2]
        shots = args.shots or 8192
        repeats = args.repeats or 30

    circuits = []
    keys = []
    exact_by_n = {}
    for n in args.scenario_qubits:
        strata = load_strata(n, args.input_tag)
        weights = strata.weight.to_numpy(float)
        risks = strata.probability_mean.to_numpy(float)
        exact_by_n[n] = float(np.dot(weights, risks))
        a_gate, _ = make_a_gate(weights, risks)
        for power in args.powers:
            circuits.append(make_qae_circuit(a_gate, n, power, measure=True))
            keys.append((n, power))
    physical = transpile(circuits, basis_gates=BASIS, optimization_level=args.optimization)

    rows = []
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
            run_seed = args.seed + scale_index * 1000 + repeat
            result = backend.run(physical, shots=shots, seed_simulator=run_seed).result()
            counts = [result.get_counts(i) for i in range(len(physical))]
            for n in args.scenario_qubits:
                selected = [i for i, key in enumerate(keys) if key[0] == n]
                goods = [good_count(counts[i]) for i in selected]
                estimate = mle_amplitude(goods, shots, args.powers)
                exact = exact_by_n[n]
                rows.append({
                    "mode": args.mode,
                    "scenario_qubits": n,
                    "risk_strata": 2**n,
                    "total_qubits": n + 1,
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
                    **mle_fit_diagnostics(estimate, goods, shots, args.powers),
                })
            print(f"finished scale={scale:g}, repeat={repeat + 1}/{repeats}", flush=True)

    raw = pd.DataFrame(rows)
    summary = raw.groupby(["scenario_qubits", "risk_strata", "noise_scale"], as_index=False).agg(
        exact_probability=("exact_probability", "first"),
        mean_mle_probability=("mle_probability", "mean"),
        std_mle_probability=("mle_probability", "std"),
        mean_bias=("signed_error", "mean"),
        mean_absolute_error=("absolute_error", "mean"),
        rmse=("signed_error", lambda x: float(np.sqrt(np.mean(x**2)))),
        mean_fit_rmse=("mle_fit_rmse", "mean"),
        model_mismatch_rate=("fit_status", lambda x: float(np.mean(x == "model_mismatch"))),
    )
    summary["relative_rmse"] = summary.rmse / summary.exact_probability
    summary["usable_10pct"] = (summary.relative_rmse <= 0.10) & (summary.model_mismatch_rate == 0)
    summary["powers"] = "|".join(map(str, args.powers))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUTPUT_DIR / f"noise_raw_{args.mode}_{args.tag}.csv"
    summary_path = OUTPUT_DIR / f"noise_summary_{args.mode}_{args.tag}.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print("\nHeterogeneous QAE noise summary")
    print(summary.to_string(index=False))
    print(f"\nRaw: {raw_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
