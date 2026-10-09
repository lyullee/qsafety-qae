"""Audit oracle-cost break-even conditions for the HyRAM+ QAE comparison.

Quantum gates and classical samples are deliberately not declared equivalent.
The main output is the maximum admissible ratio c_Q/c_C under which the QAE
query advantage survives for each classical comparator.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_powers(text: str) -> list[int]:
    return [int(value) for value in str(text).split("|")]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    selection_dir = root / "results" / "hyram" / "uncertainty_aware_selection_map"
    selection = pd.read_csv(selection_dir / "uncertainty_aware_algorithm_selection.csv")
    existing = pd.read_csv(
        root / "results" / "hyram" / "precision_conditional_cost_map"
        / "qae_validation_summary_full.csv"
    )
    strict = pd.read_csv(
        root / "results" / "hyram" / "uncertainty_error_budget_validation"
        / "strict_validation_summary_full.csv"
    ).iloc[0]
    resource_model = json.loads(
        (
            root / "results" / "hyram" / "rare_trajectory_qae_scaling"
            / "resource_models_full.json"
        ).read_text(encoding="utf-8")
    )

    schedule_by_queries: dict[int, tuple[list[int], int]] = {}
    for row in existing.itertuples(index=False):
        powers = parse_powers(row.powers)
        denominator = sum(2 * power + 1 for power in powers)
        shots = int(round(int(row.qae_queries) / denominator))
        schedule_by_queries[int(row.qae_queries)] = (powers, shots)
    schedule_by_queries[int(strict["actual_query_budget"])] = (
        parse_powers(strict["powers"]), int(strict["shots_per_power"])
    )

    resource_rows = []
    boundary_rows = []
    comparator_columns = {
        "crude_trajectory_mc": "crude_trajectory_mc_queries",
        "ideal_conditional_splitting": "ideal_conditional_splitting_queries",
        "oracle_importance_splitting": "oracle_importance_splitting_queries",
    }
    for row in selection.itertuples(index=False):
        qae_queries = int(row.validated_mlqae_queries)
        powers, shots = schedule_by_queries[qae_queries]
        denominator = shots * sum(2 * power + 1 for power in powers)
        if denominator != qae_queries:
            raise AssertionError("Reconstructed QAE schedule does not match query budget")
        totals = {}
        maxima = {}
        for resource, model in resource_model.items():
            intercept = float(model["intercept"])
            slope = float(model["slope_per_grover_power"])
            values = np.asarray([intercept + slope * power for power in powers])
            totals[resource] = float(shots * values.sum())
            maxima[resource] = float(values.max())
        resource_rows.append(
            {
                "maximum_numerical_rmse_fraction_of_input_sd": (
                    row.maximum_numerical_rmse_fraction_of_input_sd
                ),
                "qae_queries": qae_queries,
                "powers": "|".join(map(str, powers)),
                "shots_per_power": shots,
                "maximum_power": max(powers),
                "maximum_transpiled_depth": int(round(maxima["transpiled_depth"])),
                "total_executed_transpiled_depth": int(round(totals["transpiled_depth"])),
                "maximum_transpiled_gate_count": int(
                    round(maxima["transpiled_gate_count"])
                ),
                "total_executed_transpiled_gate_count": int(
                    round(totals["transpiled_gate_count"])
                ),
                "maximum_transpiled_cx": int(round(maxima["transpiled_cx"])),
                "total_executed_transpiled_cx": int(round(totals["transpiled_cx"])),
                "average_transpiled_depth_per_qae_query": (
                    totals["transpiled_depth"] / qae_queries
                ),
                "average_transpiled_gates_per_qae_query": (
                    totals["transpiled_gate_count"] / qae_queries
                ),
                "average_transpiled_cx_per_qae_query": (
                    totals["transpiled_cx"] / qae_queries
                ),
            }
        )
        for comparator, column in comparator_columns.items():
            classical_samples = int(getattr(row, column))
            break_even = classical_samples / qae_queries
            boundary_rows.append(
                {
                    "maximum_numerical_rmse_fraction_of_input_sd": (
                        row.maximum_numerical_rmse_fraction_of_input_sd
                    ),
                    "required_relative_rmse": row.required_relative_rmse,
                    "classical_comparator": comparator,
                    "classical_primitive_samples": classical_samples,
                    "qae_queries": qae_queries,
                    "maximum_qae_to_classical_unit_cost_ratio_for_qae_parity": break_even,
                    "qae_has_unit_query_count_advantage": break_even > 1,
                    "interpretation": (
                        f"QAE lower total cost only if c_Q/c_C < {break_even:.6g}"
                    ),
                }
            )

    resources = pd.DataFrame(resource_rows)
    boundaries = pd.DataFrame(boundary_rows)
    access_models = pd.DataFrame(
        [
            {
                "access_model": "explicit_40_value_probability_table",
                "available_information": "all scenario values explicitly stored",
                "implemented_quantum_state": False,
                "included_in_resource_estimate": False,
                "valid_claim": "direct summation in 40 value evaluations",
                "prohibited_claim": "quantum advantage over direct table summation",
            },
            {
                "access_model": "median_parameter_factorized_trajectory",
                "available_information": "scenario weights plus reversible leak/isolation/ignition rotations",
                "implemented_quantum_state": True,
                "included_in_resource_estimate": True,
                "valid_claim": "ideal query comparison and 10-qubit circuit-resource estimate",
                "prohibited_claim": "current-QPU runtime advantage",
            },
            {
                "access_model": "reversible_classical_sampler_or_qrom",
                "available_information": "coherent callable generator assumed",
                "implemented_quantum_state": False,
                "included_in_resource_estimate": False,
                "valid_claim": "conditional query-complexity statement only",
                "prohibited_claim": "end-to-end speedup without reversible implementation cost",
            },
            {
                "access_model": "uncertainty_augmented_460_lognormal_variables",
                "available_information": "92 physical components times five leak-size distributions",
                "implemented_quantum_state": False,
                "included_in_resource_estimate": False,
                "valid_claim": "classical uncertainty sensitivity result",
                "prohibited_claim": "free coherent state preparation of all uncertain inputs",
            },
        ]
    )

    output_dir = root / "results" / "hyram" / "oracle_cost_boundary"
    output_dir.mkdir(parents=True, exist_ok=True)
    resources.to_csv(output_dir / "qae_transpiled_resource_totals.csv", index=False)
    boundaries.to_csv(output_dir / "oracle_unit_cost_break_even.csv", index=False)
    access_models.to_csv(output_dir / "access_model_scope.csv", index=False)
    (output_dir / "claim_guardrails.json").write_text(
        json.dumps(
            {
                "cost_equation": "QAE wins iff Q*c_Q < N_C*c_C, equivalently c_Q/c_C < N_C/Q",
                "unit_warning": (
                    "A coherent quantum-oracle query and a classical conditional sample are not "
                    "physically equal-cost operations. Break-even ratios are thresholds, not measured runtimes."
                ),
                "resource_scope": (
                    "Transpiled resource totals cover the implemented 10-qubit median-parameter "
                    "factorized trajectory circuit only. They exclude 460-variable uncertainty "
                    "state preparation, fault-tolerance, routing to a named QPU, and error correction."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    labels = {
        "crude_trajectory_mc": "Crude trajectory MC",
        "ideal_conditional_splitting": "Ideal conditional splitting",
        "oracle_importance_splitting": "Oracle-informed importance",
    }
    colors = {
        "crude_trajectory_mc": "#777777",
        "ideal_conditional_splitting": "#e67e22",
        "oracle_importance_splitting": "#2e8b57",
    }
    fig, ax = plt.subplots(figsize=(9.3, 5.7))
    for comparator in comparator_columns:
        subset = boundaries.loc[boundaries["classical_comparator"].eq(comparator)]
        x = 100 * subset["maximum_numerical_rmse_fraction_of_input_sd"].to_numpy(float)
        y = subset[
            "maximum_qae_to_classical_unit_cost_ratio_for_qae_parity"
        ].to_numpy(float)
        ax.plot(x, y, marker="o", linewidth=2, label=labels[comparator], color=colors[comparator])
    ax.axhline(1, color="#c62828", linestyle="--", label="Equal unit cost")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("Maximum numerical RMSE (% of input standard deviation)")
    ax.set_ylabel("Maximum admissible $c_Q/c_C$ for QAE parity")
    ax.set_title("Oracle-cost break-even boundary")
    ax.grid(which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"oracle_cost_break_even.{extension}", dpi=220)
    plt.close(fig)

    print("Oracle unit-cost break-even thresholds")
    print(boundaries.to_string(index=False))
    print("\nImplemented-circuit resource totals")
    print(resources.to_string(index=False))
    print("\nAccess-model scope")
    print(access_models.to_string(index=False))
    print(f"\nresults: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
