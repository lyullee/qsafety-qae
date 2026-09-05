"""Command-line interface for QSafety QAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import RiskDistribution
from .estimators import compare_estimators


def _model(args) -> RiskDistribution:
    return RiskDistribution.from_csv(
        args.csv,
        weight_column=args.weight_column,
        probability_column=args.probability_column,
        label_column=args.label_column,
    )


def _add_input(parser):
    parser.add_argument("csv", type=Path)
    parser.add_argument("--weight-column", default="weight")
    parser.add_argument("--probability-column", default="probability")
    parser.add_argument("--label-column")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qsafety", description="QAE tools for safety probabilities")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate a risk-distribution CSV")
    _add_input(validate)
    compare = sub.add_parser("compare", help="compare MC and ideal ML-QAE at equal query budget")
    _add_input(compare)
    compare.add_argument("--powers", type=int, nargs="+", default=[0, 1, 2])
    compare.add_argument("--shots", type=int, default=1024)
    compare.add_argument("--repeats", type=int, default=1000)
    compare.add_argument("--seed", type=int, default=20260905)
    compare.add_argument("--grid-points", type=int, default=65537)
    circuits = sub.add_parser("circuits", help="validate Qiskit state preparation and Grover circuits")
    _add_input(circuits)
    circuits.add_argument("--powers", type=int, nargs="+", default=[0, 1, 2])
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    model = _model(args)
    if args.command == "validate":
        output = {
            "strata": model.size,
            "weight_sum": float(model.weights.sum()),
            "exact_probability": model.probability,
            "scenario_qubits": (
                model.scenario_qubits
                if model.size >= 2 and not model.size & (model.size - 1)
                else None
            ),
        }
    elif args.command == "compare":
        mc, qae = compare_estimators(
            model,
            args.powers,
            shots=args.shots,
            repeats=args.repeats,
            seed=args.seed,
            grid_points=args.grid_points,
        )
        output = {"classical_mc": mc.to_dict(), "ideal_mlqae": qae.to_dict()}
        output["rmse_ratio_classical_over_qae"] = mc.rmse / qae.rmse if qae.rmse else None
    else:
        from .circuits import validate_bundle

        output = validate_bundle(model, args.powers)
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
