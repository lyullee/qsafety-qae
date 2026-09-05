"""Audit PHMSA gas transmission/gathering source files for QAE research.

The script does not modify raw data. It writes a compact JSON summary without
operator names, contact details, coordinates, narratives, or other row data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ANNUAL_YEARS = range(2017, 2026)


def missing_summary(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for column in columns:
        if column not in frame:
            continue
        series = frame[column]
        result[column] = {
            "nonmissing": int(series.notna().sum()),
            "missing_pct": round(float(series.isna().mean() * 100), 2),
            "unique": int(series.nunique(dropna=True)),
        }
    return result


def audit(project_root: Path) -> dict:
    raw = project_root / "data" / "raw" / "phmsa"
    incident_path = (
        raw
        / "incident_gtg_2010_present"
        / "incident_gas_transmission_gathering_jan2010_present.txt"
    )
    annual_dir = raw / "annual_gtg_2010_present"

    incidents = pd.read_csv(
        incident_path,
        sep="\t",
        encoding="cp1252",
        low_memory=False,
    )

    annual_frames = []
    annual_by_year = {}
    for year in ANNUAL_YEARS:
        path = annual_dir / f"GT AR {year} Part A to D.csv"
        frame = pd.read_csv(path, encoding="cp1252", low_memory=False)
        annual_frames.append(frame)
        annual_by_year[str(year)] = {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "operators": int(frame["OPERATOR_ID"].nunique(dropna=True)),
            "total_miles_missing_pct": round(
                float(frame["PARTDTOTALMILES"].isna().mean() * 100), 2
            ),
        }
    annual = pd.concat(annual_frames, ignore_index=True)

    numeric = lambda column: pd.to_numeric(incidents[column], errors="coerce").fillna(0)
    provisional_severe = (
        (numeric("FATAL") > 0)
        | (numeric("INJURE") > 0)
        | (numeric("NUM_PUB_EVACUATED") > 0)
        | incidents["IGNITE_IND"].eq("YES")
        | incidents["EXPLODE_IND"].eq("YES")
    )

    annual_keys_frame = (
        annual[["OPERATOR_ID", "REPORT_YEAR"]]
        .dropna()
        .astype(int)
        .drop_duplicates()
    )
    annual_keys = set(map(tuple, annual_keys_frame.to_numpy()))
    join_window = incidents[
        incidents["IYEAR"].between(min(ANNUAL_YEARS), max(ANNUAL_YEARS))
        & incidents["OPERATOR_ID"].notna()
    ].copy()
    joinable = pd.Series(
        [
            (int(operator_id), int(year)) in annual_keys
            for operator_id, year in zip(join_window["OPERATOR_ID"], join_window["IYEAR"])
        ],
        index=join_window.index,
    )

    cause_counts = incidents["CAUSE"].value_counts(dropna=False)
    year_counts = incidents.groupby("IYEAR").size()
    annual_miles = pd.to_numeric(annual["PARTDTOTALMILES"], errors="coerce")

    return {
        "incident": {
            "rows": int(len(incidents)),
            "columns": int(incidents.shape[1]),
            "year_min": int(incidents["IYEAR"].min()),
            "year_max": int(incidents["IYEAR"].max()),
            "unique_reports": int(incidents["REPORT_NUMBER"].nunique()),
            "unique_operators": int(incidents["OPERATOR_ID"].nunique()),
            "year_counts": {str(int(k)): int(v) for k, v in year_counts.items()},
            "cause_counts": {str(k): int(v) for k, v in cause_counts.items()},
            "selected_field_quality": missing_summary(
                incidents,
                [
                    "FATAL",
                    "INJURE",
                    "NUM_PUB_EVACUATED",
                    "UNINTENTIONAL_RELEASE",
                    "EST_COST_PROP_DAMAGE",
                    "PRPTY",
                    "CAUSE",
                    "PIPE_DIAMETER",
                    "INSTALLATION_YEAR",
                    "ACCIDENT_PSIG",
                    "MOP_PSIG",
                    "LOCATION_LATITUDE",
                    "LOCATION_LONGITUDE",
                    "NARRATIVE",
                ],
            ),
            "provisional_severe_definition": (
                "fatality>0 OR injury>0 OR evacuated>0 OR ignition=YES OR explosion=YES"
            ),
            "provisional_severe_rows": int(provisional_severe.sum()),
            "provisional_severe_pct": round(float(provisional_severe.mean() * 100), 2),
        },
        "annual_2017_2025": {
            "rows": int(len(annual)),
            "columns_part_a_to_d": int(annual.shape[1]),
            "operator_year_keys": int(len(annual_keys_frame)),
            "invalid_key_rows": int(
                annual[["OPERATOR_ID", "REPORT_YEAR"]].isna().any(axis=1).sum()
            ),
            "part_a_to_d_by_year": annual_by_year,
            "total_miles": {
                "nonmissing": int(annual_miles.notna().sum()),
                "missing_pct": round(float(annual_miles.isna().mean() * 100), 2),
                "median": round(float(annual_miles.median()), 3),
                "maximum": round(float(annual_miles.max()), 3),
            },
        },
        "incident_annual_join_2017_2025": {
            "incident_rows": int(len(join_window)),
            "joinable_rows": int(joinable.sum()),
            "join_rate_pct": round(float(joinable.mean() * 100), 2),
            "key": ["OPERATOR_ID", "YEAR"],
            "caveat": (
                "Annual reports can contain multiple state/system rows per operator-year; "
                "final exposure aggregation must preserve state and system type."
            ),
        },
        "limitations": [
            "The composite severe label is provisional and must be reconciled with PHMSA definitions.",
            "EXPLODE_IND is missing for part of the incident schema history.",
            "PIPE_DIAMETER applies only to relevant incident facility types and is structurally missing elsewhere.",
            "The official annual ZIP has CRC/header errors in several entries; CSV Part A-D files for 2017-2025 were recovered.",
            "Incident records include post-event fields that must be excluded from predictive inputs to prevent leakage.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    output = args.output or args.project_root / "data" / "audit" / "phmsa_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = audit(args.project_root)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
