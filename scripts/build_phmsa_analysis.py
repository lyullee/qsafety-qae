"""Build leakage-aware PHMSA tables for classical Monte Carlo and QAE.

Raw source files are never modified. The outputs contain no names, contact
details, street addresses, coordinates, or incident narratives.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIO_START_YEAR = 2010
EXPOSURE_START_YEAR = 2017
END_YEAR = 2025


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_incidents(raw_dir: Path) -> pd.DataFrame:
    path = (
        raw_dir
        / "incident_gtg_2010_present"
        / "incident_gas_transmission_gathering_jan2010_present.txt"
    )
    return pd.read_csv(path, sep="\t", encoding="cp1252", low_memory=False)


def load_annual(raw_dir: Path) -> pd.DataFrame:
    frames = []
    annual_dir = raw_dir / "annual_gtg_2010_present"
    for year in range(EXPOSURE_START_YEAR, END_YEAR + 1):
        path = annual_dir / f"GT AR {year} Part A to D.csv"
        frames.append(pd.read_csv(path, encoding="cp1252", low_memory=False))
    return pd.concat(frames, ignore_index=True)


def build_incident_scenarios(incidents: pd.DataFrame) -> pd.DataFrame:
    frame = incidents[
        incidents["IYEAR"].between(SCENARIO_START_YEAR, END_YEAR)
        & incidents["COMMODITY_RELEASED_TYPE"].eq("NATURAL GAS")
    ].copy()

    fatal = numeric(frame["FATAL"]).fillna(0)
    injury = numeric(frame["INJURE"]).fillna(0)
    installation_year = numeric(frame["INSTALLATION_YEAR"])
    pipe_age = numeric(frame["IYEAR"]) - installation_year
    pipe_age = pipe_age.where(pipe_age.between(0, 200))
    accident_pressure = numeric(frame["ACCIDENT_PSIG"])
    maximum_pressure = numeric(frame["MOP_PSIG"])
    pressure_ratio = accident_pressure / maximum_pressure.replace(0, np.nan)

    scenarios = pd.DataFrame(
        {
            "report_number": frame["REPORT_NUMBER"].astype(str),
            "year": numeric(frame["IYEAR"]).astype("Int64"),
            "operator_id": numeric(frame["OPERATOR_ID"]).astype("Int64"),
            "state": frame["ONSHORE_STATE_ABBREVIATION"],
            "on_offshore": frame["ON_OFF_SHORE"],
            "system_part": frame["SYSTEM_PART_INVOLVED"],
            "facility_type": frame["PIPE_FACILITY_TYPE"],
            "pipeline_function": frame["PIPELINE_FUNCTION"],
            "material": frame["MATERIAL_INVOLVED"],
            "installation_year": installation_year.astype("Int64"),
            "pipe_age_years": pipe_age,
            "class_location": frame["CLASS_LOCATION_TYPE"],
            "hca": frame["COULD_BE_HCA"],
            "pir_radius_ft": numeric(frame["PIR_RADIUS"]),
            "incident_pressure_psig": accident_pressure,
            "maop_psig": maximum_pressure,
            "pressure_ratio": pressure_ratio,
            "pressure_status": frame["ACCIDENT_PRESSURE"],
            "pressure_restriction": frame["PRESSURE_RESTRICTION_IND"],
            "internal_inspection_capable": frame["INTERNAL_INSPECTION_IND"],
            "scada_in_place": frame["SCADA_IN_PLACE_IND"],
            "scada_operating": frame["SCADA_OPERATING_IND"],
            "scada_functional": frame["SCADA_FUNCTIONAL_IND"],
            "apparent_cause": frame["CAUSE"],
            # Official PHMSA Serious Incident definition: fatality or injury
            # requiring inpatient hospitalization.
            "serious_incident": ((fatal > 0) | (injury > 0)).astype("int8"),
        }
    )
    return scenarios.reset_index(drop=True)


def build_exposure_table(
    annual: pd.DataFrame, scenarios: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annual_ng = annual[
        annual["PARTA5COMMODITY"].eq("Natural Gas")
        & annual["REPORT_YEAR"].between(EXPOSURE_START_YEAR, END_YEAR)
    ].copy()
    annual_ng = annual_ng.dropna(subset=["OPERATOR_ID", "REPORT_YEAR"])

    duplicate_keys = annual_ng.duplicated(["OPERATOR_ID", "REPORT_YEAR"], keep=False)
    if duplicate_keys.any():
        raise ValueError("Natural-gas annual operator-year keys are not unique")

    exposure = pd.DataFrame(
        {
            "operator_id": numeric(annual_ng["OPERATOR_ID"]).astype("Int64"),
            "year": numeric(annual_ng["REPORT_YEAR"]).astype("Int64"),
            "annual_report_number": numeric(annual_ng["REPORT_NUMBER"])
            .astype("Int64")
            .astype("string"),
            "total_miles": numeric(annual_ng["PARTDTOTALMILES"]),
            "hca_miles": numeric(annual_ng["PARTBHCATOTAL"]),
            "section_192_710_miles": numeric(annual_ng["PARTB192MILESTOTAL"]),
            "class_3_4_miles": numeric(annual_ng["PARTBCLASS34MILESTOTAL"]),
            "class_1_2_miles": numeric(annual_ng["PARTBCLASS12MILESTOTAL"]),
            "onshore_volume_mmscf": numeric(annual_ng["PARTCONNG"]),
            "offshore_volume_mmscf": numeric(annual_ng["PARTCOFFNG"]),
        }
    )

    exposure_scenarios = scenarios[
        scenarios["year"].between(EXPOSURE_START_YEAR, END_YEAR)
    ]
    event_counts = (
        exposure_scenarios.dropna(subset=["operator_id", "year"])
        .groupby(["operator_id", "year"], as_index=False)
        .agg(
            incident_count=("report_number", "count"),
            serious_incident_count=("serious_incident", "sum"),
        )
    )
    exposure = exposure.merge(event_counts, on=["operator_id", "year"], how="left")
    exposure[["incident_count", "serious_incident_count"]] = exposure[
        ["incident_count", "serious_incident_count"]
    ].fillna(0).astype("int64")
    exposure["incidents_per_1000_mile_year"] = np.where(
        exposure["total_miles"] > 0,
        1000 * exposure["incident_count"] / exposure["total_miles"],
        np.nan,
    )
    exposure["serious_per_1000_mile_year"] = np.where(
        exposure["total_miles"] > 0,
        1000 * exposure["serious_incident_count"] / exposure["total_miles"],
        np.nan,
    )

    annual_keys = set(
        map(tuple, exposure[["operator_id", "year"]].astype(int).to_numpy())
    )
    matched = exposure_scenarios.apply(
        lambda row: (
            pd.notna(row["operator_id"])
            and pd.notna(row["year"])
            and (int(row["operator_id"]), int(row["year"])) in annual_keys
        ),
        axis=1,
    )
    unmatched = exposure_scenarios.loc[
        ~matched, ["report_number", "year", "operator_id", "serious_incident"]
    ].copy()
    return exposure.sort_values(["year", "operator_id"]), unmatched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    args = parser.parse_args()
    raw_dir = args.project_root / "data" / "raw" / "phmsa"
    output_dir = args.project_root / "data" / "processed" / "phmsa"
    output_dir.mkdir(parents=True, exist_ok=True)

    incidents = load_incidents(raw_dir)
    annual = load_annual(raw_dir)
    scenarios = build_incident_scenarios(incidents)
    exposure, unmatched = build_exposure_table(annual, scenarios)

    scenario_path = output_dir / "incident_scenarios_2010_2025.csv"
    exposure_path = output_dir / "operator_year_exposure_2017_2025.csv"
    unmatched_path = output_dir / "unmatched_incidents_2017_2025.csv"
    metadata_path = output_dir / "metadata.json"

    scenarios.to_csv(scenario_path, index=False, encoding="utf-8")
    exposure.to_csv(exposure_path, index=False, encoding="utf-8")
    unmatched.to_csv(unmatched_path, index=False, encoding="utf-8")

    metadata = {
        "scope": (
            "PHMSA gas transmission/gathering, natural gas; incident scenarios "
            "2010-2025 and annual exposure 2017-2025"
        ),
        "serious_incident_definition": (
            "fatality or injury requiring inpatient hospitalization"
        ),
        "incident_scenario_rows": int(len(scenarios)),
        "serious_incident_rows": int(scenarios["serious_incident"].sum()),
        "operator_year_rows": int(len(exposure)),
        "exposure_period_incidents": int(
            scenarios["year"].between(EXPOSURE_START_YEAR, END_YEAR).sum()
        ),
        "matched_incidents": int(
            scenarios["year"].between(EXPOSURE_START_YEAR, END_YEAR).sum()
            - unmatched.shape[0]
        ),
        "unmatched_incidents": int(len(unmatched)),
        "conditioning_note": (
            "Apparent cause and incident-time operating fields are scenario "
            "conditions, not guaranteed pre-incident predictors."
        ),
        "excluded_for_leakage": [
            "fatality and injury counts except as the outcome",
            "evacuation",
            "ignition and explosion",
            "released volume",
            "property damage and costs",
            "shutdown and response times",
            "narrative",
        ],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
