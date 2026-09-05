import json

import pytest

from qsafety.cli import main


def _write_model(tmp_path):
    path = tmp_path / "model.csv"
    path.write_text(
        "label,weight,probability\na,1,0.1\nb,3,0.3\n",
        encoding="utf-8",
    )
    return path


def test_validate_command_prints_json(tmp_path, capsys):
    path = _write_model(tmp_path)
    exit_code = main(["validate", str(path), "--label-column", "label"])
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["exact_probability"] == pytest.approx(0.25)
    assert output["scenario_qubits"] == 1


def test_compare_command_prints_both_methods(tmp_path, capsys):
    path = _write_model(tmp_path)
    exit_code = main(
        [
            "compare",
            str(path),
            "--powers",
            "0",
            "1",
            "--shots",
            "16",
            "--repeats",
            "4",
            "--grid-points",
            "1001",
            "--seed",
            "2",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["classical_mc"]["method"] == "classical_mc"
    assert output["ideal_mlqae"]["method"] == "ideal_mlqae"


def test_validate_single_stratum_reports_no_circuit_size(tmp_path, capsys):
    path = tmp_path / "single.csv"
    path.write_text("weight,probability\n1,0.2\n", encoding="utf-8")
    assert main(["validate", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["scenario_qubits"] is None
