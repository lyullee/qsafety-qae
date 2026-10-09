# QSafety QAE

`qsafety-qae` is a small research package for testing quantum amplitude
estimation (QAE) on safety and reliability probability models. It separates a
validated probability model from the estimator, simulator, circuit builder,
and optional QPU runner so that examples are reproducible without depending on
the original project directory.

The package supports:

- finite mixtures of fault, failure, or accident-scenario probabilities;
- exact weighted probability calculation;
- classical Monte Carlo and ideal maximum-likelihood QAE (ML-QAE) at the same
  equivalent oracle-query budget;
- Qiskit state preparation, Grover amplification checks, and Aer simulation;
- opt-in IBM Quantum preview and execution with a mandatory confirmation flag.

## Installation

Core estimators require only NumPy:

```bash
python -m pip install qsafety-qae
```

Install optional circuit or IBM Runtime support when needed:

```bash
python -m pip install "qsafety-qae[qiskit]"
python -m pip install "qsafety-qae[ibm]"
```

## Python interface

```python
from qsafety import RiskDistribution, compare_estimators

risk = RiskDistribution(
    weights=[0.25, 0.25, 0.25, 0.25],
    probabilities=[0.0095, 0.0145, 0.0231, 0.0638],
    labels=["very low", "low", "medium", "high"],
)

print(risk.probability)

mc, qae = compare_estimators(
    risk,
    powers=[0, 1, 2],
    shots=1024,
    repeats=1000,
    seed=42,
)
print(mc.rmse, qae.rmse)
```

Build and validate actual Qiskit circuits:

```python
from qsafety.circuits import build_qae_circuits, run_aer, validate_bundle

print(validate_bundle(risk, powers=[0, 1, 2]))
bundle = build_qae_circuits(risk, powers=[0, 1, 2])
result = run_aer(risk, powers=[0, 1, 2], shots=4096, seed=42)
print(result.estimate.probability)
```

## CSV and command-line interface

Input CSV files need one row per scenario or risk stratum:

```csv
label,weight,probability
very_low,0.25,0.0095
low,0.25,0.0145
medium,0.25,0.0231
high,0.25,0.0638
```

```bash
qsafety validate examples/phmsa_risk_strata.csv --label-column label
qsafety compare examples/phmsa_risk_strata.csv --label-column label \
  --powers 0 1 2 --shots 1024 --repeats 1000
qsafety circuits examples/phmsa_risk_strata.csv --label-column label
```

Commands print machine-readable JSON. The circuit command is available only
with the `qiskit` extra.

## IBM Quantum safety guard

`preview_ibm` transpiles circuits but does not submit them. The returned backend
and circuit resources should be reviewed before calling `run_ibm`. Submission
is blocked unless `confirm=True` is passed explicitly.

```python
from qsafety.ibm import preview_ibm, run_ibm

preview = preview_ibm(risk, shots=512, open_plan_only=True)
print(preview.backend, preview.executions, preview.resources)

# This is the only call that submits a QPU workload.
result = run_ibm(preview, confirm=True)
```

Credentials are read by `qiskit-ibm-runtime`; this package never stores API
tokens.

## Interpretation limits

The MC/QAE comparison is an algorithm-level comparison at an equal equivalent
oracle-query budget. It is not a wall-clock speed, energy-use, or end-to-end
quantum-advantage claim. State preparation, transpilation, QPU queue time, and
error mitigation are separate costs. Input-data and model uncertainty can also
be larger than either estimator's sampling error.

The included PHMSA-derived four-stratum file is a compact worked example, not a
pipeline-segment prediction model. It estimates a conditional outcome among
reported incidents and must not be used as an operational safety decision rule.

## Reproducible development

```bash
python -m pip install -e ".[dev,qiskit]"
pytest
python -m build
twine check dist/*
```

Legacy research scripts remain in the repository for traceability. Generated
study data and results stay local and are excluded from the public package.
Stable third-party integrations should use the public `qsafety` interface
documented above.

## License and citation

The project's independently authored software is released under the MIT License.
Third-party software, standards, data, and documentation retain their own terms;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Citation metadata is in
`CITATION.cff`. The version DOI will be added after the first Zenodo archive is
created.

## Research scripts

Selected estimator, uncertainty, and finite-oracle scripts are available in this
repository. Their optional dependencies can be installed with
`python -m pip install -e ".[dev,qiskit,research]"`.
See [RESEARCH_CODE.md](RESEARCH_CODE.md) for the published scope, required local
inputs, and data-free tests. This code-only update does not bundle the full
manuscript reproduction dataset and does not change the existing PyPI release.
