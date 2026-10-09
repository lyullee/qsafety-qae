# Research code publication scope

This code-only update publishes selected independent research methods while
keeping third-party implementations and uncertain redistribution material
outside the repository. It does not claim to provide a complete reproduction
archive for every manuscript result. Release 0.1.3 archives these scripts in
GitHub/Zenodo; its PyPI wheel and source distribution contain the core `qsafety`
package, not these repository-only scripts. The `research` extra installs their
optional dependencies but does not install the scripts themselves.

## Install and run data free checks

From a clone of the versioned repository:

```bash
git clone --branch v0.1.3 https://github.com/lyullee/qsafety-qae.git
cd qsafety-qae
python -m pip install -e ".[dev,qiskit,research]"
python -m pytest tests/test_research_methods.py
python scripts/audit_public_release.py
```

The tests use synthetic in-memory inputs. They require no Sandia code, API or
ASME lookup table, manuscript CSV, cloud account, or QPU execution. They check
equivalent-query accounting, bounded maximum-likelihood fitting, classical
product-estimator variance, lognormal propagation, and finite-table Grover
amplification. Synthetic values are not study observations or field data.

## Published methods

| Code | Purpose | Additional inputs for archived command-line runs |
| --- | --- | --- |
| `research_schedule.py` | Data-independent geometric schedule and exact query accounting | None |
| `hyram_rare_trajectory_qae_scaling.py` | Ideal bounded MLAE and scaling/resource analysis | Local trajectory summary and circuit resources |
| `hyram_strong_rare_event_baselines.py` | Conditional and oracle-informed importance baselines | Local scenario inventory, trajectory summary, and MLAE precision table |
| `hyram_precision_conditional_cost_map.py` | Precision and conditional-sampling cost sensitivity | Local scenario inventory and trajectory summary |
| `hyram_qae_schedule_sensitivity.py` | Independent training/validation schedule comparisons | Local classical comparison table for the reporting join |
| `hyram_qae_prior_bound_sensitivity.py` | Likelihood-search interval sensitivity | Local trajectory summary |
| `hyram_system_scenario_inventory.py` | Independent inventory calculations from caller-supplied parameters | Rights-cleared `data/external/hyram/benchmark_manifest.json` |
| `hyram_leak_frequency_uncertainty.py` | Independent lognormal input propagation | Local parameter manifest, trajectory summary, and cost map |
| `hyram_stochastic_trajectory_oracle.py` | Collapsed and explicit leak/isolation/ignition circuits | Local `component_leak_scenarios.csv` |
| `hyram_uncertainty_compressed_oracle.py` | Independent-mean quadrature circuit | Local scenario inventory and uncertainty summary |
| `hyram_oracle_cost_boundary.py` | Unit-cost break-even and circuit-resource extrapolation | Local selection/validation tables and resource model |
| `helpr_empirical_qrom_oracle.py` | Sparse finite-table membership oracle | Caller-supplied frozen lifetime CSVs |
| `helpr_empirical_qrom_mlqae.py` | Finite-table Grover amplification and ideal MLAE | Caller-supplied event flags through `build_qrom_from_events`, or local lifetime CSVs for its CLI |
| `helpr_mlqae_importance_boundary.py` | Ideal MLAE/importance query comparison | Local classical query-budget summary for reporting |
| `helpr_mlqae_crossover_refinement.py` | Frozen-grid precision crossover refinement | Local importance-sampling validation summary |

All listed scripts are under `scripts/`. Their `main` entry points retain the
study's expected local directory layout. Dataset-dependent commands cannot
reproduce archived results from this public repository alone. Do not invent
replacement inputs and present the resulting numbers as manuscript results.
Some default scalar targets are study-generated estimates, not upstream data
tables. Full continuous HELPR evaluation and all 460 jointly uncertain HyRAM
inputs are not implemented by the published finite/compressed circuits.

Archived column names containing `splitting` are retained for compatibility.
The relevant oracle-informed comparator is conditional importance sampling,
not a conventional multilevel splitting algorithm. Finite-table values can
also be summed classically; query experiments do not prove an advantage over
reading a known table or an end-to-end hardware speedup.

## Excluded material

This update excludes HyRAM+/HELPR source, API/ASME lookup tables and related
kernel-development scripts, original model parameter collections, actual study
datasets/results, serialized models, PDFs, manuscript drafts, handoff notes,
installed environments, fonts, credentials, and account-specific QPU runners.
These local files are preserved, not deleted. Existing generic IBM examples
retain their explicit submission guard; publication/testing does not submit jobs.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The publication audit checks
tracked paths, reviewed additions, and common secret patterns; it cannot prove
copyright ownership or detect every possible secret. New material still needs
human provenance and rights review before it enters the allowlist.
