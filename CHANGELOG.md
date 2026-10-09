# Changelog

## 0.1.3 - 2026-10-09

- Published selected independently authored research scripts for input
  uncertainty, strong classical baselines, query schedules, finite-table QROM,
  and ideal maximum-likelihood QAE comparisons in the GitHub software archive.
- Added 41 synthetic-input research tests and a tracked-file publication guard.
- Added third-party notices and explicit exclusions for upstream model source,
  standards tables, study datasets, manuscripts, and credentials.
- Added an optional `research` dependency group for repository-based work.
- Kept research scripts and their tests outside the core PyPI distribution;
  use the versioned GitHub/Zenodo archive for the research scripts.
- Added release-time version, publication, and test checks before PyPI upload.
- No changes to the core estimator interface or numerical methods; no QPU
  jobs are submitted by the release checks.

## 0.1.2 - 2026-09-05

- Relicensed the current codebase under the MIT License.
- No changes to the public estimator interface or numerical methods.

## 0.1.1 - 2026-09-05

- Published a patch release to establish archival metadata and a durable versioned software record.
- No changes to the public estimator interface or numerical methods.

## 0.1.0 - 2026-09-05

- Added validated finite-mixture safety probability models.
- Added classical Monte Carlo and ideal ML-QAE estimators at equal query budgets.
- Added optional Qiskit circuit validation, Aer simulation, and guarded IBM QPU execution.
- Added a command-line interface and PHMSA-derived worked example.
