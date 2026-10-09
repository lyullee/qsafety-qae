# Third party software and data notices

The MIT license in this repository applies to its independently authored
software. It does not relicense third-party code, model parameter collections,
standards, publications, artwork, or datasets. Citations do not replace license
compliance or permission to reproduce protected material.

## HyRAM+

HyRAM+ is developed by Sandia National Laboratories. The study audited version
6.1 at commit `b45abf9a6d995951311be6aad836f1874e4d420b`, whose package metadata
specifies **GPL-3.0-only**. Copyright 2015-2025 National Technology & Engineering
Solutions of Sandia, LLC (NTESS); the U.S. Government retains certain rights.

- [Official repository](https://github.com/sandialabs/hyram)
- [License at the audited commit](https://github.com/sandialabs/hyram/blob/b45abf9a6d995951311be6aad836f1874e4d420b/COPYING.txt)
- [Version-specific package metadata](https://github.com/sandialabs/hyram/blob/b45abf9a6d995951311be6aad836f1874e4d420b/pyproject.toml)
- [Technical reference manual](https://doi.org/10.2172/2563814)

The published `hyram_*` research scripts are project implementations of
probability, uncertainty, and estimator calculations. They do not import or
bundle HyRAM+ source code. The original model parameter manifest and default
parameter tables are not included in this code-only update. Users supplying
external inputs must record their provenance and obtain applicable rights.

Running GPL software does not by itself place a research article or numerical
output under the GPL. However, copying, adapting, or combining covered code
and distributing the resulting program requires a separate GPL compliance
review. A separate installation instruction is not a blanket exemption.

## HELPR

HELPR is developed by Sandia National Laboratories. The study audited version
2.1.0 at commit `ae12d20b6608123af41dd9d5f70db6b6547fece5` under **BSD-3-Clause**.
Copyright 2023-2025 National Technology and Engineering Solutions of Sandia,
LLC (NTESS); the U.S. Government retains certain rights.

- [Official repository](https://github.com/sandialabs/helpr)
- [License at the audited commit](https://github.com/sandialabs/helpr/blob/ae12d20b6608123af41dd9d5f70db6b6547fece5/LICENSE.md)
- [Copyright notice](https://github.com/sandialabs/helpr/blob/ae12d20b6608123af41dd9d5f70db6b6547fece5/COPYRIGHT.txt)

This update publishes only independently authored finite-table QROM and ideal
estimator-comparison code. It does not bundle HELPR code, its examples, its
lookup tables, or the study's model-evaluation files. Any future redistribution
of HELPR code must retain the BSD copyright, conditions, and disclaimer, and
must not imply endorsement by NTESS, Sandia, or the contributors.

## API and ASME standards tables

HELPR includes lookup files named `Table_9.3.nc`, `Table_9B10.nc`, `Table_9B12.nc`,
and `Table_9B13.nc`, described upstream as API 579-1 tables. These files and
code directly reproducing standards-table contents are excluded here. The
precise redistribution rights for the standards-derived tables have not been
independently established. The upstream BSD license is not represented as a
blanket permission to republish third-party standards material.

See [API's rights and usage policy](https://www.api.org/products-and-services/standards/rights-and-usage-policy).
Do not add these files, scans of standards, or copied tables without documented
permission or another verified legal basis.

## Dependencies and names

NumPy, pandas, Matplotlib, Qiskit, Qiskit Aer, and IBM Runtime retain their own
licenses. They are installed as dependencies, not vendored in this update.
This notice is not a complete compliance review of a bundled application or
executable. No proprietary fonts, logos, or installers are redistributed.
Mentioning HyRAM+, HELPR, Sandia, API, ASME, or IBM identifies sources and tools;
it does not imply endorsement, certification, or operational safety approval.
