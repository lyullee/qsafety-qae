# Security policy

Do not report secrets or credentials in a public issue. Report a suspected
credential leak privately to the repository owner through GitHub's private
vulnerability reporting feature.

IBM Quantum and PyPI credentials are never required by the core package.
`qsafety.ibm` reads an account already configured by `qiskit-ibm-runtime` and
requires an explicit confirmation argument before submitting QPU work.

Only the latest released minor version receives security fixes while the
project remains below version 1.0.
