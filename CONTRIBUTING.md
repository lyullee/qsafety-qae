# Contributing

Contributions that improve correctness, validation, documentation, or
reproducibility are welcome. Please open an issue before proposing a breaking
change to the public interface.

## Development setup

```bash
python -m pip install -e ".[dev,qiskit]"
python -m ruff check .
python -m pytest
python -m build
python -m twine check dist/*
```

New behavior should include tests. Do not commit API tokens, raw restricted
data, QPU account files, generated results, or personally identifiable data.

## Scientific claims

State whether a result is exact, sampled, simulated, or obtained from a QPU.
Comparisons must report the resource definition used (for example, oracle
queries, shots, circuit depth, or wall-clock time). Quantum advantage must not
be inferred from ideal query scaling alone.
