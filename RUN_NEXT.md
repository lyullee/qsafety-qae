# Next computations

## Current IBM QPU status and next run

The first publication-scale raw run is complete on both `ibm_fez` and
`ibm_marrakesh` (4096 shots, 4/8/16 strata, powers 0/1/2). Together with the
earlier smoke test, measured QPU use is 27 seconds. Do not run another QPU job
immediately. Follow `IBM_QPU_REPEAT_PROTOCOL_KO.md`: obtain one paired run at each
of two later calibration dates, preferably at least 24 hours apart.

The paired runner defaults to preview and cannot submit without `-Mode run` and
`-ConfirmQpu`. Rebuild comparison tables and the PNG/PDF response figure with
`scripts/analyze_ibm_qpu_results.py` after each paired run.

The input tables are already prepared. Use the bundled or project Python environment with `numpy` and `pandas`.

## PSP classical screening pilot

Run the lightweight data audit, lagged exposure count models, and temporally held-out
conditional-severity model in the stable `qae_runtime` environment:

```powershell
conda run -n qae_runtime python scripts/psp_classical_pilot.py
```

This is a short screening calculation. It writes missingness, category-support,
annual-outcome, frequency-model, and severity-model tables to
`results/psp_pilot/`. It is not final inference: operator-cluster bootstrap and
model-stability checks remain required. The `qaoa` environment currently terminates
inside SciPy optimization on this Windows installation, so use `qae_runtime` for
this script.

## Overnight local-simulator batch

The overnight runner contains no IBM Quantum submission. It checkpoints the
1,000-repeat operator-cluster bootstrap, separates ideal query-scaling outputs by
problem size, and separates Aer noise outputs by size and noise scale. Failed jobs
are logged and do not erase earlier completed outputs.

Recommended run, including the final custom CuPy GPU statevector scaling job:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_overnight_psp.ps1 -Profile core -IncludeGpuScaling
```

Use the longer tail of high-noise scenarios as well:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_overnight_psp.ps1 -Profile extended
```

Logs and a continuously updated `manifest.csv` are written under
`results/overnight_logs/<timestamp>/`. The bootstrap itself supports `--resume` and
checkpoints every 25 repeats. The installed Qiskit Aer build reports CPU-only, so Aer
density-matrix noise jobs use the CPU. The optional final statevector scaling uses
the RTX A4000 through CuPy. Windows may turn off the display, but the computer must
not enter sleep or hibernation while the batch is running.

If only the final GPU job failed with `CuPy is not importable`, do not rerun the
whole overnight batch. The GPU simulator is installed in the separate `qaoa`
environment; rerun only that job with:

```powershell
C:\Users\lyul\AppData\Local\anaconda3\envs\qaoa\python.exe scripts/qae_risk_demo.py --mode full --backend gpu --sizes 16 20 24 --levels 2 3 4 5 6 --shots 2048 --grid-points 500001
```

The overnight runner now selects `qae_runtime` for SciPy/Qiskit work and `qaoa`
for CuPy GPU statevector work automatically.

## Lightweight check

```powershell
python scripts/classical_risk_baseline.py --mode quick
```

This estimates pooled PHMSA rates and runs a small Monte Carlo convergence check for portfolios of 2, 4, 8, and 16 risk units. One risk unit represents 1,000 pipeline mile-years by default.

## Full classical run

Run this locally when final Monte Carlo figures are needed:

```powershell
python scripts/classical_risk_baseline.py --mode full
```

Full mode evaluates 2 to 64 risk units, up to `2^64` binary combinations, Monte Carlo budgets up to 1,048,576, and 10,000 repeated estimates. The code samples the sufficient binomial count rather than materializing every binary scenario, but the reported sample budget remains the number of classical portfolio evaluations.

To change the physical size represented by each risk unit:

```powershell
python scripts/classical_risk_baseline.py --mode full --unit-miles 500
```

## Interpretation

The current model is an intercept-only pooled reference. It is intentionally simple and supplies the exact probability that the later QAE circuit must reproduce. Do not present the Pearson dispersion statistic as a final Poisson-versus-Negative-Binomial model-selection result. A covariate and temporal-bootstrap model will be added after the QAE oracle reproduces this baseline.

Outputs are written to `results/classical_risk/`.

## Data-driven heterogeneous PHMSA QAE

Build and run the 4/8/16-stratum empirical-weight circuits. This includes
distribution state preparation, stratum-specific objective rotations, Grover
validation, QASM export, and transpiled CX/depth accounting:

```powershell
conda run -n qae_runtime python scripts/qae_heterogeneous_risk.py --mode ideal --scenario-qubits 2 3 4 --powers 0 1 2 4 --shots 8192 --output-tag psp_v1
```

The quick repeated noise screens compare a shallower three-power schedule with the
more accurate but deeper four-power schedule:

```powershell
conda run -n qae_runtime python scripts/qae_heterogeneous_noise_sweep.py --mode quick --powers 0 1 2 --tag p012
conda run -n qae_runtime python scripts/qae_heterogeneous_noise_sweep.py --mode quick --powers 0 1 2 4 --tag p0124
```

For publication figures, run the full repeated simulations locally. These are
simulator-only and do not access IBM Quantum:

```powershell
conda run -n qae_runtime python scripts/qae_heterogeneous_noise_sweep.py --mode full --powers 0 1 2 --shots 8192 --repeats 30 --tag p012_full
conda run -n qae_runtime python scripts/qae_heterogeneous_noise_sweep.py --mode full --powers 0 1 2 4 --shots 8192 --repeats 30 --tag p0124_full
```

Outputs are written to `results/qae_heterogeneous/`. The exact weighted mean is
unchanged when 4, 8, or 16 rank strata are used; the larger register refines risk
heterogeneity and increases controlled-rotation cost.

## QAE statevector check

First verify the actual logical gate construction for 2, 4, and 8 risk units:

```powershell
python scripts/qae_gate_circuit_check.py
```

This uses `Ry`, `X`, and logical multi-controlled gates and checks both state
preparation and Grover amplification against the analytic answer. The exported
gate manifest is `results/qae_risk/gate_circuits.json`. Multi-controlled gates
are not yet decomposed into hardware-native gates; that step depends on the
chosen backend and a working Qiskit installation.

CPU quick check:

```powershell
python scripts/qae_risk_demo.py --mode quick --backend cpu
```

GPU quick check in the `qaoa` environment:

```powershell
python scripts/qae_risk_demo.py --mode quick --backend gpu
```

Do not start with the full 26-qubit sweep. Benchmark one size at a time:

```powershell
python scripts/qae_risk_demo.py --mode full --backend gpu --sizes 16
python scripts/qae_risk_demo.py --mode full --backend gpu --sizes 20
python scripts/qae_risk_demo.py --mode full --backend gpu --sizes 24
```

The 26-qubit run can require several GiB of GPU memory and many repeated statevector reflections:

```powershell
python scripts/qae_risk_demo.py --mode full --backend gpu --sizes 26
```

The script applies a 70% free-memory safety limit and skips a size when three complex state buffers would exceed it. Results are written to `results/qae_risk/`. GPU timings describe classical simulation on a GPU, not quantum-hardware execution time.

## Isolated Qiskit circuit environment

The existing `qaoa` environment currently crashes while importing Qiskit on
Windows. Do not modify it. Create a separate environment instead:

```powershell
conda create -n qae_runtime python=3.11 -y
conda run -n qae_runtime python -m pip install -r requirements-qae.txt
```

For the current data-driven hardware workflow, follow `IBM_QPU_GUIDE_KO.md`.
The safe sequence is `qae_heterogeneous_ibm.py --mode check`, then `--mode preview`.
Neither consumes QPU time. Actual submission requires both `--mode run` and the
explicit `--confirm-qpu` flag.

Only build and export the 2, 4, and 8-risk-unit circuits (default and safe):

```powershell
conda run -n qae_runtime python scripts/qae_qiskit_runner.py --mode build
```

Decompose circuits to the Aer native basis and report physical resource growth
without executing any shots:

```powershell
conda run -n qae_runtime python scripts/qae_qiskit_runner.py --mode transpile --sizes 2 4 8 --powers 0 1 2
```

For 4 or more risk qubits, compare the one-clean-ancilla linear-depth MCX
synthesis before running noise or hardware jobs:

```powershell
conda run -n qae_runtime python scripts/qae_qiskit_runner.py --mode transpile --sizes 8 --powers 0 1 2 --mcx-synthesis one-clean
```

Run ideal or custom-noise Aer simulations only when desired:

```powershell
conda run -n qae_runtime python scripts/qae_qiskit_runner.py --mode ideal --shots 4096
conda run -n qae_runtime python scripts/qae_qiskit_runner.py --mode noisy --shots 4096 --p1 0.001 --p2 0.01 --readout-error 0.02
```

Save the IBM Quantum token through hidden input, then submit a deliberately
small first hardware job. The setup restricts automatic selection to the free
Open Plan in us-east. First preview the backend-specific ISA circuits without
submitting a job:

```powershell
conda run -n qae_runtime python scripts/ibm_account_setup.py
conda run -n qae_runtime python scripts/qae_qiskit_runner.py --mode ibm-preview --sizes 2 --powers 0 1 2
```

Only after reviewing the preview, submit a small job. Hardware submission
consumes Open Plan QPU time and requires an explicit confirmation flag:

```powershell
conda run -n qae_runtime python scripts/qae_qiskit_runner.py --mode ibm --sizes 2 --powers 0 1 2 --shots 256 --confirm-hardware
```

After the 2-risk-unit job is checked, explicitly request larger circuits with
`--sizes 4` or `--sizes 8`. The 8-risk-unit logical circuit contains
multi-controlled gates whose native two-qubit decomposition can become deep and
noise-sensitive.

## Repeated noise sensitivity

Run the screening sweep locally after the individual circuit checks. It uses
the validated one-clean-ancilla circuits, five noise scales, three seeds, and
does not access IBM hardware:

```powershell
conda run -n qae_runtime python scripts/qae_noise_sweep.py --mode quick
```

The full sweep is intentionally expensive and should only be run for final
figures:

```powershell
conda run -n qae_runtime python scripts/qae_noise_sweep.py --mode full
```

Focused K=8 threshold refinement (user-run; may take substantially longer than
the quick screen):

```powershell
conda run -n qae_runtime python scripts/qae_noise_sweep.py --mode quick --sizes 8 --scales 0.1 0.15 0.2 0.25 0.3 0.4 0.5 --shots 2048 --repeats 10 --tag k8_threshold
```

## Equal-query classical Monte Carlo versus ideal QAE

This comparison counts state-preparation/oracle-equivalent calls, not wall-clock
time, and therefore must not be presented as measured hardware speedup:

```powershell
conda run -n qae_runtime python scripts/qae_query_scaling.py --mode quick
```

The full repeated experiment is intentionally left for the user:

```powershell
conda run -n qae_runtime python scripts/qae_query_scaling.py --mode full
```
