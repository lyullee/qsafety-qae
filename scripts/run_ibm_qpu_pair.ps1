param(
    [ValidateSet("preview", "run")]
    [string]$Mode = "preview",
    [string[]]$Backends = @("ibm_fez", "ibm_marrakesh"),
    [int]$Shots = 4096,
    [string]$PythonPath = "C:\Users\lyul\AppData\Local\anaconda3\envs\qae_runtime\python.exe",
    [switch]$ConfirmQpu
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Mode -eq "run" -and -not $ConfirmQpu) {
    throw "QPU submission blocked. Add -ConfirmQpu only after reviewing the protocol."
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python executable not found: $PythonPath"
}

$runner = Join-Path $PSScriptRoot "qae_heterogeneous_ibm.py"
foreach ($backend in $Backends) {
    $runnerArgs = @(
        $runner,
        "--mode", $Mode,
        "--backend", $backend,
        "--scenario-qubits", "2", "3", "4",
        "--powers", "0", "1", "2",
        "--shots", $Shots,
        "--optimization", "3",
        "--max-execution-time", "300"
    )
    if ($Mode -eq "run") {
        $runnerArgs += "--confirm-qpu"
    }
    Write-Host "[$backend] mode=$Mode shots=$Shots"
    & $PythonPath @runnerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "IBM run failed for $backend with exit code $LASTEXITCODE"
    }
}

Write-Host "Both backend tasks completed. Rebuild the comparison with:"
Write-Host "$PythonPath scripts\analyze_ibm_qpu_results.py --shots $Shots"
