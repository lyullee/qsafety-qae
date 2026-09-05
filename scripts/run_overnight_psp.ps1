param(
    [ValidateSet('core', 'extended')]
    [string]$Profile = 'core',
    [switch]$DryRun,
    [switch]$IncludeGpuScaling
)

$ErrorActionPreference = 'Continue'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = 'C:\Users\lyul\AppData\Local\anaconda3\envs\qae_runtime\python.exe'
$GpuPython = 'C:\Users\lyul\AppData\Local\anaconda3\envs\qaoa\python.exe'
$RunStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$LogRoot = Join-Path $ProjectRoot "results\overnight_logs\$RunStamp"
$ManifestPath = Join-Path $LogRoot 'manifest.csv'

if (-not (Test-Path -LiteralPath $Python)) {
    throw "qae_runtime Python was not found: $Python"
}
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null

$Manifest = [System.Collections.Generic.List[object]]::new()

function Invoke-PspJob {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$Executable = $Python
    )
    $Started = Get-Date
    $LogPath = Join-Path $LogRoot "$Name.log"
    Write-Host "`n[$($Started.ToString('s'))] START $Name" -ForegroundColor Cyan
    Write-Host "$Executable $($Arguments -join ' ')"
    if ($DryRun) {
        $ExitCode = 0
        $Status = 'DRY_RUN'
    }
    else {
        & $Executable @Arguments 2>&1 | Tee-Object -FilePath $LogPath
        $ExitCode = $LASTEXITCODE
        $Status = if ($ExitCode -eq 0) { 'SUCCESS' } else { 'FAILED' }
    }
    $Finished = Get-Date
    $Manifest.Add([pscustomobject]@{
        name = $Name
        status = $Status
        exit_code = $ExitCode
        started = $Started.ToString('s')
        finished = $Finished.ToString('s')
        elapsed_minutes = [math]::Round(($Finished - $Started).TotalMinutes, 2)
        log = $LogPath
        command = "$Executable $($Arguments -join ' ')"
    })
    $Manifest | Export-Csv -LiteralPath $ManifestPath -NoTypeInformation -Encoding UTF8
    Write-Host "[$($Finished.ToString('s'))] $Status $Name ($([math]::Round(($Finished-$Started).TotalMinutes, 1)) min)" -ForegroundColor $(if ($ExitCode -eq 0) {'Green'} else {'Red'})
}

Set-Location -LiteralPath $ProjectRoot
Write-Host "PSP overnight profile: $Profile"
Write-Host "Logs and manifest: $LogRoot"
Write-Host 'Actual IBM Quantum jobs are intentionally excluded.'

# Fast reproducibility check and the publication-grade clustered resampling.
Invoke-PspJob -Name '01_classical_pilot' -Arguments @('scripts\psp_classical_pilot.py')
Invoke-PspJob -Name '02_cluster_bootstrap_1000' -Arguments @(
    'scripts\psp_bootstrap_strata.py', '--mode', 'full', '--repeats', '1000',
    '--checkpoint-every', '25', '--resume'
)

# Closed-form/Monte Carlo reference and ideal equal-query scaling. Each K is a
# separate file so an interruption does not discard completed sizes.
Invoke-PspJob -Name '03_classical_reference_full' -Arguments @('scripts\classical_risk_baseline.py', '--mode', 'full')
foreach ($Size in @(2, 4, 8)) {
    Invoke-PspJob -Name "04_query_scaling_k$Size" -Arguments @(
        'scripts\qae_query_scaling.py', '--mode', 'full', '--sizes', "$Size",
        '--repeats', '10000', '--tag', "overnight_k$Size"
    )
}

# Noise batches are split by size and scale. A completed scale remains usable
# even if a later Aer calculation fails. These are the validated pooled-circuit
# feasibility controls; the new heterogeneous circuit is built in the next phase.
$NoisePlan = @(
    [pscustomobject]@{ Size=2; Repeats=30; Scales=@('0','0.1','0.25','0.5','1') },
    [pscustomobject]@{ Size=4; Repeats=30; Scales=@('0','0.1','0.25','0.5','1') },
    [pscustomobject]@{ Size=8; Repeats=20; Scales=@('0','0.025','0.05','0.075','0.1','0.125','0.15','0.2','0.25') }
)
if ($Profile -eq 'extended') {
    $NoisePlan += [pscustomobject]@{ Size=8; Repeats=12; Scales=@('0.3','0.4','0.5','0.75','1') }
}
foreach ($Plan in $NoisePlan) {
    foreach ($Scale in $Plan.Scales) {
        $SafeScale = $Scale.Replace('.', 'p')
        Invoke-PspJob -Name "05_noise_k$($Plan.Size)_s$SafeScale" -Arguments @(
            'scripts\qae_noise_sweep.py', '--mode', 'full', '--sizes', "$($Plan.Size)",
            '--powers', '0', '1', '2', '--scales', "$Scale", '--shots', '4096',
            '--repeats', "$($Plan.Repeats)", '--tag', "overnight_k$($Plan.Size)_s$SafeScale"
        )
    }
}

# One deeper Grover schedule tests the accuracy/depth tradeoff near the useful
# low-noise region without multiplying every noise batch.
foreach ($Scale in @('0','0.05','0.1')) {
    $SafeScale = $Scale.Replace('.', 'p')
    Invoke-PspJob -Name "06_deep_power_k8_s$SafeScale" -Arguments @(
        'scripts\qae_noise_sweep.py', '--mode', 'full', '--sizes', '8',
        '--powers', '0', '1', '2', '4', '--scales', "$Scale", '--shots', '4096',
        '--repeats', '12', '--tag', "overnight_deep_k8_s$SafeScale"
    )
}

# GPU statevector scaling is last: all core statistical and circuit-noise results
# survive if the largest allocation is too slow. K=26 is deliberately excluded.
if ($IncludeGpuScaling -or $Profile -eq 'extended') {
    if (-not (Test-Path -LiteralPath $GpuPython)) {
        throw "CuPy GPU Python was not found: $GpuPython"
    }
    Invoke-PspJob -Name '07_gpu_statevector_k16_20_24' -Arguments @(
        'scripts\qae_risk_demo.py', '--mode', 'full', '--backend', 'gpu',
        '--sizes', '16', '20', '24', '--levels', '2', '3', '4', '5', '6',
        '--shots', '2048', '--grid-points', '500001'
    ) -Executable $GpuPython
}

$Succeeded = @($Manifest | Where-Object status -eq 'SUCCESS').Count
$Failed = @($Manifest | Where-Object status -eq 'FAILED').Count
$Dry = @($Manifest | Where-Object status -eq 'DRY_RUN').Count
Write-Host "`nOvernight batch finished. success=$Succeeded failed=$Failed dry_run=$Dry"
Write-Host "Manifest: $ManifestPath"
if ($Failed -gt 0) {
    Write-Host 'Some jobs failed. Completed outputs were preserved; inspect the corresponding logs.' -ForegroundColor Yellow
}
