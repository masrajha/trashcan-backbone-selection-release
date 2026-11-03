[CmdletBinding()]
param(
    [string]$BenchmarkTag = (Get-Date -Format "yyyyMMdd_HHmmss"),
    [string]$Jobs = "1-15",
    [switch]$AllowDeviceMismatch,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $repoRoot ".trashcan-env\python.exe"
$entryPath = Join-Path $repoRoot "scripts\run_phase3_formal_inference_benchmark.py"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Python environment tidak ditemukan: $pythonPath"
}

$arguments = @(
    $entryPath,
    "--tag", $BenchmarkTag,
    "--jobs", $Jobs
)
if ($AllowDeviceMismatch) { $arguments += "--allow-device-mismatch" }
if ($DryRun) { $arguments += "--dry-run" }

Push-Location $repoRoot
try {
    & $pythonPath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Benchmark inferensi formal gagal dengan exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
