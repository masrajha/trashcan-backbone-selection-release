[CmdletBinding()]
param(
    [string]$Reviewer,
    [switch]$ValidateOnly,
    [switch]$Force,
    [string]$Output = "Riset/output/research_phase3/phase3_formal_protocol_freeze.json"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".trashcan-env/python.exe"
$script = Join-Path $PSScriptRoot "freeze_phase3_formal_protocol.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found: $python"
}

$arguments = @($script, "--output", $Output)
if ($ValidateOnly) {
    $arguments += "--validate-only"
}
else {
    if ([string]::IsNullOrWhiteSpace($Reviewer)) {
        throw "-Reviewer is required unless -ValidateOnly is used."
    }
    $arguments += @("--reviewer", $Reviewer)
}
if ($Force) {
    $arguments += "--force"
}

Push-Location $projectRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Formal protocol freeze failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
