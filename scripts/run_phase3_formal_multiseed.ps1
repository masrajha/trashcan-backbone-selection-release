[CmdletBinding()]
param(
    [string]$ManifestPath = "Riset\output\research_phase1\visual_review\web_app\rebuilds\candidate_10ab2a514764\visual_review\web_app\rebuilds\candidate_eb4dae55b2c4\visual_review\web_app\frozen\internal_split_manifest_eb4dae55b2c4afb4a2167448ad81b5cd8fed7bc7399ae57a36b27851300c2914.json",
    [string]$FormalTag = (Get-Date -Format "yyyyMMdd_HHmmss"),
    [string[]]$Backbones = @("resnet50", "efficientnet_b0", "convnext_tiny", "swin_t", "pvtv2_b1"),
    [int[]]$Seeds = @(42, 123, 2026),
    [switch]$PlanOnly,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$previousDontWriteBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONDONTWRITEBYTECODE = "1"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $repoRoot ".trashcan-env\python.exe"
$preflightScript = Join-Path $PSScriptRoot "validate_phase3_formal_preflight.py"
$outputRoot = Join-Path $repoRoot "Riset\output\runs"
$researchOutput = Join-Path $repoRoot "Riset\output\research_phase3"
$stateDirectory = Join-Path $outputRoot "_phase3_formal_execution_${FormalTag}"
$statePath = Join-Path $stateDirectory "execution_state.json"
$resolvedManifest = if ([IO.Path]::IsPathRooted($ManifestPath)) { $ManifestPath } else { Join-Path $repoRoot $ManifestPath }

function Get-FailureCategory([int]$ExitCode, [string]$LogPath) {
    $text = if (Test-Path -LiteralPath $LogPath) { Get-Content -Raw -LiteralPath $LogPath } else { "" }
    if ($text -match "(?i)out of memory|cuda oom") { return "resource_infeasible" }
    if ($text -match "(?i)nan|non-finite|not finite") { return "numerical_failure" }
    if ($text -match "(?i)preflight|hash.*(berubah|tidak cocok)|official test|worktree|protocol violation") { return "protocol_violation" }
    if ($text -match "(?i)cuda driver|dataloader worker|disk full|no space|winerror|connection") { return "infrastructure_failure" }
    if ($ExitCode -ne 0) { return "implementation_failure" }
    return $null
}

function Save-State($Rows, [string]$Status, $Preflight) {
    New-Item -ItemType Directory -Path $stateDirectory -Force | Out-Null
    [ordered]@{
        schema_version = 1
        stage = "formal"
        tag = $FormalTag
        status = $Status
        manifest_path = $resolvedManifest
        protocol_sha256 = $Preflight.protocol_sha256
        freeze_record = $Preflight.freeze_record
        freeze_record_sha256 = $Preflight.freeze_record_sha256
        source_git_commit = $Preflight.source_git_commit
        execution_git_commit = $Preflight.head_commit
        backbones = @($Backbones)
        seeds = @($Seeds)
        expected_run_count = $Backbones.Count * $Seeds.Count
        updated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        results = @($Rows)
    } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $statePath -Encoding UTF8
}

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) { throw "Python environment tidak ditemukan: $pythonPath" }
if (-not (Test-Path -LiteralPath $preflightScript -PathType Leaf)) { throw "Preflight script tidak ditemukan: $preflightScript" }
if (-not (Test-Path -LiteralPath $resolvedManifest -PathType Leaf)) { throw "Manifest beku tidak ditemukan: $resolvedManifest" }
if (-not $Backbones -or -not $Seeds) { throw "Backbone dan seed formal tidak boleh kosong." }

$preflightArgs = @($preflightScript)
foreach ($itemBackbone in $Backbones) { $preflightArgs += @("--backbone", $itemBackbone) }
foreach ($itemSeed in $Seeds) { $preflightArgs += @("--seed", "$itemSeed") }
if ($PlanOnly) { $preflightArgs += "--allow-dirty" }

Push-Location $repoRoot
try {
    $preflightJson = & $pythonPath @preflightArgs
    if ($LASTEXITCODE -ne 0) { throw "Formal preflight gagal." }
    $preflight = ($preflightJson -join [Environment]::NewLine) | ConvertFrom-Json

    $plan = @()
    for ($seedIndex = 0; $seedIndex -lt $Seeds.Count; $seedIndex++) {
        $offset = $seedIndex % $Backbones.Count
        $orderedBackbones = @($Backbones[$offset..($Backbones.Count - 1)])
        if ($offset -gt 0) { $orderedBackbones += @($Backbones[0..($offset - 1)]) }
        foreach ($itemBackbone in $orderedBackbones) {
            $plan += [pscustomobject]@{
                order = $plan.Count + 1
                seed = $Seeds[$seedIndex]
                backbone = $itemBackbone
                run_id = "phase3_formal_$($itemBackbone)_seed$($Seeds[$seedIndex])_$FormalTag"
            }
        }
    }

    Write-Host "Formal preflight: PASSED" -ForegroundColor Green
    Write-Host "Freeze SHA-256 : $($preflight.freeze_record_sha256)"
    Write-Host "Protocol SHA-256: $($preflight.protocol_sha256)"
    $plan | Format-Table order, seed, backbone, run_id -AutoSize
    if ($PlanOnly) {
        Write-Host "PlanOnly aktif; tidak ada training atau file state yang ditulis." -ForegroundColor Yellow
        exit 0
    }

    $rows = @()
    if ($Resume -and (Test-Path -LiteralPath $statePath -PathType Leaf)) {
        $existing = Get-Content -Raw -LiteralPath $statePath | ConvertFrom-Json
        if ($existing.freeze_record_sha256 -ne $preflight.freeze_record_sha256) { throw "Resume ditolak: freeze hash berubah." }
        $rows = @($existing.results)
    }
    elseif (Test-Path -LiteralPath $stateDirectory) {
        throw "State sudah ada; gunakan -Resume atau FormalTag baru: $stateDirectory"
    }

    foreach ($item in $plan) {
        $previous = @($rows | Where-Object { $_.run_id -eq $item.run_id } | Select-Object -Last 1)
        if ($previous.Count -gt 0 -and $previous[0].status -eq "completed") {
            Write-Host "SKIP completed: $($item.run_id)" -ForegroundColor DarkGray
            continue
        }
        if ($previous.Count -gt 0) { throw "Resume run gagal harus ditinjau sesuai kebijakan rerun: $($item.run_id)" }

        $runPath = Join-Path $outputRoot $item.run_id
        if (Test-Path -LiteralPath $runPath) { throw "Run directory sudah ada: $runPath" }
        New-Item -ItemType Directory -Path $stateDirectory -Force | Out-Null
        $consoleLog = Join-Path $stateDirectory "$($item.run_id).log"
        $startedAt = (Get-Date).ToUniversalTime().ToString("o")
        Write-Host "`n=== FORMAL $($item.order)/$($plan.Count): $($item.backbone), seed $($item.seed) ===" -ForegroundColor Cyan

        $trainingArgs = @(
            "main.py", "--mode", "train", "--config", "rtx3060",
            "--backbone", $item.backbone, "--phase3-stage", "formal",
            "--seed", "$($item.seed)", "--manifest-path", $resolvedManifest,
            "--run-id", $item.run_id
        )
        # Windows PowerShell 5.1 wraps native stderr (including tqdm progress)
        # as NativeCommandError. Keep stderr in the log without treating normal
        # progress output as a terminating PowerShell exception; the native
        # process exit code remains the source of truth.
        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & $pythonPath @trainingArgs 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $consoleLog
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }
        $finishedAt = (Get-Date).ToUniversalTime().ToString("o")

        $statusPath = Join-Path $runPath "run_status.json"
        $resolvedPath = Join-Path $runPath "resolved_config.json"
        $historyPath = Join-Path $runPath "metrics\history.json"
        $resourcePath = Join-Path $runPath "metrics\resource_usage.json"
        $bestCheckpoint = Join-Path $runPath "checkpoints\best_model.pth"
        $finalCheckpoint = Join-Path $runPath "checkpoints\final_model.pth"
        $criticalFiles = @(
            Get-ChildItem -LiteralPath (Join-Path $runPath "metrics") -Recurse -File -Filter "critical_false_negatives.json" -ErrorAction SilentlyContinue
        )
        $problems = @()
        foreach ($required in @($statusPath, $resolvedPath, $historyPath, $resourcePath, $bestCheckpoint, $finalCheckpoint)) {
            if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { $problems += $required }
        }

        if ($criticalFiles.Count -ne 10) { $problems += "critical_fn_artifact_count=$($criticalFiles.Count)" }

        $resolved = if (Test-Path -LiteralPath $resolvedPath) { Get-Content -Raw -LiteralPath $resolvedPath | ConvertFrom-Json } else { $null }
        $runStatus = if (Test-Path -LiteralPath $statusPath) { Get-Content -Raw -LiteralPath $statusPath | ConvertFrom-Json } else { $null }
        if ($resolved) {
            if ($resolved.phase3_stage -ne "formal") { $problems += "phase3_stage" }
            if ([int]$resolved.num_epochs -ne 30) { $problems += "num_epochs" }
            if ([double]$resolved.data_fraction -ne 1.0) { $problems += "data_fraction" }
            if ([int]$resolved.seed -ne [int]$item.seed) { $problems += "seed" }
            if ($resolved.backbone -ne $item.backbone) { $problems += "backbone" }
            if ($resolved.phase3_training_protocol_sha256 -ne $preflight.protocol_sha256) { $problems += "protocol_sha256" }
            if ($resolved.split_manifest_sha256 -ne $preflight.manifest_sha256) { $problems += "manifest_sha256" }
            if ($resolved.phase3_formal_preflight.freeze_record_sha256 -ne $preflight.freeze_record_sha256) { $problems += "freeze_sha256" }
            if ($resolved.git.dirty -ne $false) { $problems += "git_dirty" }
            if ($resolved.official_test_role -ne "locked_final_test") { $problems += "official_test_role" }
        }
        $completed = $exitCode -eq 0 -and $runStatus -and $runStatus.status -eq "completed" -and $problems.Count -eq 0
        $category = if ($completed) { $null } elseif ($problems.Count -gt 0 -and $exitCode -eq 0) { "protocol_violation" } else { Get-FailureCategory $exitCode $consoleLog }
        $resources = if (Test-Path -LiteralPath $resourcePath) { Get-Content -Raw -LiteralPath $resourcePath | ConvertFrom-Json } else { $null }

        $rows += [pscustomobject]@{
            order = $item.order; backbone = $item.backbone; seed = $item.seed; run_id = $item.run_id
            status = if ($completed) { "completed" } else { "failed" }
            failure_category = $category; exit_code = $exitCode; artifact_problems = @($problems)
            started_at_utc = $startedAt; finished_at_utc = $finishedAt
            total_duration_seconds = if ($resources) { $resources.total_duration_seconds } else { $null }
            peak_allocated_mb = if ($resources) { $resources.max_epoch_peak_allocated_mb } else { $null }
            peak_reserved_mb = if ($resources) { $resources.max_epoch_peak_reserved_mb } else { $null }
            run_directory = $runPath; console_log = $consoleLog
            best_checkpoint = if (Test-Path -LiteralPath $bestCheckpoint) { $bestCheckpoint } else { $null }
            final_checkpoint = if (Test-Path -LiteralPath $finalCheckpoint) { $finalCheckpoint } else { $null }
        }
        Save-State $rows $(if ($completed) { "running" } else { "failed" }) $preflight
        if (-not $completed) { throw "Formal run gagal: $($item.run_id), kategori=$category" }
        [System.GC]::Collect()
        [System.GC]::WaitForPendingFinalizers()
    }

    Save-State $rows "completed" $preflight
    if ($rows.Count -ne $plan.Count -or @($rows | Where-Object status -ne "completed").Count -gt 0) { throw "Matriks formal belum lengkap." }
    New-Item -ItemType Directory -Path $researchOutput -Force | Out-Null
    $csvPath = Join-Path $researchOutput "formal_execution_$FormalTag.csv"
    $jsonPath = Join-Path $researchOutput "formal_execution_$FormalTag.json"
    $rows | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding UTF8
    Copy-Item -LiteralPath $statePath -Destination $jsonPath
    Write-Host "`nMatriks formal selesai: $($rows.Count)/$($plan.Count)" -ForegroundColor Green
    Write-Host "CSV : $csvPath"
    Write-Host "JSON: $jsonPath"
}
finally {
    $env:PYTHONDONTWRITEBYTECODE = $previousDontWriteBytecode
    Pop-Location
}
