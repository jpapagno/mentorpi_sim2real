param(
    [string]$Checkpoint = 'logs/goal_baseline/baseline.pt',
    [string]$Output = 'logs/goal_baseline_challenges',
    [int]$Episodes = 100,
    [int]$Seed = 314159,
    [string]$LabRoot = "$env:USERPROFILE\Desktop\mentorpi\IsaacLab"
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pythonPath = Join-Path $LabRoot '.venv\Scripts\python.exe'
Push-Location $projectRoot
try {
    foreach ($scenarioName in @('varied', 'passages', 'long_routes')) {
        & $pythonPath -m mentorpi_nav.cli.evaluate --checkpoint $Checkpoint --scenario-file "scenarios/$scenarioName.json" --episodes $Episodes --seeds $Seed --output "$Output/$scenarioName"
        if ($LASTEXITCODE -ne 0) { throw "Scenario evaluation failed: $scenarioName" }
    }
} finally { Pop-Location }
