param(
    [string]$LabRoot = "$env:USERPROFILE\Desktop\mentorpi\IsaacLab",
    [Parameter(ValueFromRemainingArguments=$true)] [string[]]$TaskArgs
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pythonPath = Join-Path $LabRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Isaac Lab Python not found: $pythonPath" }
Push-Location $projectRoot
try {
    & $pythonPath -m mentorpi_nav.cli.simulation @TaskArgs
    $taskExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $taskExitCode
