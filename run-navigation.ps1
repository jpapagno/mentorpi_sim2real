param(
    [string]$LabRoot = "$env:USERPROFILE\Desktop\mentorpi\IsaacLab",
    [Parameter(ValueFromRemainingArguments=$true)] [string[]]$TaskArgs
)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $LabRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Isaac Lab Python not found: $pythonPath" }
Push-Location $PSScriptRoot
try {
    & $pythonPath (Join-Path $PSScriptRoot 'run_navigation.py') @TaskArgs
    $taskExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $taskExitCode
