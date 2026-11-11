param(
    [string]$LabRoot = "$env:USERPROFILE\Desktop\mentorpi\IsaacLab",
    [Parameter(ValueFromRemainingArguments=$true)] [string[]]$ViewerArgs
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pythonPath = Join-Path $LabRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python environment not found: $pythonPath" }
Push-Location $projectRoot
try {
    & $pythonPath -m mentorpi_nav.cli.view @ViewerArgs
    $viewerExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $viewerExitCode
