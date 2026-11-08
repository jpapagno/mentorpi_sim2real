param(
    [string]$LabRoot = "$env:USERPROFILE\Desktop\mentorpi\IsaacLab",
    [Parameter(ValueFromRemainingArguments=$true)] [string[]]$ViewerArgs
)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $LabRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "Python environment not found: $pythonPath" }
Push-Location $PSScriptRoot
try {
    & $pythonPath (Join-Path $PSScriptRoot 'view_navigation.py') @ViewerArgs
    $viewerExitCode = $LASTEXITCODE
} finally { Pop-Location }
exit $viewerExitCode
