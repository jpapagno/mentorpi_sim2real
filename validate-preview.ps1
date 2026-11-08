param([int]$Steps = 3, [string]$RunName = (Get-Date -Format 'yyyyMMdd_HHmmss'))
$ErrorActionPreference = 'Stop'
if ($Steps -lt 1) { throw 'Steps must be positive' }
$runFolder = Join-Path $PSScriptRoot "logs\preview_validation\$RunName"
if (Test-Path -LiteralPath $runFolder) { throw "Choose a fresh RunName: $runFolder" }
New-Item -ItemType Directory -Path $runFolder -Force | Out-Null
$consolePath = Join-Path $runFolder 'console.log'
Push-Location $PSScriptRoot
try {
    # Native stderr is captured as evidence; it must not abort the wrapper
    # before artifact validation can report the actual failure.
    $ErrorActionPreference = 'Continue'
    & '.\preview-navigation.bat' --steps $Steps --log-dir $runFolder *> $consolePath
    $previewExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    $resultPath = Join-Path $runFolder 'result.json'
    $failure = Select-String -LiteralPath $consolePath -Pattern 'DLL load failed|Failed to startup python extension|Traceback \(most recent call last\)'
    if ($previewExitCode -ne 0 -or $failure -or -not (Test-Path -LiteralPath $resultPath)) {
        Get-Content -LiteralPath $consolePath -Tail 35
        throw "Preview failed. Exit=$previewExitCode. Log: $consolePath"
    }
    $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    if ($result.steps_executed -ne $Steps) { throw "Preview did not complete $Steps steps. Log: $consolePath" }
    if (-not (Test-Path -LiteralPath (Join-Path $runFolder 'occupancy_binary.png'))) { throw 'Policy grid output missing' }
    Write-Output "PASS: preview initialized, completed $Steps steps, and generated the occupancy map."
    Write-Output "Evidence: $runFolder"
} finally { Pop-Location }
