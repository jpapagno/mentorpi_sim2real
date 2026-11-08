param(
    [string]$RunFolder = 'logs/goal_safe_four',
    [int[]]$Iterations = @(300, 600, 999),
    [int]$Episodes = 100,
    [int]$Seed = 314159,
    [string]$ScenarioFile = 'scenarios/original_four.json',
    [string]$LabRoot = "$env:USERPROFILE\Desktop\mentorpi\IsaacLab"
)
$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $LabRoot '.venv\Scripts\python.exe'
Push-Location $PSScriptRoot
try {
    $absoluteRun = (Resolve-Path -LiteralPath $RunFolder).Path
    foreach ($iteration in $Iterations) {
        $name = "model_$iteration.pt"
        $checkpoint = Join-Path $absoluteRun $name
        $watcher = [IO.FileSystemWatcher]::new($absoluteRun, $name)
        try {
            $watcher.EnableRaisingEvents = $true
            while (-not (Test-Path -LiteralPath $checkpoint)) {
                # Native file-change wait, not repeated agent/tool polling.
                $null = $watcher.WaitForChanged([IO.WatcherChangeTypes]::Created, 30000)
            }
        } finally { $watcher.Dispose() }
        $previousLength = -1
        do {
            $info = Get-Item -LiteralPath $checkpoint
            $stable = $info.Length -gt 0 -and $info.Length -eq $previousLength
            $previousLength = $info.Length
            if (-not $stable) { Start-Sleep -Seconds 1 }
        } while (-not $stable)
        & $pythonPath './evaluate_suite.py' --checkpoint $checkpoint --scenario-file $ScenarioFile --seeds $Seed --episodes $Episodes --output "$RunFolder/development_$iteration"
        if ($LASTEXITCODE -ne 0) { throw "Checkpoint evaluation failed: $checkpoint" }
        Write-Output "DEVELOPMENT_CHECK_COMPLETE $iteration"
    }
} finally { Pop-Location }
