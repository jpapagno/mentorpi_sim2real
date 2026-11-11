param(
    [string]$Command,
    [string]$Script,
    [string]$Upload,
    [string]$Download,
    [string]$Destination,
    [string]$RobotAddress,
    [int]$TimeoutSeconds = 90
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$connection = Get-Content -LiteralPath (Join-Path $projectRoot '.local/robot-connection.json') -Raw | ConvertFrom-Json
$credential = Import-Clixml -LiteralPath (Join-Path $projectRoot '.local/robot-credentials.xml')
if ($RobotAddress) { $connection.host = $RobotAddress }
$env:MENTORPI_SSH_PASSWORD = $credential.GetNetworkCredential().Password
try {
    $sshArgs = @('-m', 'mentorpi_nav.tools.ssh', '--host', $connection.host,
        '--user', $connection.user, '--known-hosts', (Join-Path $projectRoot '.local/robot_known_hosts'),
        '--timeout', $TimeoutSeconds)
    if ($Command) { $sshArgs += @('--command', $Command) }
    if ($Script) { $sshArgs += @('--script', (Resolve-Path -LiteralPath $Script).Path) }
    if ($Upload) { $sshArgs += @('--upload', (Resolve-Path -LiteralPath $Upload).Path, '--destination', $Destination) }
    if ($Download) { $sshArgs += @('--download', $Download, '--destination', $Destination) }
    Push-Location $projectRoot
    & (Join-Path $projectRoot '.local/ssh-env/Scripts/python.exe') @sshArgs
    if ($LASTEXITCODE -ne 0) { throw "Robot SSH operation failed (exit $LASTEXITCODE)" }
} finally {
    Pop-Location
    Remove-Item Env:MENTORPI_SSH_PASSWORD -ErrorAction SilentlyContinue
}
