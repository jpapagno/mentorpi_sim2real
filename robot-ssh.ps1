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
$connection = Get-Content -LiteralPath (Join-Path $PSScriptRoot '.local/robot-connection.json') -Raw | ConvertFrom-Json
$credential = Import-Clixml -LiteralPath (Join-Path $PSScriptRoot '.local/robot-credentials.xml')
if ($RobotAddress) { $connection.host = $RobotAddress }
$env:MENTORPI_SSH_PASSWORD = $credential.GetNetworkCredential().Password
try {
    $sshArgs = @((Join-Path $PSScriptRoot 'deployment/ssh_client.py'), '--host', $connection.host,
        '--user', $connection.user, '--known-hosts', (Join-Path $PSScriptRoot '.local/robot_known_hosts'),
        '--timeout', $TimeoutSeconds)
    if ($Command) { $sshArgs += @('--command', $Command) }
    if ($Script) { $sshArgs += @('--script', (Resolve-Path -LiteralPath $Script).Path) }
    if ($Upload) { $sshArgs += @('--upload', (Resolve-Path -LiteralPath $Upload).Path, '--destination', $Destination) }
    if ($Download) { $sshArgs += @('--download', $Download, '--destination', $Destination) }
    & (Join-Path $PSScriptRoot '.local/ssh-env/Scripts/python.exe') @sshArgs
    if ($LASTEXITCODE -ne 0) { throw "Robot SSH operation failed (exit $LASTEXITCODE)" }
} finally {
    Remove-Item Env:MENTORPI_SSH_PASSWORD -ErrorAction SilentlyContinue
}
