<#!
.SYNOPSIS
Launch one authenticated Windows-bridge / WSL-Gemma T2 group.

.DESCRIPTION
The bridge stays bound to the local WSL subnet only for the duration of this
command. Each invocation requires a fresh output directory because rollout
artifacts are append-only evidence.
#>
[CmdletBinding()]
param(
    [string]$Root,
    [string]$Python,
    [string]$WslRoot = '/mnt/d/FactorioRL-agentic-t2',
    [string]$Output,
    [string]$LogPath,
    [int]$Seed = 20260911,
    [int]$MaxTurns = 32,
    [double]$GameSpeed = 120
)

$ErrorActionPreference = 'Stop'
if (-not $Root) { $Root = Split-Path -Parent $PSScriptRoot }
if (-not $Python) {
    $Python = Join-Path $Root '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {
        $Python = 'D:\FactorioRL\.venv\Scripts\python.exe'
    }
}
if (-not (Test-Path -LiteralPath $Python)) { throw "Python executable not found: $Python" }
if (-not $Output) {
    $Output = "$WslRoot/runtime/agentic-t2/gemma-group-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}
if (-not $LogPath) {
    $LogPath = Join-Path $Root "runtime\agentic-t2\$(Split-Path -Leaf $Output).launcher.log"
}
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
$token = [guid]::NewGuid().ToString('N')
$rule = "FactorioRL T2 $token"
$env:FACTORIORL_BRIDGE_TOKEN = $token
$env:PYTHONPATH = "$Root\src"
$env:FACTORIO_RL_GAME_SPEED = [string]$GameSpeed

New-NetFirewallRule -DisplayName $rule -Direction Inbound -Action Allow -Protocol TCP `
    -LocalPort 8765 -RemoteAddress '172.29.224.0/20' -Profile Any | Out-Null
$bridge = Start-Process $Python -ArgumentList @(
    '-u', 'tools/agentic_bridge.py', '--token-env', 'FACTORIORL_BRIDGE_TOKEN',
    '--host', '0.0.0.0', '--port', '8765', '--game-speed', $GameSpeed
) -WorkingDirectory $Root -WindowStyle Hidden -PassThru

try {
    Start-Sleep -Seconds 8
    $route = & wsl.exe -d Ubuntu-24.04 -- sh -lc 'ip route show default'
    $gateway = ($route -split '\s+')[2]
    if (-not $gateway) { throw 'could not determine the Windows gateway from WSL' }
    & wsl.exe -d Ubuntu-24.04 -- env `
        "FACTORIORL_BRIDGE_URL=http://${gateway}:8765" `
        "FACTORIORL_BRIDGE_TOKEN=$token" `
        "PYTHONPATH=$WslRoot/training" `
        /root/factoriorl-t0/.venv/bin/python "$WslRoot/training/run_gemma_group.py" `
        --output $Output --seed $Seed --max-turns $MaxTurns 2>&1 | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) { throw "Gemma WSL process failed: $LASTEXITCODE" }
    & wsl.exe -d Ubuntu-24.04 -- env "PYTHONPATH=$WslRoot/training" `
        /root/factoriorl-t0/.venv/bin/python "$WslRoot/training/audit_rollout.py" --run $Output 2>&1 |
        Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) { throw "rollout audit failed: $LASTEXITCODE" }
}
finally {
    if (-not $bridge.HasExited) { Stop-Process -Id $bridge.Id -Force }
    Get-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue | Remove-NetFirewallRule
}
