# Wait for ci.yml on this commit; on failure print logs and stop; on success deploy.
# Usage (from repo root):
#   .\deploy\watch-ci-and-deploy.ps1
#   .\deploy\watch-ci-and-deploy.ps1 -Commit abc1234
#   .\deploy\watch-ci-and-deploy.ps1 -SkipDeploy

param(
    [string]$Commit = '',
    [string]$Branch = 'main',
    [int]$StartTimeoutSec = 180,
    [switch]$SkipDeploy
)

$ErrorActionPreference = 'Stop'
$Gh = 'C:\Program Files\GitHub CLI\gh.exe'
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$DeployScript = Join-Path $ScriptDir 'deploy-production.ps1'
$RepoRoot = (git rev-parse --show-toplevel 2>$null)
if (-not $RepoRoot) { throw 'Not inside a git repository.' }
Set-Location $RepoRoot

if (-not (Test-Path $Gh)) {
    $Gh = 'gh'
}

if (-not $Commit) {
    $Commit = (git rev-parse HEAD).Trim()
}
Write-Host "==> CI watch: workflow ci.yml, branch $Branch, commit $Commit"

Write-Host "==> waiting for GitHub Actions run to start (up to ${StartTimeoutSec}s)"
$deadline = (Get-Date).AddSeconds($StartTimeoutSec)
$runId = $null
while ((Get-Date) -lt $deadline) {
    $json = & $Gh run list --workflow ci.yml --branch $Branch --commit $Commit --limit 1 --json databaseId,status,conclusion,url 2>$null
    if ($json) {
        $run = ($json | ConvertFrom-Json)[0]
        if ($run -and $run.databaseId) {
            $runId = $run.databaseId
            Write-Host "==> found run $($run.url)"
            break
        }
    }
    Start-Sleep -Seconds 5
}

if (-not $runId) {
    throw "No ci.yml run found for commit $Commit within ${StartTimeoutSec}s"
}

Write-Host '==> watching CI run (exit on complete)'
& $Gh run watch $runId --exit-status
$watchExit = $LASTEXITCODE

if ($watchExit -ne 0) {
    Write-Host '==> CI FAILED - failed step logs:'
    & $Gh run view $runId --log-failed
    throw 'CI failed - fix ci.yml / tests, push again, then re-run watch-ci-and-deploy.ps1'
}

Write-Host '==> CI passed'

if ($SkipDeploy) {
    Write-Host 'Deploy skipped (-SkipDeploy). Done.'
    exit 0
}

Write-Host '==> deploying to production'
& $DeployScript
if ($LASTEXITCODE -ne 0) { throw 'Production deploy failed' }

Write-Host 'CI passed and production deploy complete.'
