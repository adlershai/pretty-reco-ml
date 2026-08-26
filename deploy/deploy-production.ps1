# Deploy pretty-reco-ml on Adler via PuTTY session "step".
# Usage (from repo root): .\deploy\deploy-production.ps1

param(
    [switch]$PipInstall,
    [switch]$SkipPipInstall
)

$ErrorActionPreference = 'Stop'
$Plink = 'C:\Program Files\PuTTY\plink.exe'
$Session = 'step'
$SshUser = 'ubuntu'
$RepoDir = '/home/ubuntu/pretty-reco-ml'

if (-not (Test-Path $Plink)) {
    throw "plink not found at $Plink"
}

function Invoke-Remote([string]$Command) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $output = & $Plink -batch -l $SshUser $Session $Command 2>&1
        if ($LASTEXITCODE -ne 0) {
            if ($output) { Write-Host $output }
            throw "Remote command failed (exit $LASTEXITCODE): $Command"
        }
        return $output
    } finally {
        $ErrorActionPreference = $prev
    }
}

Write-Host "==> git fetch ($RepoDir)"
Invoke-Remote "cd $RepoDir && git fetch origin main" | Out-Null

$depsChangedRaw = Invoke-Remote "cd $RepoDir && git diff HEAD..origin/main --name-only -- requirements.txt"
$depsChanged = ($depsChangedRaw -match 'requirements\.txt')

if ($depsChanged) {
    Write-Host 'requirements.txt changes: Y'
} else {
    Write-Host 'requirements.txt changes: N'
}

$runPip = (-not $SkipPipInstall) -and ($PipInstall -or $depsChanged)

Write-Host "==> git pull ($RepoDir)"
Invoke-Remote "cd $RepoDir && git pull origin main" | ForEach-Object { Write-Host $_ }

if ($runPip) {
    Write-Host '==> uv pip install (CPU torch; skip CUDA wheels)'
    $pipCmd = @'
export PATH="$HOME/.local/bin:$PATH"
cd /home/ubuntu/pretty-reco-ml
grep -vE '^torch==' requirements.txt > /tmp/pretty-reco-req-notorch.txt
uv pip install -p .venv/bin/python -r /tmp/pretty-reco-req-notorch.txt --no-cache
uv pip install -p .venv/bin/python torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu --no-cache
'@
    Invoke-Remote $pipCmd | ForEach-Object { Write-Host $_ }
} else {
    Write-Host '==> pip install skipped'
}

Write-Host '==> restart pretty-reco-ml.service'
Invoke-Remote 'sudo systemctl reset-failed pretty-reco-ml.service 2>/dev/null; sudo systemctl restart pretty-reco-ml.service' | Out-Null

Write-Host '==> wait for /health'
$healthCmd = @'
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  if curl -sf http://127.0.0.1:8000/health >/dev/null; then
    curl -sf http://127.0.0.1:8000/health
    echo
    exit 0
  fi
  sleep 3
done
echo 'health check failed' >&2
journalctl -u pretty-reco-ml.service -n 40 --no-pager >&2
exit 1
'@
$health = Invoke-Remote $healthCmd
Write-Host $health

Write-Host '==> public health'
$public = Invoke-Remote 'curl -sf https://ai.adler-backend.com/health'
Write-Host $public

Write-Host '==> service status'
Write-Host (Invoke-Remote 'systemctl is-active pretty-reco-ml.service')
Write-Host 'Deploy complete.'
