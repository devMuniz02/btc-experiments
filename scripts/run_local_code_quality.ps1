param(
    [switch]$SkipPytest
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$PythonCandidates = @()
if ($env:CONDA_PREFIX) {
    $PythonCandidates += (Join-Path $env:CONDA_PREFIX "python.exe")
}
$PythonCandidates += "C:\Users\emman\miniforge3\envs\btc-quant-stream\python.exe"
$PythonCandidates += "python"

$Python = $null
foreach ($Candidate in $PythonCandidates) {
    if ($Candidate -eq "python") {
        $Resolved = Get-Command python -ErrorAction SilentlyContinue
        if ($Resolved) {
            $Python = $Resolved.Source
            break
        }
    } elseif (Test-Path -LiteralPath $Candidate) {
        $Python = $Candidate
        break
    }
}

if (-not $Python) {
    throw "Could not find Python. Activate conda env btc-quant-stream or update scripts/run_local_code_quality.ps1."
}

function Invoke-Checked {
    param(
        [string]$Name,
        [string]$Command,
        [string[]]$Arguments
    )

    Write-Host "==> $Name"
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Invoke-Checked "git diff whitespace check" "git" @("diff", "--check")
Invoke-Checked "compile active source tree" $Python @("-m", "compileall", "-q", "src", "scripts", "tests", "privateCREATEHFREPO.py")
Invoke-Checked "public audit" $Python @("scripts/run_public_audit.py")

if (-not $SkipPytest) {
    $TempRoot = Join-Path $RepoRoot "tmp"
    New-Item -ItemType Directory -Force -Path $TempRoot | Out-Null
    $PytestTemp = Join-Path $TempRoot ("pytest-" + [guid]::NewGuid().ToString("N"))
    $env:TMP = $TempRoot
    $env:TEMP = $TempRoot
    $env:PYTEST_DEBUG_TEMPROOT = $TempRoot
    Invoke-Checked "pytest" $Python @("-m", "pytest", "tests", "--basetemp", $PytestTemp, "-p", "no:cacheprovider")
}

Write-Host "Local code quality checks passed."
