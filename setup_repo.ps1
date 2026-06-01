param(
    [string]$EnvName = "btc-quant-stream",
    [string]$PythonVersion = "3.11",
    [switch]$ForceRecreate
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path -LiteralPath $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

function Assert-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found in PATH. Install Anaconda or Miniconda, then open a new PowerShell window."
    }
}

Assert-Command -Name "conda"

$CondaEnvs = conda env list
$EnvExists = $false
foreach ($Line in $CondaEnvs) {
    if ($Line -match "^\s*$([regex]::Escape($EnvName))\s") {
        $EnvExists = $true
        break
    }
}

if ($ForceRecreate -and $EnvExists) {
    Write-Host "Removing existing Conda environment: $EnvName"
    conda env remove -n $EnvName -y
    $EnvExists = $false
}

if (-not $EnvExists) {
    Write-Host "Creating Conda environment: $EnvName"
    conda create -n $EnvName "python=$PythonVersion" -y
}
else {
    Write-Host "Using existing Conda environment: $EnvName"
}

Write-Host "Installing Conda dependencies..."
conda install -n $EnvName -c conda-forge `
    numpy `
    pandas `
    pyarrow `
    pyyaml `
    streamlit `
    plotly `
    scikit-learn `
    xgboost `
    joblib `
    requests `
    black `
    flake8 `
    mypy `
    pytest `
    -y

Write-Host "Installing pip dependencies..."
conda run -n $EnvName python -m pip install --upgrade pip
conda run -n $EnvName python -m pip install ccxt

Write-Host "Verifying active project checks..."
conda run -n $EnvName python -m compileall automation_runner.py state_sync.py dashboard.py src
conda run -n $EnvName python automation_runner.py --once
conda run -n $EnvName python state_sync.py --check

Write-Host ""
Write-Host "Setup complete."
Write-Host ""
Write-Host "Activate the environment:"
Write-Host "conda activate $EnvName"
Write-Host ""
Write-Host "Optional local CI/CD setup:"
Write-Host "powershell -ExecutionPolicy Bypass -File .\automation\setup_local_ci_cd.ps1"
Write-Host ""
Write-Host "Optional dashboard:"
Write-Host "streamlit run dashboard.py"
