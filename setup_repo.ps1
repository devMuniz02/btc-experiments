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

function Invoke-OptionalCommand {
    param(
        [string]$Description,
        [scriptblock]$Command
    )

    Write-Host $Description
    & $Command
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "$Description failed or is unavailable on this platform. Setup will continue with CPU fallback where needed."
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
    optuna `
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
$PreviousPipRequireVirtualenv = $env:PIP_REQUIRE_VIRTUALENV
$env:PIP_REQUIRE_VIRTUALENV = "false"
conda run -n $EnvName python -m pip --isolated install --upgrade pip
conda run -n $EnvName python -m pip --isolated install torch torchvision --index-url https://download.pytorch.org/whl/cu132
conda run -n $EnvName python -m pip --isolated install ccxt
Invoke-OptionalCommand -Description "Installing optional CuPy CUDA dependencies..." -Command {
    conda run -n $EnvName python -m pip --isolated install cupy-cuda12x
}
Invoke-OptionalCommand -Description "Installing optional RAPIDS cuML CUDA dependencies..." -Command {
    conda install -n $EnvName -c rapidsai -c conda-forge -c nvidia `
        cuml `
        cuda-version=12.0 `
        -y
}
if ($null -eq $PreviousPipRequireVirtualenv) {
    Remove-Item Env:\PIP_REQUIRE_VIRTUALENV -ErrorAction SilentlyContinue
}
else {
    $env:PIP_REQUIRE_VIRTUALENV = $PreviousPipRequireVirtualenv
}

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
