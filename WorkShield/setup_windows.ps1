$ErrorActionPreference = "Stop"
Write-Host "============================================"
Write-Host "             WORKSHIELD SETUP"
Write-Host "============================================"
if (Get-Command py -ErrorAction SilentlyContinue) {
    Write-Host "Creating Python 3.11 virtual environment..."
    py -3.11 -m venv .venv
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host "Python launcher not found. Using default Python..."
    python -m venv .venv
} else {
    throw "Python was not found. Install Python 3.11 and try again."
}
$PythonExe = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) { throw "Virtual environment creation failed." }
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -r requirements.txt
Write-Host "Setup completed. Activate with .\.venv\Scripts\Activate.ps1"
Write-Host "Launch with .\.venv\Scripts\python.exe main.py"
