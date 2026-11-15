@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title OSH Quant Lab - HF Model Arena (separate Testnet engine)

echo.
echo  OSH QUANT LAB - HF MODEL ARENA
echo  ----------------------------------------------
echo  Separate engine on 127.0.0.1:8766
echo  The existing H96 C2 terminal stays on 8765, untouched.
echo  Host pinned to testnet.binancefuture.com (NON-production).
echo  Order sending starts OFF. No live endpoint exists.
echo.

if not exist "requirements.txt" (
  echo [ERROR] requirements.txt not found. Run from the OSH project folder.
  pause
  exit /b 1
)

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [SETUP] Creating the project virtual environment...
  where py >nul 2>&1
  if errorlevel 1 (
    where python >nul 2>&1
    if errorlevel 1 (
      echo [ERROR] Python not found in PATH. Install Python 3.11 or 3.12.
      pause
      exit /b 1
    )
    python -m venv "%~dp0.venv"
  ) else (
    py -3 -m venv "%~dp0.venv"
  )
)

"%PY%" -c "import fastapi,uvicorn,httpx,keyring,numpy,pandas,sklearn,lightgbm,joblib,sqlalchemy,psycopg2" >nul 2>&1
if errorlevel 1 (
  echo [SETUP] Installing missing dependencies...
  "%PY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
  )
)

set "OSH_HF_PORT=8766"
echo.
echo [OK] Starting at http://127.0.0.1:8766
echo [OK] Open https://sherzoddd2002.github.io/osh-btcusdt-winrate-audit/hf-arena.html
echo [SAFE] Testnet orders OFF until you tick the confirmation box on the site.
echo.
"%PY%" -m hf_arena.app

echo.
echo [STOPPED] The engine has exited.
pause
