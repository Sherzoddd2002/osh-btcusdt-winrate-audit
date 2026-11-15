@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title OSH Quant Lab - Local Demo Engine

echo.
echo  OSH QUANT LAB - BINANCE FUTURES DEMO ENGINE
echo  ------------------------------------------
echo  This server listens only on 127.0.0.1:8765.
echo  Demo endpoint only. Live order endpoint is not available.
echo.

if not exist "requirements.txt" (
  echo [ERROR] requirements.txt was not found.
  echo Run this file from the OSH project folder.
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
      echo [ERROR] Python was not found in PATH. Activate/install Python 3.11 or 3.12, then retry.
      pause
      exit /b 1
    )
    python -m venv "%~dp0.venv"
  ) else (
    py -3 -m venv "%~dp0.venv"
  )
  if errorlevel 1 (
    echo [ERROR] Could not create .venv.
    pause
    exit /b 1
  )
)

"%PY%" -c "import fastapi,uvicorn,httpx,dotenv,keyring,numpy,pandas,sklearn,lightgbm,joblib,sqlalchemy,psycopg2" >nul 2>&1
if errorlevel 1 (
  echo [SETUP] Installing missing Python dependencies from requirements.txt...
  "%PY%" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed. Check internet access and disk space.
    pause
    exit /b 1
  )
)

rem Keep the browser URL and server bind address in sync.
set "OSH_LOCAL_PORT=8765"
rem Never enable Demo orders just by launching the local server.
set "OSH_ENABLE_DEMO_ORDERS=false"

echo.
echo [OK] Starting the local engine at http://127.0.0.1:8765
echo [OK] Keep this window open while using the GitHub Pages terminal.
echo [SAFE] Order sending starts OFF; the website requires a separate confirmation.
echo.
"%PY%" -m demo_executor.app

echo.
echo [STOPPED] The local engine has exited.
pause
