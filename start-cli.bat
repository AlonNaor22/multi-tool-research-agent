@echo off
title Multi-Tool Research Agent (CLI)
cd /d "%~dp0"

:: Check if venv exists, create if not
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

:: Load .env if it exists (skip comments and blank lines)
if exist ".env" (
    for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do (
        if not "%%a"=="" if not "%%b"=="" set "%%a=%%b"
    )
)

echo.
echo ==========================================
echo   Multi-Tool Research Agent - CLI Mode
echo ==========================================
echo.

python main.py

pause
