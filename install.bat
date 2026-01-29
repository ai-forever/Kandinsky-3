@echo off

setlocal enabledelayedexpansion
echo.

REM Step 1: Check Python version
echo [1/4] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)
python --version
echo.

REM Step 2: Install dependencies from requirements.txt
echo [2/4] Installing dependencies from requirements.txt...
if exist requirements.txt (
    echo Installing pip, setuptools, and wheel...
    python -m pip install --upgrade pip setuptools wheel
    if errorlevel 1 (
        echo Error: Failed to upgrade pip, setuptools, and wheel
        exit /b 1
    )
    
    echo Installing requirements...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Error: Failed to install requirements
        exit /b 1
    )
    echo Installation completed successfully
) else (
    echo Error: requirements.txt not found
    exit /b 1
)
echo.

REM Step 3: Install the package using setup.py
echo [3/4] Installing kandinsky3 package using setup.py...
if exist setup.py (
    pip install -e .
    if errorlevel 1 (
        echo Error: Failed to install package
        exit /b 1
    )
    echo Package installed successfully
) else (
    echo Error: setup.py not found
    exit /b 1
)
echo.

REM Step 4: Test import
echo [4/4] Testing kandinsky3 import...
python test_import.py
if errorlevel 1 (
    echo.
    echo Error: Installation failed during import test
    exit /b 1
)

echo.
echo Installation completed successfully
exit /b 0
