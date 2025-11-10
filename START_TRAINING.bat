@echo off
REM ===================================================================
REM Image Colorization - Windows Setup & Training Launcher
REM ===================================================================

echo.
echo ====================================================================
echo IMAGE COLORIZATION - SETUP AND TRAINING
echo ====================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo [OK] Python is installed
echo.

REM Check if requirements are installed
echo Checking dependencies...
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [INFO] Dependencies not installed. Installing now...
    echo.
    echo This may take 5-10 minutes...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install dependencies
        echo.
        echo Try manually: pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
    echo.
    echo [OK] Dependencies installed successfully
) else (
    echo [OK] Dependencies already installed
)

echo.
echo ====================================================================
echo Running system check...
echo ====================================================================
echo.

REM Run system check
python check_setup.py
if errorlevel 1 (
    echo.
    echo [WARNING] Some checks failed. Review the output above.
    echo.
    echo Do you want to continue anyway? (y/n)
    set /p continue=
    if /i not "%continue%"=="y" (
        echo.
        echo Setup cancelled.
        pause
        exit /b 1
    )
)

echo.
echo ====================================================================
echo Starting training...
echo ====================================================================
echo.
echo This will take approximately 8-10 hours on GPU (RTX 3080)
echo You can press Ctrl+C to stop training at any time
echo.
echo Press any key to start training, or close this window to cancel...
pause >nul

REM Start training
python train.py

echo.
echo ====================================================================
echo Training completed or stopped
echo ====================================================================
echo.
pause
