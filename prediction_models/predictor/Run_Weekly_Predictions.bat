@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM  Weekly-schedule batch predictor (rf, xgb, lgbm).
REM  Enriches every file in weekly_schedules\ from vessel history and writes one
REM  <name>_predictions.csv per file into predictions\.
REM  First run builds a local .venv and installs the dependencies (a few minutes,
REM  needs internet). Every run after that goes straight to predicting.
REM  No GPU needed: the tree models run on CPU.
REM ============================================================================

set "HERE=%~dp0"
set "VENV=%HERE%.venv"
set "VENV_PY=%VENV%\Scripts\python.exe"

REM ---- one-time environment setup -------------------------------------------
if not exist "%VENV_PY%" (
    echo First run - setting up the local Python environment...
    call :find_python
    if errorlevel 1 goto :no_python
    echo Using !PY! to create .venv
    !PY! -m venv "%VENV%"
    if errorlevel 1 goto :venv_fail
)

if not exist "%VENV%\.deps_installed" (
    echo Installing dependencies ^(this can take a few minutes^)...
    "%VENV_PY%" -m pip install --upgrade pip
    "%VENV_PY%" -m pip install -r "%HERE%requirements.txt"
    if errorlevel 1 goto :pip_fail
    echo ok> "%VENV%\.deps_installed"
    echo.
    echo Setup complete.
    echo.
)

REM ---- check there is something to predict -----------------------------------
echo ==========================================
echo    Weekly vessel service-time predictor
echo ==========================================
echo.
REM weekly_schedules\ is gitignored (real vessel data), so a fresh clone lacks it - create it.
if not exist "%HERE%weekly_schedules\" mkdir "%HERE%weekly_schedules"
if not exist "%HERE%weekly_schedules\*.xlsx" if not exist "%HERE%weekly_schedules\*.csv" (
    echo No schedule files found in the weekly_schedules\ folder.
    echo Put your weekly .xlsx ^(or .csv^) files in:  %HERE%weekly_schedules\
    echo then run this again.
    pause & exit /b 1
)

echo Predicting every file in weekly_schedules\ ...
echo.
"%VENV_PY%" "%HERE%predict_weeks.py"
if errorlevel 1 goto :run_fail

echo.
echo ------------------------------------------------------------
echo   RESULTS SAVED TO:
echo   %HERE%predictions\
echo ------------------------------------------------------------
pause
exit /b 0

REM ---- helpers ---------------------------------------------------------------
:find_python
py -3.11 --version >nul 2>&1 && ( set "PY=py -3.11" & exit /b 0 )
py -3 --version    >nul 2>&1 && ( set "PY=py -3"    & exit /b 0 )
python --version   >nul 2>&1 && ( set "PY=python"   & exit /b 0 )
exit /b 1

:no_python
echo.
echo ERROR: Python 3 was not found on this computer.
echo Install Python 3.11 from https://www.python.org/downloads/
echo (check "Add python.exe to PATH" during install), then run this again.
pause & exit /b 1

:venv_fail
echo ERROR: could not create the virtual environment. & pause & exit /b 1

:pip_fail
echo ERROR: dependency install failed (check your internet connection). & pause & exit /b 1

:run_fail
echo.
echo ERROR: prediction failed - see the messages above.
pause & exit /b 1
