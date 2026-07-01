@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM  Portable vessel service-time predictor (rf, xgb, lgbm).
REM  First run builds a local .venv and installs the dependencies (a few minutes,
REM  needs internet). Every run after that starts straight at the prompt.
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

REM ---- pick the input CSV ----------------------------------------------------
echo ==========================================
echo    Vessel service-time predictor
echo ==========================================
echo.
echo CSV files in this folder:
set count=0
for %%F in ("%HERE%*.csv") do call :maybe_add "%%F"

if "!count!"=="0" (
    echo   (none^) - put your vessel CSV in this folder and run again.
    pause & exit /b 1
)
if "!count!"=="1" (
    set "INPUT=!csv[1]!"
) else (
    echo.
    set /p "PICK=Which CSV number? "
    call set "INPUT=%%csv[!PICK!]%%"
)
if not defined INPUT (
    echo Invalid choice.
    pause & exit /b 1
)

REM ---- pick the models -------------------------------------------------------
echo.
echo Using CSV : !INPUT!
echo Models    : rf  xgb  lgbm
echo   ENTER   = rf only ^(the most accurate^)
echo   'all'   = run all three
echo   or list them, e.g.  rf,xgb
set /p "MODELS=Which models? "

for %%A in ("!INPUT!") do set "OUTPUT=%%~dpnA_predictions.csv"

REM Default to rf; 'all' means every shipped model (predict.py's default with no --models).
if "!MODELS!"=="" set "MODELS=rf"
if /I "!MODELS!"=="all" set "MODELS="

echo.
if "!MODELS!"=="" (
    "%VENV_PY%" "%HERE%predict.py" --input "!INPUT!" --output "!OUTPUT!"
) else (
    "%VENV_PY%" "%HERE%predict.py" --input "!INPUT!" --models "!MODELS!" --output "!OUTPUT!"
)

echo.
echo ------------------------------------------------------------
echo   RESULTS SAVED TO:
echo   !OUTPUT!
echo ------------------------------------------------------------
pause
exit /b 0

REM ---- helpers ---------------------------------------------------------------
:find_python
py -3.11 --version >nul 2>&1 && ( set "PY=py -3.11" & exit /b 0 )
py -3 --version    >nul 2>&1 && ( set "PY=py -3"    & exit /b 0 )
python --version   >nul 2>&1 && ( set "PY=python"   & exit /b 0 )
exit /b 1

:maybe_add
REM Add a CSV to the menu, skipping our own *_predictions.csv outputs and the
REM read-only column reference (sample_vessels.csv).
echo %~nx1|findstr /I /E "_predictions.csv" >nul && exit /b
if /I "%~nx1"=="sample_vessels.csv" exit /b
set /a count+=1
set "csv[!count!]=%~f1"
echo   !count!^) %~nx1
exit /b

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
