@echo off
setlocal enabledelayedexpansion

REM Interactive vessel service-time predictor: pick the CSV in this folder and the
REM models to run, then write <input>_predictions.csv next to it.
REM ponytail: interpreter hardcoded for this machine; override with PORTSDFL_PY if the env moves.
if "%PORTSDFL_PY%"=="" set "PORTSDFL_PY=C:\Users\juanp\anaconda32\envs\portsdfl\python.exe"
set "HERE=%~dp0"

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

echo.
echo Using CSV : !INPUT!
echo Models    : xgb  lgbm  rf  linear  realmlp  tabm  node
set /p "MODELS=Which models? (comma-separated, or ENTER = all): "

for %%A in ("!INPUT!") do set "OUTPUT=%%~dpnA_predictions.csv"

echo.
if "!MODELS!"=="" (
    "%PORTSDFL_PY%" "%HERE%predict.py" --input "!INPUT!" --output "!OUTPUT!"
) else (
    "%PORTSDFL_PY%" "%HERE%predict.py" --input "!INPUT!" --models "!MODELS!" --output "!OUTPUT!"
)

echo.
echo ------------------------------------------------------------
echo   RESULTS SAVED TO:
echo   !OUTPUT!
echo ------------------------------------------------------------
pause
exit /b 0

:maybe_add
REM Add a CSV to the menu, skipping our own *_predictions.csv outputs and the
REM read-only column reference (sample_vessels.csv).
echo %~nx1|findstr /I /E "_predictions.csv" >nul && exit /b
if /I "%~nx1"=="sample_vessels.csv" exit /b
set /a count+=1
set "csv[!count!]=%~f1"
echo   !count!^) %~nx1
exit /b
