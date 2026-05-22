@echo off
REM ====================================================================
REM make_infografia.bat
REM Genera la infografia PNG (single-image, para X).
REM
REM Uso (lo mas comun -- solo el ticker):
REM    make_infografia.bat BAC
REM        -> consulta el MCP, arma los datos y genera el PNG en
REM           scripts\reports\output\BAC_<FECHA>_ig.png
REM
REM Tambien acepta una ruta YAML ya generada:
REM    make_infografia.bat scripts\reports\output\BAC_2026-05-20.yaml
REM
REM La infografia es 100%% datos (sin LLM, sin narrativa).
REM ====================================================================

if "%~1"=="" (
    echo.
    echo Uso: make_infografia.bat ^<TICKER^>            (ej: make_infografia.bat BAC^)
    echo  o   make_infografia.bat ^<ruta_a_yaml^>
    echo.
    exit /b 1
)

setlocal
pushd "%~dp0..\.."

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: no encuentro el venv en venv\Scripts\activate.bat
    popd
    endlocal
    exit /b 2
)

call venv\Scripts\activate.bat
python scripts\reports\make_infografia.py "%~1"
set RC=%errorlevel%
call venv\Scripts\deactivate.bat 2>nul

popd
endlocal & exit /b %RC%
