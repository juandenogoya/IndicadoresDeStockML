@echo off
REM ====================================================================
REM build_yaml.bat
REM Llama al MCP para un ticker y genera el YAML pre-cargado para el PDF.
REM
REM Uso:
REM    build_yaml.bat <TICKER>
REM    build_yaml.bat <TICKER> <output.yaml>
REM
REM Output default: scripts\reports\output\<TICKER>_<YYYY-MM-DD>.yaml
REM Despues editar veredicto y conclusion, y correr make_report.bat.
REM ====================================================================

if "%~1"=="" (
    echo.
    echo Uso: build_yaml.bat ^<TICKER^> [output.yaml]
    echo Ej:  build_yaml.bat JPM
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

if "%~2"=="" (
    python scripts\reports\build_yaml.py "%~1"
) else (
    python scripts\reports\build_yaml.py "%~1" -o "%~2"
)
set RC=%errorlevel%

call venv\Scripts\deactivate.bat 2>nul

popd
endlocal & exit /b %RC%
