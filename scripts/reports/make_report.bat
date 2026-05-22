@echo off
REM ====================================================================
REM make_report.bat
REM Genera un PDF de analisis tecnico desde un YAML.
REM
REM Uso:
REM    make_report.bat <input.yaml>              -> PDF junto al YAML
REM    make_report.bat <input.yaml> <output.pdf> -> PDF en ruta custom
REM
REM Activa el venv del proyecto y corre scripts\reports\make_report.py.
REM ====================================================================

if "%~1"=="" (
    echo.
    echo Uso: make_report.bat ^<input.yaml^> [output.pdf]
    echo Ej:  make_report.bat scripts\reports\examples\ul_2026-05-16.yaml
    echo.
    exit /b 1
)

setlocal

REM Ubicarse en la raiz del proyecto (este .bat vive en scripts\reports\)
pushd "%~dp0..\.."

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: no encuentro el venv en venv\Scripts\activate.bat
    echo Asegurate de estar en la raiz del proyecto y tener el venv creado.
    popd
    endlocal
    exit /b 2
)

call venv\Scripts\activate.bat

if "%~2"=="" (
    python scripts\reports\make_report.py "%~1"
) else (
    python scripts\reports\make_report.py "%~1" -o "%~2"
)
set RC=%errorlevel%

call venv\Scripts\deactivate.bat 2>nul

popd
endlocal & exit /b %RC%
