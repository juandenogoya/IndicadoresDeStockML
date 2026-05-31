@echo off
chcp 65001 > nul
REM ============================================================
REM  make_infografia_fundamental.bat <TICKER>
REM  Genera la infografia fundamental (PNG) de un ticker con
REM  datos reales de las tablas locales fundamentales_*.
REM  Sin LLM, sin proyecciones (foto descriptiva del ultimo Q).
REM
REM  Ejemplo:
REM    scripts\reports\make_infografia_fundamental.bat AAPL
REM ============================================================

if "%~1"=="" (
    echo Uso: make_infografia_fundamental.bat ^<TICKER^>
    exit /b 1
)

SET "ROOT=%~dp0..\..\"
SET "PYTHON=%ROOT%venv\Scripts\python.exe"
SET "SCRIPT=%ROOT%scripts\reports\make_infografia_fundamental.py"

"%PYTHON%" "%SCRIPT%" %*
echo.
echo Presiona una tecla para cerrar...
pause > nul
