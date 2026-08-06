@echo off
chcp 65001 > nul
REM ============================================================
REM  compute_perfiles_carteras.bat
REM  Wrapper Windows del computo de perfiles de riesgo (carteras).
REM
REM  Corre el motor PURO Fase 1 (metricas: ATR% multi-TF + beta +
REM  drawdown) + Fase 2 (clasificacion data-driven por percentil
REM  del universo) sobre los 200 tickers y hace UPSERT en la tabla
REM  perfiles_ticker (LOCAL).
REM
REM  Cadencia MENSUAL: el perfil es una propiedad estable del
REM  instrumento, NO va en el recovery diario.
REM
REM  Default: fecha snapshot = hoy, target = LOCAL (Plan C).
REM
REM  Argumentos opcionales:
REM    --dry-run              calcula y muestra la distribucion, no escribe
REM    --fecha YYYY-MM-DD     fecha del snapshot (default hoy)
REM
REM  Ejemplos:
REM    scripts\manual\compute_perfiles_carteras.bat
REM    scripts\manual\compute_perfiles_carteras.bat --dry-run
REM    scripts\manual\compute_perfiles_carteras.bat --fecha 2026-08-06
REM ============================================================

SET "ROOT=%~dp0..\..\"
SET "PYTHON=%ROOT%venv\Scripts\python.exe"
SET "SCRIPT=%ROOT%scripts\compute_perfiles_carteras.py"
SET "LOGDIR=%ROOT%logs"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set DT=%%I
SET "LOGFILE=%LOGDIR%\compute_perfiles_%DT:~0,8%_%DT:~8,4%.log"

echo.
echo ============================================================
echo   COMPUTE PERFILES CARTERAS  [perfiles_ticker]
echo   Log: %LOGFILE%
echo ============================================================
echo.

"%PYTHON%" "%SCRIPT%" %* 2>&1 | tee "%LOGFILE%"
SET EXITCODE=%ERRORLEVEL%

echo.
echo Exit code: %EXITCODE%
exit /b %EXITCODE%
