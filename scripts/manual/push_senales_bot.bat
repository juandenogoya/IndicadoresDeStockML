@echo off
REM ============================================================
REM  push_senales_bot.bat
REM  Wrapper standalone del productor de senales_bot_diaria.
REM
REM  Arma la tabla masticada (Plan B, Tarea 16) desde la DB LOCAL
REM  y la UPSERTea a Railway. El bot Alpaca "solo opera": lee esta
REM  tabla, no las crudas.
REM
REM  CUANDO: normalmente se corre solo desde ft_run_diario.bat
REM  (paso final). Este .bat es para re-pushear de forma standalone
REM  SIN re-correr todo Forward Testing (requiere que la data local
REM  del dia ya este cargada: precios/indicadores/scanner/opciones).
REM
REM  Argumentos opcionales:
REM    --dry-run   arma las filas y reporta, no escribe en Railway
REM
REM  Ejemplos:
REM    scripts\manual\push_senales_bot.bat
REM    scripts\manual\push_senales_bot.bat --dry-run
REM ============================================================

SET "ROOT=%~dp0..\..\"
SET "PYTHON=%ROOT%venv\Scripts\python.exe"
SET "SCRIPT=%ROOT%scripts\push_senales_bot.py"
SET "LOGDIR=%ROOT%logs"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM Timestamp del log. Antes salia de `wmic`, que Windows 11 ya no incluye:
REM el for /f no devuelve nada, DT queda SIN DEFINIR y el nombre del log sale
REM literal, con la sintaxis de substring adentro. PowerShell siempre esta; y
REM si aun asi fallara, el fallback garantiza un nombre valido -- perder el
REM log por no poder fecharlo seria el peor de los dos males.
for /f "delims=" %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set "DT=%%I"
if not defined DT set "DT=sin-fecha"
SET "LOGFILE=%LOGDIR%\push_senales_%DT%.log"

echo.
echo ============================================================
echo   PUSH senales_bot_diaria  (local -^> Railway)
echo   Log: %LOGFILE%
echo ============================================================
echo.

"%PYTHON%" "%SCRIPT%" %* > "%LOGFILE%" 2>&1
SET EXITCODE=%ERRORLEVEL%
type "%LOGFILE%"

echo.
echo ============================================================
if %EXITCODE%==0 (
    echo   PUSH COMPLETO
) else (
    echo   PUSH CON ERRORES  --  ver detalle en log
)
echo   Log completo: %LOGFILE%
echo ============================================================
echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
