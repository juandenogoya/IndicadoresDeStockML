@echo off
chcp 65001 > nul
REM ============================================================
REM  recovery_incremental.bat
REM  Wrapper Windows del recovery incremental de precios + futuros.
REM
REM  Diferencia con cron_paso1_precios.bat / recovery_pipeline_local.bat:
REM    - NO descarga los 199 tickers ciegamente
REM    - Detecta cuales tienen MAX(fecha) < ultimo dia habil
REM    - Descarga SOLO los pendientes (batches de 3 por defecto)
REM    - Verifica explicitamente y reporta tickers NO completados
REM
REM  Default: target = LOCAL.
REM
REM  Argumentos opcionales:
REM    --dry-run             solo diagnostico
REM    --target railway      apuntar a Railway en lugar de local
REM    --batch-size N        tickers por lote (default 3)
REM    --max-cycles N        reintentos (default 5)
REM    --skip-futuros        solo precios, no futuros
REM    --skip-indicadores    no recalcula indicadores
REM
REM  Ejemplo de uso normal:
REM    scripts\manual\recovery_incremental.bat
REM
REM  Ejemplo dry-run para ver que tickers faltan:
REM    scripts\manual\recovery_incremental.bat --dry-run
REM ============================================================

SET "ROOT=%~dp0..\..\"
SET "PYTHON=%ROOT%venv\Scripts\python.exe"
SET "SCRIPT=%ROOT%scripts\recovery_incremental.py"
SET "LOGDIR=%ROOT%logs"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set DT=%%I
SET "LOGFILE=%LOGDIR%\recovery_incremental_%DT:~0,8%_%DT:~8,4%.log"

echo.
echo ============================================================
echo   RECOVERY INCREMENTAL  (precios + futuros + indicadores)
echo   Log: %LOGFILE%
echo ============================================================
echo.

"%PYTHON%" "%SCRIPT%" %* 2>&1 | tee "%LOGFILE%"
SET EXITCODE=%ERRORLEVEL%

echo.
echo ============================================================
if %EXITCODE%==0 (
    echo   RECOVERY COMPLETO  --  todos los pendientes resueltos
) else (
    echo   RECOVERY PARCIAL   --  ver detalle en log
)
echo   Log completo: %LOGFILE%
echo ============================================================
echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
