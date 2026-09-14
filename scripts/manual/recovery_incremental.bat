@echo off
chcp 65001 > nul
REM ============================================================
REM  recovery_incremental.bat
REM  Wrapper Windows del recovery incremental de precios + futuros.
REM
REM  A diferencia de un pipeline ciego de todo el universo:
REM    - Detecta cuales tienen MAX de fecha menor al ultimo cierre
REM    - Descarga SOLO los pendientes, en lotes de 3 por defecto
REM    - Verifica explicitamente y reporta tickers NO completados
REM
REM  Default: target LOCAL, engine yfinance. El Paso 1 de la rutina
REM  diaria es este mismo motor con --engine yahooquery.
REM
REM  Argumentos opcionales, se pasan tal cual:
REM    --dry-run             solo diagnostico
REM    --target railway      apuntar a Railway en lugar de local
REM    --engine yahooquery   cliente alternativo
REM    --batch-size N        tickers por lote
REM    --max-cycles N        reintentos
REM    --skip-futuros        solo precios, no futuros
REM    --skip-indicadores    no recalcula indicadores
REM
REM  Ejemplo dry-run:  scripts\manual\recovery_incremental.bat --dry-run
REM
REM  13/9/2026: corre a traves de scripts\manual\rutina_diaria.py. Antes
REM  usaba python ... pipe tee y leia el ERRORLEVEL del tee, que siempre es
REM  0: decia RECOVERY COMPLETO aunque quedaran tickers pendientes. Ahora
REM  el codigo es el de Python y el log va a logs\rutina\pasos\.
REM
REM  OJO al editar: nada de parentesis sin escapar adentro de un bloque
REM  IF ( ... ). Cierran el bloque antes de tiempo y cmd aborta con "no se
REM  esperaba X en este momento": le paso a cron_paso2_features.bat hasta
REM  el 13/9/2026. Por eso estos .bat usan IF de una linea y GOTO, sin
REM  bloques. Editar en modo binario, nunca con sed -i: ver CLAUDE.md.
REM ============================================================

SET "ROOT=%~dp0..\..\"
SET "PYTHON=%ROOT%venv\Scripts\python.exe"
cd /d "%ROOT%"

echo.
echo ============================================================
echo   RECOVERY INCREMENTAL  - precios + futuros + indicadores
echo ============================================================
echo.

"%PYTHON%" "%ROOT%scripts\manual\rutina_diaria.py" correr --nombre recovery_incremental -- %*
set "EXITCODE=%ERRORLEVEL%"

echo.
echo ============================================================
if "%EXITCODE%"=="0" echo   RECOVERY COMPLETO  --  todos los pendientes resueltos
if "%EXITCODE%"=="2" echo   RECOVERY PARCIAL o con huecos  --  ver el detalle de arriba
if "%EXITCODE%"=="1" echo   RECOVERY CON ERRORES  --  ver el log
echo   Log: logs\rutina\pasos\
echo ============================================================
echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
exit /b %EXITCODE%
