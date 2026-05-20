@echo off
chcp 65001 > nul
REM ============================================================
REM  cron_paso1_precios_yq.bat
REM  Paso 1 del pipeline diario usando YAHOOQUERY (no yfinance).
REM
REM  Motivo (20/5/2026): yfinance dejo de responder confiablemente
REM  el endpoint de opciones desde ~18/5/2026. Antes de migrar el
REM  snapshot de opciones, usamos precios diarios como banco de
REM  pruebas del cliente alternativo (yahooquery) sobre el mismo
REM  provider Yahoo.
REM
REM  TARGET: PostgreSQL LOCAL (Plan C - local es fuente de verdad
REM  para OHLCV). NO carga .env.local, asi get_engine() cae a
REM  DB_CONFIG (DB_HOST/etc del .env).
REM
REM  Motor   : recovery_incremental.py --engine yahooquery
REM            (detecta vias MAX(fecha) que tickers faltan y baja
REM            solo los pendientes -- ideal para completar 18/5 y 19/5)
REM
REM  Tiempo  : depende de cuanto este atrasada la DB
REM  Ventana : despues de las 21:00 UTC (cierre NYSE)
REM ============================================================

SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe

REM Posicionarse en la raiz para que config.py:load_dotenv() encuentre .env
cd /d "%ROOT%"

echo.
echo ============================================================
echo   PASO 1 (YAHOOQUERY) - Precios + Futuros + Indicadores
echo   TARGET: LOCAL  ^|  Fecha : %DATE%  Hora: %TIME%
echo ============================================================
echo.
echo Estado actual de la DB LOCAL:
"%PYTHON%" "%ROOT%scripts\manual\db_status.py" --target local
echo.
echo Engine de descarga: YAHOOQUERY (alternativo a yfinance)
echo El motor recovery_incremental.py detecta tickers desactualizados
echo via MAX(fecha) y baja SOLO los pendientes (no toca lo ya cargado).
echo.
echo Asegurate de correrlo DESPUES de las 21:00 UTC.
echo El Paso 2 (features) debe correr luego de este.
echo.
set /p CONFIRM="Ejecutar Paso 1 con yahooquery? (s/n): "
if /i not "%CONFIRM%"=="s" (
    echo Operacion cancelada.
    pause
    exit /b 0
)

echo.
echo Ejecutando Paso 1 con yahooquery...
echo ----------------------------------------
"%PYTHON%" "%ROOT%scripts\recovery_incremental.py" --target local --engine yahooquery
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] El Paso 1 termino con errores. Revisar output arriba.
) ELSE (
    echo.
    echo [OK] Paso 1 completado.
)
echo ----------------------------------------
echo.
echo Estado post-ejecucion (LOCAL):
"%PYTHON%" "%ROOT%scripts\manual\db_status.py" --target local
pause
