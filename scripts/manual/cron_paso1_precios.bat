@echo off
chcp 65001 > nul
REM ============================================================
REM  cron_paso1_precios.bat
REM  Contingencia manual: Paso 1 del pipeline diario.
REM  Descarga precios EOD + futuros + recalcula indicadores
REM  tecnicos para los 199 tickers.
REM
REM  TARGET: PostgreSQL LOCAL (Plan C - local es fuente de verdad
REM  para OHLCV). NO carga .env.local, por lo que DATABASE_URL no
REM  se setea y get_engine() cae a DB_CONFIG (DB_HOST/etc del .env).
REM
REM  Tiempo estimado : ~35-40 minutos
REM  Ventana valida  : despues de las 21:00 UTC (cierre NYSE)
REM  Evitar          : 00:00-12:00 UTC (mantenimiento Yahoo)
REM ============================================================

SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe

REM Posicionarse en la raiz para que config.py:load_dotenv() encuentre .env
cd /d "%ROOT%"

echo.
echo ============================================================
echo   PASO 1 - Precios + Futuros + Indicadores Tecnicos
echo   TARGET: LOCAL  ^|  Fecha : %DATE%  Hora: %TIME%
echo ============================================================
echo.
echo Estado actual de la DB LOCAL:
"%PYTHON%" "%ROOT%scripts\manual\db_status.py" --target local
echo.
echo ATENCION: descarga precios de cierre y recalcula indicadores
echo para los 199 tickers. Tiempo estimado: 35-40 minutos.
echo.
echo Asegurate de correrlo DESPUES de las 21:00 UTC.
echo El Paso 2 (features) debe correr luego de este.
echo.
set /p CONFIRM="Ejecutar Paso 1? (s/n): "
if /i not "%CONFIRM%"=="s" (
    echo Operacion cancelada.
    pause
    exit /b 0
)

echo.
echo Ejecutando Paso 1...
echo ----------------------------------------
"%PYTHON%" "%ROOT%scripts\cron_diario.py" --step precios
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
