@echo off
chcp 65001 > nul
cd /d %~dp0..\..

REM ============================================================
REM  poblar_opciones_yq.bat
REM  Carga manual del snapshot de opciones US usando YAHOOQUERY
REM  en lugar de yfinance (engine alternativo).
REM
REM  Motivo (20/5/2026): yfinance dejo de responder confiablemente
REM  el endpoint /v7/finance/options desde ~18/5/2026 (verificado
REM  en 5 IPs distintas: Oracle, Azure GH, residencial, 4G).
REM  yahooquery usa otro endpoint sobre el mismo provider Yahoo
REM  y trae el chain ENTERO en UNA call por ticker (vs ~30 de
REM  yfinance). Para 199 tickers: ~199 requests vs ~6.000 -- 30x
REM  menos carga sobre la IP.
REM
REM  TARGET: Railway (Plan C: opciones es la unica tabla en remoto).
REM  El script carga .env.local con override -> DATABASE_URL=Railway.
REM
REM  Idempotency: si Railway ya tiene >= MIN_FILAS para esa fecha,
REM  el script sale sin tocar Yahoo. Seguro re-correr.
REM ============================================================

echo.
echo ============================================
echo   CARGA MANUAL OPCIONES SNAPSHOT (YAHOOQUERY)
echo ============================================
echo.
set /p FECHA="Fecha a poblar (YYYY-MM-DD, ej: 2026-05-20): "
if "%FECHA%"=="" (
    echo ERROR: fecha requerida.
    pause
    exit /b 1
)

echo.
echo [1/3] Verificando si %FECHA% es dia habil NYSE...
echo ----------------------------------------
python scripts/manual/check_fecha.py %FECHA%
if errorlevel 1 (
    echo.
    echo ATENCION: %FECHA% NO es un dia habil NYSE.
    echo.
    set /p IGUAL="Continuar de todas formas? (s/n): "
    if /i not "%IGUAL%"=="s" (
        echo Operacion cancelada.
        pause
        exit /b 0
    )
)

echo.
echo [2/3] Confirmacion antes de llamar a yahooquery...
echo ----------------------------------------
echo IMPORTANTE: el snapshot SOLO esta disponible mientras el contrato
echo este vigente en Yahoo. Una vez que el mercado del dia siguiente
echo abre (13:30 UTC), Yahoo deja de exponer la chain previa y los datos
echo ya no se pueden recuperar.
echo.
echo Ventana valida: L-V entre 20:00 UTC (cierre NYSE) y 13:30 UTC del
echo dia siguiente (apertura).
echo.
echo Engine: YAHOOQUERY (1 call por ticker, ~5-8 min total estimado)
echo.
set /p CONFIRM="Iniciar descarga real (199 tickers con yahooquery)? (s/n): "
if /i not "%CONFIRM%"=="s" (
    echo Operacion cancelada.
    pause
    exit /b 0
)

echo.
echo [3/3] Descargando snapshot CRUDO para %FECHA% con yahooquery...
echo ----------------------------------------
REM --solo-crudo (Tarea 17): captura solo opciones_snapshot (crudo) a Railway.
REM Las derivadas (HV/resumen/zscore/pcr_plazo) se computan en LOCAL despues:
REM   sync_local.bat (o ft_run_diario paso [0]) baja el crudo + compute_opciones_derivadas.py
python scripts/33_opciones_snapshot.py --fecha %FECHA% --engine yahooquery --solo-crudo
echo ----------------------------------------
echo.
echo Verificando estado de la DB (Railway)...
python scripts/manual/db_status.py
echo.
echo RECORDÁ: las derivadas se computan en LOCAL. Tras sincronizar el crudo a local,
echo corré: python scripts/compute_opciones_derivadas.py --fecha %FECHA%
pause
