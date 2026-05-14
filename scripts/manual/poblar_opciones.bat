@echo off
chcp 65001 > nul
cd /d %~dp0..\..

REM ============================================================
REM  poblar_opciones.bat
REM  Carga manual del snapshot de opciones US para una fecha
REM  especifica (caso: cron de Oracle/GH fallaron los 4 intentos).
REM
REM  IMPORTANTE: ejecuta el script UNA SOLA VEZ contra yfinance.
REM  El script tiene idempotency interna: si Railway ya tiene los
REM  datos para esa fecha, sale sin tocar yfinance.
REM
REM  Eliminamos el paso DRY-RUN previo porque doblaba las llamadas
REM  a yfinance sin aportar verificacion real (el dry-run del
REM  script saltea el check de DB que el modo real SI hace).
REM ============================================================

echo.
echo ============================================
echo   CARGA MANUAL OPCIONES SNAPSHOT
echo ============================================
echo.
set /p FECHA="Fecha a poblar (YYYY-MM-DD, ej: 2026-05-13): "
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
    echo El mercado no opero ese dia - no hay datos de opciones.
    echo.
    set /p IGUAL="Continuar de todas formas? (s/n): "
    if /i not "%IGUAL%"=="s" (
        echo Operacion cancelada.
        pause
        exit /b 0
    )
)

echo.
echo [2/3] Confirmacion antes de llamar a yfinance...
echo ----------------------------------------
echo IMPORTANTE: el snapshot SOLO esta disponible mientras el contrato
echo este vigente en yfinance. Una vez que el mercado del dia siguiente
echo abre (13:30 UTC), Yahoo deja de exponer la chain previa y los datos
echo ya no se pueden recuperar.
echo.
echo Ventana valida: L-V entre 20:00 UTC (cierre NYSE) y 13:30 UTC del
echo dia siguiente (apertura). Fuera de esa ventana, "sin opciones"
echo es esperable.
echo.
set /p CONFIRM="Iniciar descarga real (199 tickers, ~30-45 min)? (s/n): "
if /i not "%CONFIRM%"=="s" (
    echo Operacion cancelada.
    pause
    exit /b 0
)

echo.
echo [3/3] Descargando snapshot para %FECHA% (UNA SOLA pasada)...
echo ----------------------------------------
python scripts/33_opciones_snapshot.py --fecha %FECHA%
echo ----------------------------------------
echo.
echo Verificando estado de la DB...
python scripts/manual/db_status.py
pause
