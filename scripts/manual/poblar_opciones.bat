@echo off
chcp 65001 > nul
cd /d %~dp0..\..

echo.
echo ============================================
echo   CARGA MANUAL OPCIONES SNAPSHOT
echo ============================================
echo.
set /p FECHA="Fecha a poblar (YYYY-MM-DD, ej: 2026-04-27): "
if "%FECHA%"=="" (
    echo ERROR: fecha requerida.
    pause
    exit /b 1
)

echo.
echo [0/3] Verificando si %FECHA% es dia habil NYSE...
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
echo [1/3] Verificando que fecha tienen los datos en yfinance AHORA...
echo ----------------------------------------
python scripts/manual/check_yfinance_fecha.py
echo ----------------------------------------
echo.
echo IMPORTANTE: si la fecha de yfinance NO coincide con "%FECHA%",
echo cancelar (n) y ejecutar este script en el horario correcto.
echo (Ventana valida: L-V entre 17:00-23:00 ET = 21:00-03:00 UTC)
echo (Evitar: 00:00-09:00 ET = mantenimiento Yahoo Finance)
echo.
set /p CONTINUAR="Continuar con el DRY RUN? (s/n): "
if /i not "%CONTINUAR%"=="s" (
    echo Operacion cancelada.
    pause
    exit /b 0
)

echo.
echo [2/3] DRY RUN para %FECHA% ...
echo ----------------------------------------
python scripts/33_opciones_snapshot.py --fecha %FECHA% --dry-run
echo ----------------------------------------
echo.
set /p CONFIRM="Confirmar carga a DB? (s/n): "
if /i not "%CONFIRM%"=="s" (
    echo Operacion cancelada.
    pause
    exit /b 0
)

echo.
echo [3/3] Cargando datos a DB...
echo ----------------------------------------
python scripts/33_opciones_snapshot.py --fecha %FECHA%
echo ----------------------------------------
echo.
echo Verificando estado de la DB...
python scripts/manual/db_status.py
pause
