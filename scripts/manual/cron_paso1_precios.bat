@echo off
chcp 65001 > nul
REM ============================================================
REM  cron_paso1_precios.bat
REM  Contingencia manual: Paso 1 del pipeline diario.
REM  Descarga precios EOD + futuros + recalcula indicadores
REM  tecnicos para los 199 tickers.
REM
REM  Tiempo estimado : ~35-40 minutos
REM  Ventana valida  : despues de las 21:00 UTC (cierre NYSE)
REM  Evitar          : 00:00-12:00 UTC (mantenimiento Yahoo)
REM ============================================================

SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe

REM Cargar DATABASE_URL y otras variables desde .env.local
for /f "usebackq tokens=1,* delims==" %%a in ("%ROOT%.env.local") DO set %%a=%%b

echo.
echo ============================================================
echo   PASO 1 - Precios + Futuros + Indicadores Tecnicos
echo   Fecha : %DATE%  Hora: %TIME%
echo ============================================================
echo.
echo Estado actual de la DB:
"%PYTHON%" "%ROOT%scripts\manual\db_status.py"
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
echo Estado post-ejecucion:
"%PYTHON%" "%ROOT%scripts\manual\db_status.py"
pause
