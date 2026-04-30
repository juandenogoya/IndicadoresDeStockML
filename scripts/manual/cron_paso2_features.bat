@echo off
chcp 65001 > nul
REM ============================================================
REM  cron_paso2_features.bat
REM  Contingencia manual: Paso 2 del pipeline diario.
REM  Calcula features de precio-accion y market structure
REM  para los 199 tickers y hace upsert en DB.
REM
REM  Tiempo estimado : ~4-5 minutos
REM  Prerequisito    : Paso 1 (precios) debe haber corrido hoy
REM  El Paso 3 (scanner ML) depende de este paso.
REM ============================================================

SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe

REM Cargar DATABASE_URL y otras variables desde .env.local
for /f "usebackq tokens=1,* delims==" %%a in ("%ROOT%.env.local") DO set %%a=%%b

echo.
echo ============================================================
echo   PASO 2 - Features PA + Market Structure
echo   Fecha : %DATE%  Hora: %TIME%
echo ============================================================
echo.
echo Estado actual de la DB:
"%PYTHON%" "%ROOT%scripts\manual\db_status.py"
echo.
echo PREREQUISITO: el Paso 1 (precios) debe haber corrido hoy.
echo Si los precios no estan actualizados, correr cron_paso1_precios.bat primero.
echo.
set /p CONFIRM="Ejecutar Paso 2? (s/n): "
if /i not "%CONFIRM%"=="s" (
    echo Operacion cancelada.
    pause
    exit /b 0
)

echo.
echo Ejecutando Paso 2...
echo ----------------------------------------
"%PYTHON%" "%ROOT%scripts\cron_diario.py" --step features
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] El Paso 2 termino con errores. NO correr el Paso 3 hasta resolver.
) ELSE (
    echo.
    echo [OK] Paso 2 completado. Podras correr el Paso 3 (scanner) a continuacion.
)
echo ----------------------------------------
echo.
pause
