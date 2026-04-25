@echo off
chcp 65001 > nul
cd /d %~dp0..\..

echo.
echo ============================================
echo   EJECUCION MANUAL CRON DIARIO SCANNER
echo ============================================
echo   Pasos: precios + futuros + features +
echo          scanner + verificacion
echo ============================================
echo.
echo Estado actual de la DB:
python scripts/manual/db_status.py
echo.
echo ATENCION: esto ejecuta el pipeline completo.
echo Recomendado solo si el cron del dia no corrio.
echo.
set /p CONFIRM="Confirmar ejecucion? (s/n): "
if /i not "%CONFIRM%"=="s" (
    echo Operacion cancelada.
    pause
    exit /b 0
)

echo.
echo Ejecutando cron_diario.py ...
echo ----------------------------------------
python scripts/cron_diario.py
echo ----------------------------------------
echo.
echo Verificando estado de la DB...
python scripts/manual/db_status.py
pause
