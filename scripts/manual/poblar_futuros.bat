@echo off
chcp 65001 > nul
cd /d %~dp0..\..

echo.
echo ============================================
echo   CARGA MANUAL FUTUROS DE INDICES
echo ============================================
echo   Simbolos: ES=F  YM=F  NQ=F  RTY=F
echo ============================================
echo.
echo Estado actual:
python scripts/manual/db_status.py --dias 7
echo.
set /p CONFIRM="Ejecutar actualizacion de futuros? (s/n): "
if /i not "%CONFIRM%"=="s" (
    echo Operacion cancelada.
    pause
    exit /b 0
)

echo.
echo Actualizando futuros...
echo ----------------------------------------
python scripts/34_futuros_indices.py
echo ----------------------------------------
echo.
echo Verificando estado de la DB...
python scripts/manual/db_status.py --dias 7
pause
