@echo off
chcp 65001 > nul
cd /d %~dp0..\..

echo.
echo ============================================
echo   SCANNER AR -- Indicadores BCBA (IOL)
echo ============================================
echo   Script : scripts/37_scanner_ar.py
echo   Destino: tabla alertas_scanner_ar (DB)
echo   Tickers: todos los activos en activos_ar
echo ============================================
echo.
echo   Ejecutar post-cierre BCBA (despues de 17:00 ART = 20:00 UTC).
echo   Si el mercado esta cerrado los datos de cierre del dia ya estan disponibles.
echo.

set /p TICKER_FILTRO="Tickers especificos (ej: GGAL YPFD) o Enter para todos: "

echo.
echo [1/3] DRY RUN -- validando datos y calculo de indicadores...
echo ----------------------------------------

if "%TICKER_FILTRO%"=="" (
    python scripts/37_scanner_ar.py --dry-run
) else (
    python scripts/37_scanner_ar.py --ticker %TICKER_FILTRO% --dry-run
)

if errorlevel 1 (
    echo.
    echo ERROR durante el DRY RUN. Revisar mensajes de arriba.
    pause
    exit /b 1
)

echo ----------------------------------------
echo.
set /p CONFIRM="Confirmar carga a DB? (s/n): "
if /i not "%CONFIRM%"=="s" (
    echo Operacion cancelada.
    pause
    exit /b 0
)

echo.
echo [2/3] Ejecutando scanner AR y guardando en DB...
echo ----------------------------------------

if "%TICKER_FILTRO%"=="" (
    python scripts/37_scanner_ar.py
) else (
    python scripts/37_scanner_ar.py --ticker %TICKER_FILTRO%
)

if errorlevel 1 (
    echo.
    echo ERROR durante la carga. Revisar mensajes de arriba.
    pause
    exit /b 1
)

echo ----------------------------------------
echo.
echo [3/3] Estado de la tabla alertas_scanner_ar...
echo ----------------------------------------
python scripts/37_scanner_ar.py --status
echo.
pause
