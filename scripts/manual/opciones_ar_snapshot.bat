@echo off
chcp 65001 > nul
cd /d %~dp0..\..

echo.
echo ============================================
echo   OPCIONES AR SNAPSHOT -- Greeks BCBA (IOL)
echo ============================================
echo   Script : scripts/38_opciones_ar_snapshot.py
echo   Destino: tabla opciones_ar_gregas (DB)
echo   Greeks : Delta, Gamma, Theta, Vega, Rho + IV
echo ============================================
echo.
echo   IMPORTANTE: los Greeks solo son validos DURANTE el horario BCBA.
echo   Horario valido: L-V 10:30-17:00 ART  (= 13:30-20:00 UTC).
echo   Fuera de ese horario la mayoria de los Greeks seran null.
echo.
echo   Hora UTC actual (aprox): usar 'date /t' + 'time /t' como referencia.
echo   Fecha: %DATE%   Hora: %TIME%
echo.

set /p HORA_OK="El mercado BCBA esta abierto ahora? (s/n/f=forzar igual): "

set FORCE_FLAG=
if /i "%HORA_OK%"=="f" (
    set FORCE_FLAG=--force
    echo.
    echo Continuando con --force (Greeks podrian ser null si el mercado esta cerrado).
)
if /i "%HORA_OK%"=="n" (
    echo.
    echo Operacion cancelada. Ejecutar solo en horario BCBA (L-V 10:30-17:00 ART).
    echo Para forzar de todas formas, responder "f" en lugar de "n".
    pause
    exit /b 0
)

echo.
set /p TICKER_FILTRO="Tickers especificos (ej: GGAL YPFD) o Enter para todos: "

echo.
echo [1/3] DRY RUN -- descargando chain y filtrando strikes liquidos...
echo ----------------------------------------

if "%TICKER_FILTRO%"=="" (
    python scripts/38_opciones_ar_snapshot.py --dry-run %FORCE_FLAG%
) else (
    python scripts/38_opciones_ar_snapshot.py --ticker %TICKER_FILTRO% --dry-run %FORCE_FLAG%
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
echo [2/3] Guardando Greeks en opciones_ar_gregas...
echo ----------------------------------------

if "%TICKER_FILTRO%"=="" (
    python scripts/38_opciones_ar_snapshot.py %FORCE_FLAG%
) else (
    python scripts/38_opciones_ar_snapshot.py --ticker %TICKER_FILTRO% %FORCE_FLAG%
)

if errorlevel 1 (
    echo.
    echo ERROR durante la carga. Revisar mensajes de arriba.
    pause
    exit /b 1
)

echo ----------------------------------------
echo.
echo [3/3] Estado de la tabla opciones_ar_gregas...
echo ----------------------------------------
python scripts/38_opciones_ar_snapshot.py --status
echo.
pause
