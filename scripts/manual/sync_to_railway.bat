@echo off
chcp 65001 > nul
REM ============================================================
REM  sync_to_railway.bat
REM  Sube los datos del PostgreSQL LOCAL a Railway, paso a paso.
REM  Cada paso pide confirmacion antes de continuar.
REM  Al final hace pause para que la ventana no se cierre.
REM
REM  Casos de uso:
REM    - Despues de correr cron_diario.py local (recovery por rate limit)
REM    - Recovery de varios dias en bloque
REM
REM  Orden de pasos:
REM    1. Precios Diarios  (precios_diarios + futuros_diarios)
REM    2. Indicadores      (indicadores_tecnicos + indicadores_tecnicos_futuros)
REM    3. Features         (features_pa + features_ms + features_regimen_macro
REM                         + ticker_zscore_diario)
REM    4. Scanner          (alertas_scanner)
REM ============================================================

SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe
SET SCRIPT=%ROOT%scripts\migrations\sync_local_to_railway.py

echo.
echo ============================================================
echo   SYNC LOCAL -^> RAILWAY  (paso a paso)
echo   Fecha : %DATE%  Hora: %TIME%
echo ============================================================
echo.
echo Cada paso pide confirmacion. Si fallas o cancelas un paso,
echo el script se detiene pero la ventana queda abierta.
echo.

REM ── Paso 1: Precios Diarios ──────────────────────────────────
echo ============================================================
echo   [1/4] PRECIOS DIARIOS
echo   Tablas: precios_diarios + futuros_diarios
echo ============================================================
set /p CONFIRM1="Continuar con paso 1? (s/n): "
if /i not "%CONFIRM1%"=="s" (
    echo Paso 1 cancelado por el usuario.
    goto :final
)
"%PYTHON%" "%SCRIPT%" --paso precios
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] El paso 1 fallo. Abortando sync.
    goto :final
)

REM ── Paso 2: Indicadores ──────────────────────────────────────
echo.
echo ============================================================
echo   [2/4] INDICADORES
echo   Tablas: indicadores_tecnicos + indicadores_tecnicos_futuros
echo ============================================================
set /p CONFIRM2="Continuar con paso 2? (s/n): "
if /i not "%CONFIRM2%"=="s" (
    echo Paso 2 cancelado por el usuario.
    goto :final
)
"%PYTHON%" "%SCRIPT%" --paso indicadores
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] El paso 2 fallo. Abortando sync.
    goto :final
)

REM ── Paso 3: Features ─────────────────────────────────────────
echo.
echo ============================================================
echo   [3/4] FEATURES
echo   Tablas: features_precio_accion + features_market_structure
echo           + features_regimen_macro + ticker_zscore_diario
echo ============================================================
set /p CONFIRM3="Continuar con paso 3? (s/n): "
if /i not "%CONFIRM3%"=="s" (
    echo Paso 3 cancelado por el usuario.
    goto :final
)
"%PYTHON%" "%SCRIPT%" --paso features
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] El paso 3 fallo. Abortando sync.
    goto :final
)

REM ── Paso 4: Scanner ──────────────────────────────────────────
echo.
echo ============================================================
echo   [4/4] SCANNER
echo   Tabla: alertas_scanner
echo ============================================================
set /p CONFIRM4="Continuar con paso 4? (s/n): "
if /i not "%CONFIRM4%"=="s" (
    echo Paso 4 cancelado por el usuario.
    goto :final
)
"%PYTHON%" "%SCRIPT%" --paso scanner
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] El paso 4 fallo.
    goto :final
)

echo.
echo ============================================================
echo   SYNC COMPLETO  (4/4 pasos OK)
echo ============================================================

:final
echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
