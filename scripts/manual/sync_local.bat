@echo off
chcp 65001 > nul
REM ============================================================
REM  sync_local.bat
REM  Sincroniza todas las tablas de la DB local con Railway.
REM
REM  Estrategia por tabla:
REM    - Tablas con fecha  : INCREMENTAL (solo filas nuevas)
REM    - Tablas sin fecha  : REPLACE completo (tablas pequenas/cabecera)
REM    - Si falla una tabla: continua con la siguiente
REM
REM  Uso:
REM    Doble clic en este archivo, o desde CMD.
REM    Parametro opcional --dry-run para simular sin escribir.
REM ============================================================

SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe
SET SCRIPT=%ROOT%scripts\migrations\sync_railway_to_local.py
SET DRY_RUN=%~1

REM Cargar variables de entorno
for /f "usebackq tokens=1,* delims==" %%a in ("%ROOT%.env.local") do set %%a=%%b

echo.
echo ============================================================
echo   SYNC  Railway ^> Local  --  %DATE%  %TIME%
if not "%DRY_RUN%"=="" echo   MODO: DRY-RUN
echo ============================================================
echo.

SET ERRORES=0
SET TOTAL_TABLAS=0

REM ── GRUPO 1: Activos ────────────────────────────────────────
call :sync_tabla activos
REM ── GRUPO 2: Precios y datos de mercado ─────────────────────
call :sync_tabla precios_diarios
call :sync_tabla ticker_zscore_diario
call :sync_tabla futuros_diarios
REM ── GRUPO 3: Features tecnicas ──────────────────────────────
call :sync_tabla features_precio_accion
call :sync_tabla features_market_structure
REM ── GRUPO 4: Alertas scanner ────────────────────────────────
call :sync_tabla alertas_scanner
REM ── GRUPO 5: Opciones (3 sub-tablas) ────────────────────────
call :sync_tabla opciones
REM ── GRUPO 6: Forward Testing ────────────────────────────────
call :sync_tabla forward_testing
REM ── GRUPO 7: Backtesting historico ──────────────────────────
call :sync_tabla bt_historico
REM ── GRUPO 8: Features ML ────────────────────────────────────
call :sync_tabla features_ml

REM ── Resumen final ───────────────────────────────────────────
echo.
echo ============================================================
if %ERRORES% EQU 0 (
    echo   [OK] SYNC COMPLETADO  --  %TOTAL_TABLAS% grupos procesados, 0 errores
) else (
    echo   [!!] SYNC COMPLETADO  --  %ERRORES% grupo^(s^) con error de %TOTAL_TABLAS%
)
echo   Finalizado: %DATE%  %TIME%
echo ============================================================
echo.
pause
exit /b %ERRORES%


REM ── Subrutina: sync_tabla ────────────────────────────────────
:sync_tabla
SET /A TOTAL_TABLAS+=1
echo [%TIME%] Sincronizando grupo: %~1 ...
echo --------------------------------------------------------
"%PYTHON%" "%SCRIPT%" --tabla %~1 %DRY_RUN%
IF %ERRORLEVEL% EQU 0 (
    echo [%TIME%] [OK]   %~1
) ELSE (
    echo [%TIME%] [FAIL] %~1  -- ver detalle arriba
    SET /A ERRORES+=1
)
echo.
exit /b 0
