@echo off
chcp 65001 > nul
REM ============================================================
REM  recovery_pipeline_local.bat
REM  Recovery del pipeline diario completo en Windows (LOCAL DB).
REM  Replica oracle_pipeline_diario.sh pero apuntando al
REM  PostgreSQL local. Caso de uso: cuando el cron de Oracle falla
REM  y hace falta llenar manualmente el(los) dia(s) hábil(es)
REM  faltantes en el LOCAL.
REM
REM  Flujo (secuencial, aborta si un paso falla):
REM    [0] Verifica dia habil NYSE
REM    [1] Precios + indicadores (199 tickers)    ~40 min
REM    [2] Features PA + Market Structure         ~5  min
REM    [3] Scanner ML + alertas + Telegram        ~50 min
REM
REM  Logs : logs/recovery_<YYYYMMDD_HHMM>.log
REM
REM  IMPORTANTE: si Yahoo Finance esta rate-limitando la IP, el
REM  script tiene retries automaticos pero igual puede fallar.
REM  En ese caso: esperar 30-60 min o usar otra red (4G/celular).
REM
REM  Despues del recovery local, subir a Railway con:
REM    scripts\sync_to_railway.bat
REM ============================================================

SET "ROOT=%~dp0..\..\"
SET "PYTHON=%ROOT%venv\Scripts\python.exe"
SET "LOGDIR=%ROOT%logs"

REM TARGET LOCAL: NO se carga .env.local. Asi DATABASE_URL no se setea y
REM get_engine() (src/data/database.py) cae a DB_CONFIG = PostgreSQL local.
REM El cd a la raiz garantiza que config.py:load_dotenv() encuentre .env.
cd /d "%ROOT%"

REM Crear directorio de logs si no existe
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM Timestamp YYYYMMDD_HHMM via WMIC
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set DT=%%I
SET "LOGFILE=%LOGDIR%\recovery_%DT:~0,8%_%DT:~8,4%.log"

echo. >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
echo   RECOVERY PIPELINE LOCAL  %DATE%  %TIME% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

echo.
echo ============================================================
echo   RECOVERY PIPELINE LOCAL  (paso 1 + 2 + 3 secuencial)
echo   Fecha : %DATE%  Hora: %TIME%
echo   Log   : %LOGFILE%
echo ============================================================
echo.
echo Este script ejecuta los 3 pasos del pipeline en secuencia
echo apuntando al PostgreSQL LOCAL.
echo.
echo Duracion estimada: ~95 minutos.
echo.
echo PREREQUISITOS:
echo   - Ningun otro script yfinance corriendo en paralelo (rate limit
echo     es por IP). El yfinance_lock.py aborta si detecta concurrencia.
echo   - Si fallo Yahoo Finance hace poco, esperar 30 min antes.
echo.
set /p CONFIRM="Iniciar recovery? (s/n): "
if /i not "%CONFIRM%"=="s" (
    echo Recovery cancelado por el usuario.
    pause
    exit /b 0
)

REM ── [0] Verificar dia habil ─────────────────────────────────
echo.
echo [%TIME%] Verificando dia habil NYSE...
echo [%TIME%] Verificando dia habil NYSE... >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\manual\check_fecha.py" >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] No es dia habil NYSE. Recovery cancelado.
    echo [%TIME%] No es dia habil. >> "%LOGFILE%"
    goto :final
)
echo [%TIME%] Dia habil confirmado.
echo [%TIME%] Dia habil confirmado. >> "%LOGFILE%"

REM ── [1] Paso 1: Precios + indicadores ────────────────────────
echo.
echo ============================================================
echo   [1/3] Paso 1 -- Precios + indicadores tecnicos
echo         (199 tickers, ~40 min)
echo ============================================================
echo [%TIME%] Paso 1 inicio... >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\cron_diario.py" --step precios >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [%TIME%] [ERROR] Paso 1 fallo. Abortando recovery.
    echo [%TIME%] ERROR Paso 1 -- abortando recovery >> "%LOGFILE%"
    goto :final
)
echo [%TIME%] [OK] Paso 1 completado.
echo [%TIME%] Paso 1 OK. >> "%LOGFILE%"

REM ── [2] Paso 2: Features ─────────────────────────────────────
echo.
echo ============================================================
echo   [2/3] Paso 2 -- Features PA + Market Structure (~5 min)
echo ============================================================
echo [%TIME%] Paso 2 inicio... >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\cron_diario.py" --step features >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [%TIME%] [ERROR] Paso 2 fallo. Abortando recovery.
    echo [%TIME%] ERROR Paso 2 -- abortando recovery >> "%LOGFILE%"
    goto :final
)
echo [%TIME%] [OK] Paso 2 completado.
echo [%TIME%] Paso 2 OK. >> "%LOGFILE%"

REM ── [3] Paso 3: Scanner ML ───────────────────────────────────
echo.
echo ============================================================
echo   [3/3] Paso 3 -- Scanner ML + alertas + Telegram (~50 min)
echo ============================================================
echo [%TIME%] Paso 3 inicio... >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\cron_diario.py" --step scanner >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [%TIME%] [ERROR] Paso 3 fallo.
    echo [%TIME%] ERROR Paso 3 >> "%LOGFILE%"
    goto :final
)
echo [%TIME%] [OK] Paso 3 completado.
echo [%TIME%] Paso 3 OK. >> "%LOGFILE%"

echo.
echo ============================================================
echo   RECOVERY COMPLETO  (3/3 pasos OK)
echo ============================================================
echo [%TIME%] Recovery completado exitosamente. >> "%LOGFILE%"
echo.
echo Proximos pasos sugeridos:
echo   1. Verificar estado: status_local.bat
echo   2. Subir a Railway:  scripts\sync_to_railway.bat

:final
echo.
echo ============================================================
echo Log completo en:
echo   %LOGFILE%
echo ============================================================
echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
