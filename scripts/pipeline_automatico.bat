@echo off
chcp 65001 > nul
REM ============================================================
REM  pipeline_automatico.bat
REM  Pipeline diario completo SIN confirmaciones.
REM  Disenado para Windows Task Scheduler (ejecucion desatendida).
REM
REM  Horario recomendado: L-V 10:00 ART (13:00 UTC)
REM  Duracion estimada  : ~110 minutos
REM  Secuencia          :
REM    0. Verifica dia habil NYSE
REM    1. Limpia cookies yfinance (anti rate-limit)
REM    2. Paso 1 — Precios via fast_info (45 min)
REM    3. Paso 1b — Futuros de indices (1 min)
REM    4. Paso 2 — Features PA + Market Structure (5 min)
REM    5. Paso 3 — Scanner ML + Telegram (60 min)
REM ============================================================

SET ROOT=%~dp0..\
SET PYTHON=%ROOT%venv\Scripts\python.exe
SET LOGDIR=%ROOT%logs\pipeline_auto

REM Cargar DATABASE_URL y otras variables desde .env.local
for /f "usebackq tokens=1,* delims==" %%a in ("%ROOT%.env.local") DO set %%a=%%b

REM Crear directorio de logs si no existe
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM Nombre del log con fecha y hora
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set DT=%%I
SET LOGFILE=%LOGDIR%\pipeline_%DT:~0,8%_%DT:~8,4%.log

echo. >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
echo   PIPELINE AUTOMATICO  %DATE%  %TIME% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
echo. >> "%LOGFILE%"

echo [%TIME%] Iniciando pipeline automatico... >> "%LOGFILE%"

REM ── PASO 0: Verificar dia habil ──────────────────────────────
echo [%TIME%] Verificando dia habil NYSE... >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\manual\check_fecha.py" >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] No es dia habil. Pipeline cancelado. >> "%LOGFILE%"
    exit /b 0
)
echo [%TIME%] Dia habil confirmado. >> "%LOGFILE%"

REM ── PASO 1: Limpiar cookies yfinance ─────────────────────────
echo [%TIME%] Limpiando cookies yfinance... >> "%LOGFILE%"
del /f /q "%LOCALAPPDATA%\py-yfinance\cookies.db" >> "%LOGFILE%" 2>&1
del /f /q "%TEMP%\yfinance.cache" >> "%LOGFILE%" 2>&1
echo [%TIME%] Cookies limpias. >> "%LOGFILE%"

REM ── PASO 2: Precios via fast_info ────────────────────────────
echo [%TIME%] Paso 1 - Precios fast_info (199 tickers)... >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\actualizar_precios_fast_info.py" >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] [ERROR] Paso 1 fallo. Abortando pipeline. >> "%LOGFILE%"
    exit /b 1
)
echo [%TIME%] Paso 1 completado. >> "%LOGFILE%"

REM ── PASO 3: Futuros de indices ───────────────────────────────
echo [%TIME%] Paso 1b - Futuros de indices... >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\34_futuros_indices.py" >> "%LOGFILE%" 2>&1
echo [%TIME%] Futuros completado. >> "%LOGFILE%"

REM ── PASO 4: Features PA + Market Structure ───────────────────
echo [%TIME%] Paso 2 - Features PA + Market Structure... >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\cron_diario.py" --step features >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] [ERROR] Paso 2 fallo. Abortando pipeline. >> "%LOGFILE%"
    exit /b 1
)
echo [%TIME%] Paso 2 completado. >> "%LOGFILE%"

REM ── PASO 5: Scanner ML + Telegram ───────────────────────────
echo [%TIME%] Paso 3 - Scanner ML + Telegram... >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\cron_diario.py" --step scanner >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] [ERROR] Paso 3 fallo. >> "%LOGFILE%"
    exit /b 1
)
echo [%TIME%] Paso 3 completado. >> "%LOGFILE%"

REM ── FIN ──────────────────────────────────────────────────────
echo. >> "%LOGFILE%"
echo [%TIME%] Pipeline completado exitosamente. >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
exit /b 0
