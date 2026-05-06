@echo off
chcp 65001 > nul
REM ============================================================
REM  pipeline_automatico.bat
REM  Pipeline diario completo SIN confirmaciones.
REM  Disenado para Windows Task Scheduler.
REM
REM  Horario: L-V 10:00 ART (13:00 UTC)
REM  Duracion estimada: ~110 minutos
REM ============================================================

SET "ROOT=%~dp0..\"
SET "PYTHON=%ROOT%venv\Scripts\python.exe"
SET "LOGDIR=%ROOT%logs\pipeline_auto"

REM Cargar DATABASE_URL desde .env.local
for /f "usebackq tokens=1,* delims==" %%a in ("%ROOT%.env.local") DO set "%%a=%%b"

REM Crear directorio de logs si no existe
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM Nombre del log con timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set DT=%%I
SET "LOGFILE=%LOGDIR%\pipeline_%DT:~0,8%_%DT:~8,4%.log"

echo. >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
echo   PIPELINE AUTOMATICO  %DATE%  %TIME% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

REM PASO 0: Verificar dia habil
echo [%TIME%] Verificando dia habil NYSE... >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\manual\check_fecha.py" >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] No es dia habil. Pipeline cancelado. >> "%LOGFILE%"
    exit /b 0
)
echo [%TIME%] Dia habil confirmado. >> "%LOGFILE%"

REM PASO 1: Limpiar cookies yfinance
echo [%TIME%] Limpiando cookies yfinance... >> "%LOGFILE%"
if exist "%LOCALAPPDATA%\py-yfinance\cookies.db" del /f /q "%LOCALAPPDATA%\py-yfinance\cookies.db"
if exist "%TEMP%\yfinance.cache" del /f /q "%TEMP%\yfinance.cache"
echo [%TIME%] Cookies limpias. >> "%LOGFILE%"

REM PASO 2: Precios via fast_info
echo [%TIME%] Paso 1 - Precios fast_info (199 tickers)... >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\actualizar_precios_fast_info.py" >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] ERROR en Paso 1. Abortando. >> "%LOGFILE%"
    exit /b 1
)
echo [%TIME%] Paso 1 OK. >> "%LOGFILE%"

REM PASO 3: Futuros de indices
echo [%TIME%] Paso 1b - Futuros... >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\34_futuros_indices.py" >> "%LOGFILE%" 2>&1
echo [%TIME%] Futuros OK. >> "%LOGFILE%"

REM PASO 4: Features
echo [%TIME%] Paso 2 - Features PA + Market Structure... >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\cron_diario.py" --step features >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] ERROR en Paso 2. Abortando. >> "%LOGFILE%"
    exit /b 1
)
echo [%TIME%] Paso 2 OK. >> "%LOGFILE%"

REM PASO 5: Scanner ML
echo [%TIME%] Paso 3 - Scanner ML + Telegram... >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\cron_diario.py" --step scanner >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [%TIME%] ERROR en Paso 3. >> "%LOGFILE%"
    exit /b 1
)
echo [%TIME%] Paso 3 OK. >> "%LOGFILE%"

echo [%TIME%] Pipeline completado exitosamente. >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
exit /b 0
