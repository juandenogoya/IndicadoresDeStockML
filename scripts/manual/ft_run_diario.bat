@echo off
REM ============================================================
REM  ft_run_diario.bat
REM  Ejecuta todos los bots de Forward Testing en secuencia.
REM
REM  Uso: doble click o desde consola
REM       Correr DESPUES de las 13:00 UTC (cuando el cron de GH
REM       Actions ya cargo los datos del cierre del dia anterior)
REM
REM  Para agregar una nueva estrategia:
REM       Copiar el bloque ":: --- BOT N ---" al final de la lista
REM       y cambiar el nombre del script.
REM ============================================================

REM Subir dos niveles desde scripts\manual\ hasta la raiz del proyecto
SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe
SET LOG_DIR=%ROOT%logs\forward_testing

REM Crear carpeta de logs si no existe
IF NOT EXIST "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Fecha y hora para el nombre del log (formato YYYYMMDD_HHMM)
FOR /F "tokens=1-3 delims=/" %%A IN ("%DATE%") DO (
    SET DIA=%%A
    SET MES=%%B
    SET ANIO=%%C
)
FOR /F "tokens=1-2 delims=:" %%A IN ("%TIME: =0%") DO (
    SET HORA=%%A
    SET MIN=%%B
)
SET TIMESTAMP=%ANIO%%MES%%DIA%_%HORA%%MIN%
SET LOGFILE=%LOG_DIR%\ft_%TIMESTAMP%.log

echo.
echo ============================================================
echo  FORWARD TESTING — Ejecucion diaria
echo  Fecha : %DATE%  Hora: %TIME%
echo  Log   : %LOGFILE%
echo ============================================================
echo.

REM Encabezado en el log
echo ============================================================ >> "%LOGFILE%"
echo  FORWARD TESTING - %DATE% %TIME% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"


REM ── BOT 1 — ML Scanner ──────────────────────────────────────
echo [1/4] FT_ML_SCANNER_v1...
echo. >> "%LOGFILE%"
echo --- FT_ML_SCANNER_v1 --- >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\forward_testing\ft_bot_ml_scanner.py" >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ft_bot_ml_scanner.py fallo. Ver log.
) ELSE (
    echo [OK]
)


REM ── BOT 2 — Tecnico global ──────────────────────────────────
echo [2/4] FT_TECH_v1...
echo. >> "%LOGFILE%"
echo --- FT_TECH_v1 --- >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\forward_testing\ft_bot_tecnico.py" >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ft_bot_tecnico.py fallo. Ver log.
) ELSE (
    echo [OK]
)


REM ── BOT 3 — SMC Estructura ──────────────────────────────────
echo [3/4] FT_SMC_v1...
echo. >> "%LOGFILE%"
echo --- FT_SMC_v1 --- >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\forward_testing\ft_bot_smc.py" >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ft_bot_smc.py fallo. Ver log.
) ELSE (
    echo [OK]
)


REM ── BOT 4 — Tecnico Sectorial ───────────────────────────────
echo [4/4] FT_TECH_SECTOR_v1...
echo. >> "%LOGFILE%"
echo --- FT_TECH_SECTOR_v1 --- >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\forward_testing\ft_bot_tech_sectorial.py" >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ft_bot_tech_sectorial.py fallo. Ver log.
) ELSE (
    echo [OK]
)


REM ── AGREGAR NUEVOS BOTS AQUI ─────────────────────────────────
REM Copiar el bloque de arriba y modificar:
REM   - El numero [N/N]
REM   - El nombre de la estrategia
REM   - El nombre del script .py


echo.
echo ============================================================
echo  Completado. Ver detalle en:
echo  %LOGFILE%
echo ============================================================
echo.
pause
