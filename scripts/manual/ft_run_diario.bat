@echo off
REM ============================================================
REM  ft_run_diario.bat
REM  Ejecuta todos los bots de Forward Testing en secuencia.
REM
REM  Uso: doble click o desde consola
REM       Correr DESPUES de que finalice el Oracle pipeline (~00:00 UTC / 21:00 ART)
REM       El pipeline Oracle corre a las 22:00 UTC y tarda ~95 min.
REM
REM  Para agregar una nueva estrategia:
REM       Copiar el bloque "BOT N" al final de la lista
REM       y cambiar el nombre del script.
REM ============================================================

REM Subir dos niveles desde scripts\manual\ hasta la raiz del proyecto
SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe
SET LOG_DIR=%ROOT%logs\forward_testing

REM Crear carpeta de logs si no existe
IF NOT EXIST "%LOG_DIR%" mkdir "%LOG_DIR%"

REM Timestamp YYYYMMDD_HHMM via WMIC (evita el nombre del dia del locale)
FOR /F "tokens=2 delims==" %%A IN ('wmic os get LocalDateTime /value ^| find "="') DO SET _DT=%%A
SET TIMESTAMP=%_DT:~0,8%_%_DT:~8,4%
SET LOGFILE=%LOG_DIR%\ft_%TIMESTAMP%.log

SET ERRORS=0

echo.
echo ============================================================
echo  FORWARD TESTING - Ejecucion diaria
echo  Fecha : %DATE%  Hora: %TIME%
echo  Log   : %LOGFILE%
echo ============================================================
echo.

REM Encabezado en el log
echo ============================================================ >> "%LOGFILE%"
echo  FORWARD TESTING - %DATE% %TIME% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"


REM ── BOT 1 - ML Scanner ──────────────────────────────────────
echo [1/7] FT_ML_SCANNER_v1...
echo. >> "%LOGFILE%"
echo --- FT_ML_SCANNER_v1 --- >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\forward_testing\ft_bot_ml_scanner.py" >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ft_bot_ml_scanner.py fallo. Ver log.
    SET ERRORS=1
) ELSE (
    echo [OK]
)


REM ── BOT 2 - Tecnico global ──────────────────────────────────
echo [2/7] FT_TECH_v1...
echo. >> "%LOGFILE%"
echo --- FT_TECH_v1 --- >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\forward_testing\ft_bot_tecnico.py" >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ft_bot_tecnico.py fallo. Ver log.
    SET ERRORS=1
) ELSE (
    echo [OK]
)


REM ── BOT 3 - SMC Estructura ──────────────────────────────────
echo [3/7] FT_SMC_v1...
echo. >> "%LOGFILE%"
echo --- FT_SMC_v1 --- >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\forward_testing\ft_bot_smc.py" >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ft_bot_smc.py fallo. Ver log.
    SET ERRORS=1
) ELSE (
    echo [OK]
)


REM ── BOT 4 - Tecnico Sectorial ───────────────────────────────
echo [4/7] FT_TECH_SECTOR_v1...
echo. >> "%LOGFILE%"
echo --- FT_TECH_SECTOR_v1 --- >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\forward_testing\ft_bot_tech_sectorial.py" >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ft_bot_tech_sectorial.py fallo. Ver log.
    SET ERRORS=1
) ELSE (
    echo [OK]
)


REM ── BOT 5 - Combo Tecnico + Candle Score ────────────────────
echo [5/7] FT_COMBO_v1...
echo. >> "%LOGFILE%"
echo --- FT_COMBO_v1 --- >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\forward_testing\ft_bot_combo_v1.py" >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ft_bot_combo_v1.py fallo. Ver log.
    SET ERRORS=1
) ELSE (
    echo [OK]
)


REM ── BOT 6 - Tech Sectorial v2 (retencion + rotacion) ────────
echo [6/7] FT_TECH_SECTOR_v2...
echo. >> "%LOGFILE%"
echo --- FT_TECH_SECTOR_v2 --- >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\forward_testing\ft_bot_tech_sectorial_v2.py" >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ft_bot_tech_sectorial_v2.py fallo. Ver log.
    SET ERRORS=1
) ELSE (
    echo [OK]
)


REM ── BOT 7 - SMC v2 (filtro contexto + agotamiento) ──────────
echo [7/7] FT_SMC_v2...
echo. >> "%LOGFILE%"
echo --- FT_SMC_v2 --- >> "%LOGFILE%"
"%PYTHON%" "%ROOT%scripts\forward_testing\ft_bot_smc_v2.py" >> "%LOGFILE%" 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] ft_bot_smc_v2.py fallo. Ver log.
    SET ERRORS=1
) ELSE (
    echo [OK]
)


REM ── AGREGAR NUEVOS BOTS AQUI ─────────────────────────────────
REM Copiar el bloque de arriba y modificar:
REM   - El numero [N/N] en el encabezado y en ambos echo
REM   - El nombre de la estrategia
REM   - El nombre del script .py


echo.
echo ============================================================
echo  Completado. Ver detalle en:
echo  %LOGFILE%
IF %ERRORS% EQU 1 (
    echo  ATENCION: uno o mas bots reportaron error - ver log.
)
echo ============================================================
echo.
pause
