@echo off
chcp 65001 > nul
REM ============================================================
REM  cron_paso3_scanner.bat
REM  Contingencia manual: Paso 3 del pipeline diario.
REM  Corre el scanner ML sobre los 199 tickers, persiste
REM  alertas en DB y envia resumen a Telegram.
REM
REM  Tiempo estimado : ~60 minutos
REM  Prerequisito    : Paso 1 y Paso 2 deben haber corrido hoy
REM  Nota Telegram   : el mensaje de resumen se envia igual
REM                    que en el cron automatico (comportamiento normal)
REM ============================================================

SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe

REM Cargar DATABASE_URL y otras variables desde .env.local
for /f "usebackq tokens=1,* delims==" %%a in ("%ROOT%.env.local") DO set %%a=%%b

echo.
echo ============================================================
echo   PASO 3 - Scanner ML + Alertas + Telegram
echo   Fecha : %DATE%  Hora: %TIME%
echo ============================================================
echo.
echo Estado actual de la DB:
"%PYTHON%" "%ROOT%scripts\manual\db_status.py"
echo.
echo PREREQUISITO: Paso 1 (precios) y Paso 2 (features) deben
echo haber corrido hoy para que el scanner use datos frescos.
echo.
echo AVISO: al finalizar se envia el resumen a Telegram
echo        (comportamiento identico al cron automatico).
echo.
set /p CONFIRM="Ejecutar Paso 3 (scanner ML)? (s/n): "
if /i not "%CONFIRM%"=="s" (
    echo Operacion cancelada.
    pause
    exit /b 0
)

echo.
echo Ejecutando Paso 3 (esto tarda ~60 minutos)...
echo ----------------------------------------
"%PYTHON%" "%ROOT%scripts\cron_diario.py" --step scanner
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] El Paso 3 termino con errores. Revisar output arriba.
) ELSE (
    echo.
    echo [OK] Paso 3 completado. Resumen enviado a Telegram.
)
echo ----------------------------------------
echo.
echo Estado post-ejecucion:
"%PYTHON%" "%ROOT%scripts\manual\db_status.py"
pause
