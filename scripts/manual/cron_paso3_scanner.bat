@echo off
chcp 65001 > nul
REM ============================================================
REM  cron_paso3_scanner.bat
REM  Paso 3 de la rutina diaria: scanner ML sobre el universo, persiste
REM  alertas en DB y envia el resumen del scanner a Telegram.
REM  Prerequisito: Pasos 1 y 2.
REM
REM  TARGET: PostgreSQL LOCAL - Plan C.
REM
REM  Corre a traves de scripts\manual\rutina_diaria.py desde el 13/9/2026:
REM    - log en logs\rutina\pasos\paso3_AAAAMMDD_HHMM.log
REM    - registro de la corrida en la tabla rutina_corridas
REM    - codigo de salida real: 0 OK, 2 con avisos, 1 error
REM  La rutina COMPLETA es rutina_diaria.bat; este .bat rehace solo este paso.
REM
REM  OJO al editar: nada de parentesis sin escapar adentro de un bloque
REM  IF ( ... ). Cierran el bloque antes de tiempo y cmd aborta con "no se
REM  esperaba X en este momento": le paso a cron_paso2_features.bat hasta
REM  el 13/9/2026. Por eso estos .bat usan IF de una linea y GOTO, sin
REM  bloques. Editar en modo binario, nunca con sed -i: ver CLAUDE.md.
REM ============================================================

SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe

REM Posicionarse en la raiz para que config.py encuentre el .env
cd /d "%ROOT%"

echo.
echo ============================================================
echo   PASO 3 - Scanner ML + Alertas + Telegram
echo   TARGET: LOCAL  ^|  Fecha : %DATE%  Hora: %TIME%
echo ============================================================
echo.
echo Estado actual de la DB LOCAL:
"%PYTHON%" "%ROOT%scripts\manual\db_status.py" --target local
echo.
echo PREREQUISITO: Paso 1 y Paso 2 deben haber corrido antes.
echo AVISO: al finalizar se envia el resumen del scanner a Telegram.
echo.
set /p CONFIRM="Ejecutar Paso 3? (s/n): "
if /i not "%CONFIRM%"=="s" goto :cancelado

"%PYTHON%" "%ROOT%scripts\manual\rutina_diaria.py" paso paso3
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" echo [OK] Paso 3 completado. Resumen del scanner enviado a Telegram.
if "%RC%"=="2" echo [AVISO] Terminado con avisos: ver el detalle de arriba.
if "%RC%"=="1" echo [ERROR] El Paso 3 termino con errores. Revisar el log de arriba.
echo.
echo Estado post-ejecucion (LOCAL):
"%PYTHON%" "%ROOT%scripts\manual\db_status.py" --target local
echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
exit /b %RC%

:cancelado
echo Operacion cancelada.
pause
exit /b 0
