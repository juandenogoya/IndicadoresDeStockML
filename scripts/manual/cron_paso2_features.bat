@echo off
chcp 65001 > nul
REM ============================================================
REM  cron_paso2_features.bat
REM  Paso 2 de la rutina diaria: features de precio-accion y market
REM  structure para el universo, upsert en DB. Unos 4-5 minutos.
REM  Prerequisito: Paso 1. El Paso 3 depende de este.
REM
REM  TARGET: PostgreSQL LOCAL - Plan C.
REM
REM  Corre a traves de scripts\manual\rutina_diaria.py desde el 13/9/2026:
REM    - log en logs\rutina\pasos\paso2_AAAAMMDD_HHMM.log
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
echo   PASO 2 - Features PA + Market Structure
echo   TARGET: LOCAL  ^|  Fecha : %DATE%  Hora: %TIME%
echo ============================================================
echo.
echo Estado actual de la DB LOCAL:
"%PYTHON%" "%ROOT%scripts\manual\db_status.py" --target local
echo.
echo PREREQUISITO: el Paso 1 debe haber corrido antes.
echo.
set /p CONFIRM="Ejecutar Paso 2? (s/n): "
if /i not "%CONFIRM%"=="s" goto :cancelado

"%PYTHON%" "%ROOT%scripts\manual\rutina_diaria.py" paso paso2
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" echo [OK] Paso 2 completado. Ya se puede correr el Paso 3, el scanner.
if "%RC%"=="2" echo [AVISO] Terminado con avisos: ver el detalle de arriba.
if "%RC%"=="1" echo [ERROR] El Paso 2 termino con errores. NO correr el Paso 3 hasta resolver.
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
