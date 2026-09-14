@echo off
chcp 65001 > nul
REM ============================================================
REM  sync_opciones_railway_to_local.bat
REM  Baja a la DB LOCAL el CRUDO de opciones US, opciones_snapshot, desde
REM  Railway y despues purga Railway dejando los ultimos 10 dias.
REM
REM  Desde el 6/6/2026 - Tarea 17 - trae SOLO el crudo: las 5 derivadas,
REM  resumen, zscore, sector_zscore, pcr_plazo y sector_pcr_plazo, se
REM  calculan en LOCAL en el paso [0b] de ft_run_diario.bat. Opciones AR
REM  quedan en Railway.
REM
REM  La purga, retencion_opciones_railway.py, solo borra fechas verificadas
REM  como replicadas en local, y solo si el sync termino bien. Es lo que
REM  evita que Railway se frene por limite de consumo: incidente 20/7/2026.
REM  ft_run_diario sincroniza por dentro pero NO purga.
REM
REM  Sync INCREMENTAL e idempotente. No usa Yahoo: DB Railway -> DB local.
REM
REM  Corre a traves de scripts\manual\rutina_diaria.py desde el 13/9/2026:
REM  log en logs\rutina\pasos\sync_AAAAMMDD_HHMM.log y registro en
REM  rutina_corridas. Es el primer paso de rutina_diaria.bat.
REM
REM  OJO al editar: nada de parentesis sin escapar adentro de un bloque
REM  IF ( ... ). Cierran el bloque antes de tiempo y cmd aborta con "no se
REM  esperaba X en este momento": le paso a cron_paso2_features.bat hasta
REM  el 13/9/2026. Por eso estos .bat usan IF de una linea y GOTO, sin
REM  bloques. Editar en modo binario, nunca con sed -i: ver CLAUDE.md.
REM ============================================================

SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe
cd /d "%ROOT%"

echo.
echo ============================================================
echo   SYNC OPCIONES  Railway -^> Local  + purga de Railway
echo   Fecha : %DATE%  Hora: %TIME%
echo ============================================================
echo.

"%PYTHON%" "%ROOT%scripts\manual\rutina_diaria.py" paso sync
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" echo [OK] Sync y purga completados.
if "%RC%"=="2" echo [AVISO] Sync completado, pero la purga de Railway fallo: ver arriba.
if "%RC%"=="1" echo [ERROR] El sync fallo. La purga NO corrio: sin sync verificado no se borra.
echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
exit /b %RC%
