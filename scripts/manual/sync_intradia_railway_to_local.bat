@echo off
chcp 65001 > nul
REM ============================================================
REM  sync_intradia_railway_to_local.bat
REM  Baja a la DB LOCAL las barras intradiarias, precios_intradia_30m,
REM  desde Railway y despues purga Railway dejando las ultimas 30 ruedas.
REM  Solo borra ruedas verificadas como replicadas en local.
REM
REM  Linea intradiaria EN PRUEBA en la rama main-y9tr06: se corre a mano,
REM  todavia no es un paso de rutina_diaria. Ver docs\intradia.md.
REM
REM  Sin tee: el codigo de salida que se informa es el real de Python.
REM  OJO al editar: sin bloques IF ( ... ) y en modo binario, ver CLAUDE.md.
REM ============================================================

SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe
cd /d "%ROOT%"

echo.
echo ============================================================
echo   SYNC INTRADIA  Railway -^> Local  + purga de Railway
echo   Fecha : %DATE%  Hora: %TIME%
echo ============================================================
echo.

"%PYTHON%" "%ROOT%scripts\manual\sync_intradia_railway_to_local.py" %*
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" echo [OK] Sync y purga completados.
if "%RC%"=="2" echo [AVISO] Sync completado, pero la purga dejo ruedas sin borrar o fallo: ver arriba.
if "%RC%"=="1" echo [ERROR] El sync fallo. La purga NO corrio: sin sync verificado no se borra.
echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
exit /b %RC%
