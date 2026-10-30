@echo off
chcp 65001 > nul
REM ============================================================
REM  intradia_post_rutina.bat
REM  Correr DESPUES de rutina_diaria.bat, antes de la apertura:
REM    1. push_vigilancia.py: arma la lista de vigilancia de la rueda D
REM       foto + cierre por sector, sin balances y la sube a local y
REM       Railway. Oracle la lee en los cortes de 11:00 y 15:00 ET.
REM    2. sync_intradia_railway_to_local.py: baja las barras y las
REM       alertas enviadas, y purga Railway dejando 30 ruedas.
REM
REM  Linea intradiaria EN PRUEBA en la rama main-y9tr06: todavia no es
REM  un paso de rutina_diaria. Ver docs\intradia.md.
REM
REM  Sin tee: los codigos de salida que se informan son los reales.
REM  OJO al editar: sin bloques IF ( ... ) y en modo binario, ver CLAUDE.md.
REM ============================================================

SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe
cd /d "%ROOT%"

echo.
echo ============================================================
echo   INTRADIA post rutina: lista de vigilancia + sync
echo   Fecha : %DATE%  Hora: %TIME%
echo ============================================================
echo.

"%PYTHON%" "%ROOT%scripts\intradia\push_vigilancia.py"
set "RC_PUSH=%ERRORLEVEL%"
echo.
"%PYTHON%" "%ROOT%scripts\manual\sync_intradia_railway_to_local.py"
set "RC_SYNC=%ERRORLEVEL%"

set "RC=0"
if "%RC_PUSH%"=="2" set "RC=2"
if "%RC_SYNC%"=="2" set "RC=2"
if "%RC_PUSH%"=="1" set "RC=1"
if "%RC_SYNC%"=="1" set "RC=1"

echo.
if "%RC_PUSH%"=="0" echo [OK] Lista de vigilancia subida.
if "%RC_PUSH%"=="2" echo [AVISO] Lista subida sin la foto de FT: ver arriba.
if "%RC_PUSH%"=="1" echo [ERROR] La lista NO se subio completa: Oracle no va a tener la de hoy.
if "%RC_SYNC%"=="0" echo [OK] Sync y purga completados.
if "%RC_SYNC%"=="2" echo [AVISO] Sync completado, pero la purga dejo ruedas sin borrar o fallo.
if "%RC_SYNC%"=="1" echo [ERROR] El sync fallo. La purga NO corrio.
echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
exit /b %RC%
