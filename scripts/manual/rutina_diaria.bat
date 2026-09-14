@echo off
chcp 65001 > nul
REM ============================================================
REM  rutina_diaria.bat
REM  La rutina diaria COMPLETA en una corrida, en orden:
REM    1. Sync de opciones Railway -> local + purga de Railway
REM    2. Paso 1: precios + futuros + indicadores + z-scores
REM    3. Paso 2: features
REM    4. Paso 3: scanner ML
REM    5. ft_run_diario: derivadas de opciones + 11 bots FT + reportes
REM
REM  Si falla: el sync SIGUE, porque ft_run_diario lo reintenta; los
REM  pasos 1, 2 y 3 FRENAN antes de los bots, salvo el Paso 1 con hasta
REM  10 tickers pendientes, que sigue con aviso. Al final: resumen en
REM  logs\rutina\AAAAMMDD_HHMM\resumen.txt y por Telegram, con el
REM  detalle de los tickers pendientes.
REM
REM  Retomar desde un paso:  rutina_diaria.bat --desde paso2
REM  Sin Telegram:           rutina_diaria.bat --sin-telegram
REM
REM  Motor: scripts\manual\rutina_diaria.py  -  Politica: src\utils\rutina.py
REM  Los .bat de cada paso siguen existiendo para rehacer uno solo.
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
echo   RUTINA DIARIA - sync, Paso 1, Paso 2, Paso 3, Forward Testing
echo   Fecha : %DATE%  Hora: %TIME%
echo ============================================================
"%PYTHON%" "%ROOT%scripts\manual\chequeo_rutina.py" --solo-avisar
echo Corre todo en orden. Si falla el Paso 1, 2 o 3 frena antes de los bots
echo y dice que volver a correr. Al final manda el resumen por Telegram.
echo Para retomar desde un paso: rutina_diaria.bat --desde paso2
echo.
set /p CONFIRM="Ejecutar la rutina completa? (s/n): "
if /i not "%CONFIRM%"=="s" goto :cancelado

"%PYTHON%" "%ROOT%scripts\manual\rutina_diaria.py" todo %*
set "RC=%ERRORLEVEL%"
echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
exit /b %RC%

:cancelado
echo Operacion cancelada.
pause
exit /b 0
