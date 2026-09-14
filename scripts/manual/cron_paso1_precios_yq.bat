@echo off
chcp 65001 > nul
REM ============================================================
REM  cron_paso1_precios_yq.bat
REM  Paso 1 de la rutina diaria: precios EOD + futuros + indicadores
REM  tecnicos + z-scores de acciones, via YAHOOQUERY: yfinance dejo de
REM  responder confiable el 18/5/2026.
REM
REM  Motor : recovery_incremental.py --target local --engine yahooquery
REM          Detecta por MAX de fecha que tickers faltan y baja SOLO esos.
REM  Resultado: OK / PARCIAL si quedan hasta 10 tickers sin la rueda /
REM  ERROR. Al terminar busca huecos en el medio de la serie de precios.
REM
REM  TARGET: PostgreSQL LOCAL - Plan C.
REM
REM  Corre a traves de scripts\manual\rutina_diaria.py desde el 13/9/2026:
REM    - log en logs\rutina\pasos\paso1_AAAAMMDD_HHMM.log
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
echo   PASO 1 - Precios + Futuros + Indicadores
echo   TARGET: LOCAL  ^|  Fecha : %DATE%  Hora: %TIME%
echo ============================================================
echo.
echo Estado actual de la DB LOCAL:
"%PYTHON%" "%ROOT%scripts\manual\db_status.py" --target local
echo.
echo Baja SOLO los tickers atrasados. El Paso 2 va despues de este.
echo.
set /p CONFIRM="Ejecutar Paso 1? (s/n): "
if /i not "%CONFIRM%"=="s" goto :cancelado

"%PYTHON%" "%ROOT%scripts\manual\rutina_diaria.py" paso paso1
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" echo [OK] Paso 1 completado. Ya se puede correr el Paso 2.
if "%RC%"=="2" echo [AVISO] Terminado con avisos: ver el detalle de arriba.
if "%RC%"=="1" echo [ERROR] El Paso 1 termino con errores. Revisar el log de arriba.
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
