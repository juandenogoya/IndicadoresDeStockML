@echo off
chcp 65001 > nul
REM ============================================================
REM  refresh_fundamentales_sec.bat
REM  Wrapper Windows del refresh de la fuente SEC XBRL.
REM
REM  Fuente PARALELA a yahooquery: las dos conviven y ningun
REM  consumidor actual lee las tablas fundamentales_sec_*.
REM  Este .bat NO reemplaza a refresh_fundamentales.bat.
REM
REM  Alcance: los ~147 tickers de region USA. Los ADR extranjeros
REM  presentan 20-F ANUAL y no tienen XBRL trimestral: quedan
REM  afuera por construccion. Ver docs/fuentes_fundamentales.md.
REM
REM  INCREMENTAL: consulta submissions (~164 KB/ticker) y solo baja
REM  companyfacts (~4 MB) si cambio el accession del ultimo 10-Q/10-K.
REM  Sin balances nuevos mueve ~24 MB en vez de ~522 MB.
REM
REM  REQUISITO: SEC_USER_AGENT en el .env. SEC devuelve 403 a los
REM  pedidos sin un User-Agent con mail de contacto:
REM      SEC_USER_AGENT=tu-mail@dominio.com IndicadoresDeStockML
REM
REM  Argumentos opcionales (se pasan tal cual al script):
REM    --dry-run              no escribe en la DB
REM    --tickers AAPL,JPM     solo esos tickers
REM    --solo-normalizar      no sale a la red, usa el cache
REM    --forzar               re-baja aunque no haya filing nuevo
REM    --desde YYYY-MM-DD     recorta los periodos
REM
REM  Para correrlo desatendido:  set REFRESH_NO_PAUSE=1
REM ============================================================

SET "ROOT=%~dp0..\..\"
SET "PYTHON=%ROOT%venv\Scripts\python.exe"
SET "SCRIPT=%ROOT%scripts\refresh_fundamentales_sec.py"
SET "LOGDIR=%ROOT%logs"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set DT=%%I
SET "LOGFILE=%LOGDIR%\refresh_fundamentales_sec_%DT:~0,8%_%DT:~8,4%.log"

echo.
echo ============================================================
echo   REFRESH FUNDAMENTALES SEC XBRL  (fuente paralela)
echo   Log: %LOGFILE%
echo ============================================================
echo.

REM El pipe a tee PISA %ERRORLEVEL% con el de tee (siempre 0). Se captura el
REM estado real con un marcador escrito DENTRO del bloque, antes del pipe.
REM El redirect va ANTES del echo a proposito: "echo OK > f" guardaria "OK "
REM con el espacio previo al operador -- por eso se compara por substring.
SET "RCFILE=%TEMP%\refresh_fund_sec_rc.txt"
del "%RCFILE%" 2>nul
( "%PYTHON%" "%SCRIPT%" %* && >"%RCFILE%" echo OK || >"%RCFILE%" echo FAIL ) 2>&1 | tee "%LOGFILE%"
SET "REFRESH_RC="
if exist "%RCFILE%" set /p REFRESH_RC=<"%RCFILE%"
del "%RCFILE%" 2>nul

echo.
echo ============================================================
if "%REFRESH_RC:~0,2%"=="OK" (
    echo   REFRESH SEC COMPLETO
) else (
    echo   REFRESH SEC CON ERRORES  --  ver detalle en log
)
echo   Log completo: %LOGFILE%
echo ============================================================
echo.

REM -- Pausa SOLO si se abrio con doble clic --
REM  1. REFRESH_NO_PAUSE definida -> nunca pausa (escape determinista para
REM     Programador de tareas / otro .bat / CI).
REM  2. Heuristica: al hacer doble clic, cmd.exe arranca con /c y el nombre
REM     del .bat queda en %cmdcmdline% -> frena para no perder la salida.
REM     OJO: invocarlo como cmd /c ruta\...bat tambien matchea; por eso
REM     existe el nivel 1.
if defined REFRESH_NO_PAUSE goto :fin
echo %cmdcmdline% | findstr /I /C:"%~nx0" >nul
if errorlevel 1 goto :fin
echo Presiona cualquier tecla para cerrar...
pause > nul
:fin
