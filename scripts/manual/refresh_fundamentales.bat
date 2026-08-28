@echo off
chcp 65001 > nul
REM ============================================================
REM  refresh_fundamentales.bat
REM  Wrapper Windows del refresh de fundamentales (yahooquery).
REM
REM  Trae los ultimos 8 trimestres (default) de income/balance/
REM  cashflow + valuation_measures para los 199 tickers y hace
REM  UPSERT en local. Cada corrida captura restatements.
REM
REM  Al terminar el refresh raw, encadena (todo LOCAL, instantaneo):
REM    1. compute_fundamentales_ratios.py  -> fundamentales_ratios_q
REM       (ROE/ROA/ROIC/margenes/PER/P-B/BVPS/BPA/FCF + crecimiento QoQ/YoY)
REM    2. refresh_ticker_pais.py           -> ticker_pais (country+region,
REM       1 call assetProfile/ticker; necesario para el comparativo sectorial)
REM    3. compute_fundamentales_sector.py  -> fundamentales_ticker_vs_sector
REM       (cada metrica vs mediana de pares de su region; politica 3-niveles)
REM    4. compute_multiplos_px.py          -> *_px en fundamentales_ratios_q
REM       (PER/P-B/P-S/EV-EBITDA con el cierre del dia sobre el ultimo Q)
REM    5. compute_fundamentales_sector.py --valuacion-px
REM       (pisa las 4 metricas de valuacion de vs_sector con los *_px)
REM  Si se pasa --dry-run, se omiten los 5 pasos derivados.
REM
REM  ORDEN NO NEGOCIABLE (4 y 5 se agregaron 26/8/2026): el refresh agrega un
REM  trimestre NUEVO, y los *_px viven en la fila del ultimo Q. Sin el paso 4,
REM  esa fila queda sin PER/P-B/P-S/EV-EBITDA y el dashboard muestra la
REM  valuacion vacia hasta el siguiente recovery_incremental. El paso 5 va
REM  DESPUES del 4 porque pisa vs_sector con los *_px recien calculados; si
REM  corriera antes, el comparativo sectorial quedaria con multiplos fiscales.
REM
REM  Default: target = LOCAL (Plan C: fundamentales son recuperables).
REM  Tiempo tipico: 3.5-4 min (refresh) + ~15s (pais) + segundos (ratios/sector).
REM
REM  Argumentos opcionales:
REM    --dry-run             solo loguea, no escribe
REM    --tickers AAPL,MSFT   solo refrescar tickers especificos
REM    --lookback-q N        N trimestres recientes (default 8)
REM    --chunk-size N        tickers por chunk async (default 20)
REM
REM  Ejemplos:
REM    scripts\manual\refresh_fundamentales.bat
REM    scripts\manual\refresh_fundamentales.bat --dry-run
REM    scripts\manual\refresh_fundamentales.bat --tickers AAPL,NVDA,JPM
REM    scripts\manual\refresh_fundamentales.bat --lookback-q 12
REM ============================================================

SET "ROOT=%~dp0..\..\"
SET "PYTHON=%ROOT%venv\Scripts\python.exe"
SET "SCRIPT=%ROOT%scripts\refresh_fundamentales.py"
SET "SCRIPT_RATIOS=%ROOT%scripts\compute_fundamentales_ratios.py"
SET "SCRIPT_PAIS=%ROOT%scripts\refresh_ticker_pais.py"
SET "SCRIPT_SECTOR=%ROOT%scripts\compute_fundamentales_sector.py"
SET "SCRIPT_MULTIPLOS=%ROOT%scripts\compute_multiplos_px.py"
SET "LOGDIR=%ROOT%logs"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value 2^>nul') do set DT=%%I
SET "LOGFILE=%LOGDIR%\refresh_fundamentales_%DT:~0,8%_%DT:~8,4%.log"

echo.
echo ============================================================
echo   REFRESH FUNDAMENTALES  (income/balance/cashflow/valuation)
echo   Log: %LOGFILE%
echo ============================================================
echo.

REM El pipe a tee PISA %ERRORLEVEL% con el de tee (siempre 0), asi que el .bat
REM reportaba "REFRESH COMPLETO" aunque python fallara. Verificado: python
REM saliendo con 7 se veia como 0. Se captura el estado real con un marcador
REM escrito DENTRO del bloque, antes de que el pipe se lo coma.
REM El redirect va ANTES del echo a proposito: "echo OK > f" guardaria "OK "
REM con el espacio previo al operador -- por eso ademas se compara por substring.
SET "RCFILE=%TEMP%\refresh_fundamentales_rc.txt"
del "%RCFILE%" 2>nul
( "%PYTHON%" "%SCRIPT%" %* && >"%RCFILE%" echo OK || >"%RCFILE%" echo FAIL ) 2>&1 | tee "%LOGFILE%"
SET "REFRESH_RC="
if exist "%RCFILE%" set /p REFRESH_RC=<"%RCFILE%"
del "%RCFILE%" 2>nul

REM -- Capas derivadas (omitir si fue --dry-run) --
REM Nota: sin parentesis literales en los echo dentro del bloque if(...) -->
REM cmd.exe los interpreta como cierre del bloque y rompe el flujo.
echo %* | findstr /I /C:"--dry-run" >nul
if errorlevel 1 goto :derivadas
echo.
echo DRY-RUN: se omiten las capas derivadas ratios/pais/sector/multiplos.
goto :fin_derivadas

:derivadas
echo.
echo --- 1/5 Ratios derivados [fundamentales_ratios_q] ---
"%PYTHON%" "%SCRIPT_RATIOS%" 2>&1 | tee -a "%LOGFILE%"
echo.
echo --- 2/5 Pais/region por ticker [ticker_pais] ---
"%PYTHON%" "%SCRIPT_PAIS%" 2>&1 | tee -a "%LOGFILE%"
echo.
echo --- 3/5 Comparativo vs sector [fundamentales_ticker_vs_sector] ---
"%PYTHON%" "%SCRIPT_SECTOR%" 2>&1 | tee -a "%LOGFILE%"
echo.
echo --- 4/5 Multiplos al cierre del dia [*_px en fundamentales_ratios_q] ---
"%PYTHON%" "%SCRIPT_MULTIPLOS%" 2>&1 | tee -a "%LOGFILE%"
echo.
echo --- 5/5 Valuacion vs sector con *_px [fundamentales_ticker_vs_sector] ---
"%PYTHON%" "%SCRIPT_SECTOR%" --valuacion-px 2>&1 | tee -a "%LOGFILE%"
:fin_derivadas

echo.
echo ============================================================
if "%REFRESH_RC:~0,2%"=="OK" (
    echo   REFRESH COMPLETO  --  todos los chunks OK
) else (
    echo   REFRESH PARCIAL   --  ver detalle en log
)
echo   Log completo: %LOGFILE%
echo ============================================================
echo.

REM -- Pausa SOLO si se abrio con doble clic --
REM Antes pausaba SIEMPRE, lo que hacia imposible automatizarlo o encadenarlo.
REM Dos niveles, de mas fuerte a mas debil:
REM  1. REFRESH_NO_PAUSE definida -> nunca pausa. Escape determinista para
REM     Programador de tareas / otro .bat / CI:  set REFRESH_NO_PAUSE=1
REM  2. Heuristica: al hacer doble clic, cmd.exe arranca con /c y el nombre del
REM     .bat queda en %cmdcmdline% -> frena para no perder la salida al cerrarse
REM     la ventana. OJO: invocarlo como  cmd /c ruta\refresh_fundamentales.bat
REM     tambien matchea, por eso existe el nivel 1.
REM Sin parentesis en los echo: ver nota sobre if(...) mas arriba.
if defined REFRESH_NO_PAUSE goto :fin
echo %cmdcmdline% | findstr /I /C:"%~nx0" >nul
if errorlevel 1 goto :fin
echo Presiona cualquier tecla para cerrar...
pause > nul
:fin
