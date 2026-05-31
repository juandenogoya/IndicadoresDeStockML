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
REM  Al terminar el refresh raw, recomputa fundamentales_ratios_q
REM  (capa derivada: ROE/ROA/ROIC/margenes/PER/P-B/BVPS/BPA/FCF +
REM  crecimiento QoQ/YoY). El compute es instantaneo y local. Si se
REM  pasa --dry-run, el compute de ratios se omite.
REM
REM  Default: target = LOCAL (Plan C: fundamentales son recuperables).
REM  Tiempo tipico: 3.5-4 minutos (refresh) + segundos (ratios).
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

"%PYTHON%" "%SCRIPT%" %* 2>&1 | tee "%LOGFILE%"
SET EXITCODE=%ERRORLEVEL%

REM -- Recomputar ratios derivados (omitir si fue --dry-run) --
echo %* | findstr /I /C:"--dry-run" >nul
if errorlevel 1 (
    echo.
    echo --- Computando ratios derivados (fundamentales_ratios_q) ---
    "%PYTHON%" "%SCRIPT_RATIOS%" 2>&1 | tee -a "%LOGFILE%"
) else (
    echo.
    echo DRY-RUN: se omite el computo de ratios.
)

echo.
echo ============================================================
if %EXITCODE%==0 (
    echo   REFRESH COMPLETO  --  todos los chunks OK
) else (
    echo   REFRESH PARCIAL   --  ver detalle en log
)
echo   Log completo: %LOGFILE%
echo ============================================================
echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
