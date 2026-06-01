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
REM  Si se pasa --dry-run, se omiten los 3 pasos derivados.
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

REM -- Capas derivadas (omitir si fue --dry-run) --
REM Nota: sin parentesis literales en los echo dentro del bloque if(...) -->
REM cmd.exe los interpreta como cierre del bloque y rompe el flujo.
echo %* | findstr /I /C:"--dry-run" >nul
if errorlevel 1 goto :derivadas
echo.
echo DRY-RUN: se omiten las capas derivadas ratios/pais/sector.
goto :fin_derivadas

:derivadas
echo.
echo --- 1/3 Ratios derivados [fundamentales_ratios_q] ---
"%PYTHON%" "%SCRIPT_RATIOS%" 2>&1 | tee -a "%LOGFILE%"
echo.
echo --- 2/3 Pais/region por ticker [ticker_pais] ---
"%PYTHON%" "%SCRIPT_PAIS%" 2>&1 | tee -a "%LOGFILE%"
echo.
echo --- 3/3 Comparativo vs sector [fundamentales_ticker_vs_sector] ---
"%PYTHON%" "%SCRIPT_SECTOR%" 2>&1 | tee -a "%LOGFILE%"
:fin_derivadas

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
