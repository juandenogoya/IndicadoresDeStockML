@echo off
chcp 65001 > nul
REM ============================================================
REM  refresh_earnings_historico.bat
REM  Puebla earnings_historico (fecha de anuncio de cada balance)
REM  desde Alpha Vantage. LOCAL-only.
REM
REM  Key FREE = 25 llamadas/dia, 5/min -> el backfill de ~200
REM  tickers NO entra en una corrida. El script es REANUDABLE:
REM  trae hasta 20 por corrida y sigue por los que faltan.
REM
REM  BACKFILL INICIAL: correr este .bat 1 vez por dia ~10 dias
REM  (o pasar --max-calls menor si ya gastaste cuota en el dia).
REM  Cada corrida avisa cuantos quedan en cola.
REM
REM  DE AHI EN MAS: correr sin --backfill (incremental) apendicea
REM  solo los Q recien reportados -> pocas llamadas por semana.
REM ============================================================

SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe

echo.
echo ============================================================
echo   REFRESH earnings_historico  (Alpha Vantage, LOCAL)
echo   Fecha : %DATE%  Hora: %TIME%
echo ============================================================
echo.
echo Estado actual:
"%PYTHON%" "%ROOT%scripts\refresh_earnings_historico.py" --status
echo.
echo Trayendo la proxima tanda (hasta 20 tickers, backfill + incremental)...
echo.

"%PYTHON%" "%ROOT%scripts\refresh_earnings_historico.py" --backfill
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] El refresh termino con errores. Revisar output arriba.
) ELSE (
    echo.
    echo [OK] Tanda completada. Si quedan tickers en cola, volver a correr
    echo      manana (la cuota free se renueva cada dia^).
)

echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
