@echo off
chcp 65001 > nul
REM ============================================================
REM  sync_opciones_railway_to_local.bat
REM  Trae las 6 tablas de opciones US desde Railway a la DB LOCAL:
REM    - opciones_snapshot
REM    - opciones_resumen_diario
REM    - opciones_zscore_diario
REM    - opciones_sector_zscore_diario      (agregada despues del v1)
REM    - opciones_pcr_plazo_diario          (PCR + muros OI por ventana)
REM    - opciones_sector_pcr_plazo_diario   (PCR sectorial por ventana)
REM
REM  NOTA: opciones AR (opciones_ar_gregas) NO se sincronizan: quedan en
REM  Railway. Cuando se decida usarlas en local, se traera su historia.
REM
REM  Bajo Plan C: Railway acumula el snapshot de opciones (cron
REM  Oracle + GH backup). Este bat las baja al local para que
REM  esten disponibles para analisis, scanner y backtesting.
REM
REM  Sync INCREMENTAL: solo trae las fechas nuevas (ON CONFLICT
REM  DO NOTHING). Es idempotente -- correrlo de mas no duplica.
REM
REM  NO usa yfinance: es Railway DB -> local DB. Sin rate limit.
REM  Se puede correr en cualquier momento, incluso en paralelo
REM  con un recovery de precios.
REM ============================================================

SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe

echo.
echo ============================================================
echo   SYNC OPCIONES  Railway -^> Local
echo   Fecha : %DATE%  Hora: %TIME%
echo ============================================================
echo.
echo Sincroniza incrementalmente (Railway -^> local) las 6 tablas de opciones US:
echo   snapshot, resumen, zscore, sector_zscore, pcr_plazo, sector_pcr_plazo
echo   (opciones AR quedan en Railway, no se traen)
echo.

"%PYTHON%" "%ROOT%scripts\migrations\sync_railway_to_local.py" --tabla opciones
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] El sync de opciones termino con errores. Revisar output arriba.
) ELSE (
    echo.
    echo [OK] Sync de opciones completado.
)

echo.
echo ============================================================
echo   Sync de opciones finalizado.
echo ============================================================
echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
