@echo off
REM ============================================================
REM  universo.bat
REM  Alta / baja de tickers del universo (Tarea 14).
REM
REM  El universo se sirve de la tabla `activos` (fuente unica).
REM  ALTA: backfill 2a OHLCV + indicadores + features + z-scores +
REM        fundamentales + dual-write activos (local+Railway) + log.
REM  BAJA: soft delete (activo=FALSE) dual-write + log. Guard de
REM        posiciones FT abiertas.
REM
REM  Ejemplos:
REM    scripts\manual\universo.bat list
REM    scripts\manual\universo.bat add NVDA --sector Technology --industry Semiconductors
REM    scripts\manual\universo.bat add KO --sector "Consumer Defensive" --dry-run
REM    scripts\manual\universo.bat remove XYZ --motivo "delisting"
REM ============================================================

SET "ROOT=%~dp0..\..\"
SET "PYTHON=%ROOT%venv\Scripts\python.exe"
SET "SCRIPT=%ROOT%scripts\manual\universo.py"

"%PYTHON%" "%SCRIPT%" %*

echo.
echo Presiona una tecla para cerrar...
pause > nul
