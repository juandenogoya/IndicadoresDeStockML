@echo off
chcp 65001 > nul
REM ============================================================
REM  check_yfinance.bat
REM  Verifica si yf.download() esta disponible y que fechas
REM  retorna. No escribe nada en DB — solo lectura.
REM
REM  Resultado:
REM    [OK]      -> yfinance funciona, pipeline puede correr
REM    [PARCIAL] -> algunos tickers OK, posible rate limit parcial
REM    [FALLO]   -> rate limit activo, ejecutar primero
REM                 limpiar_cookies_yfinance.bat y cambiar de red
REM ============================================================

SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe

echo.
"%PYTHON%" "%ROOT%scripts\manual\check_yfinance_download.py"
pause
