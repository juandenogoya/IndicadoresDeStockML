@echo off
:: ============================================================
::  Actualizar Google Sheets — Exportacion manual
::  Corre sheets_export.py y muestra el resultado en pantalla.
:: ============================================================

set PYTHONPATH=C:\Users\juand\OneDrive\Escritorio\Indicadores y Machine Learning

:: Cargar variables desde .env.local
for /f "usebackq tokens=1,* delims==" %%a in ("%~dp0..\.env.local") do set %%a=%%b

cd /d "C:\Users\juand\OneDrive\Escritorio\Indicadores y Machine Learning"

echo [%date% %time%] Iniciando exportacion a Google Sheets...
python scripts\sheets_export.py

echo.
echo [%date% %time%] Exportacion finalizada. Presiona cualquier tecla para cerrar.
pause > nul
