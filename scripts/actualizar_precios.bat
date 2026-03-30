@echo off
:: ============================================================
::  Actualizar Precios Railway — Task Scheduler L-V 09:00 ARG
::  Descarga cierre del dia anterior y sube a Railway.
::  El cron de GH Actions (10:00 ARG) usara estos datos.
:: ============================================================
::  DATABASE_URL se carga desde .env.local (NO commiteado, gitignored)
::  Formato del archivo .env.local:
::      DATABASE_URL=postgresql://user:pass@host:port/db
:: ============================================================

set PYTHONPATH=C:\Users\juand\OneDrive\Escritorio\Indicadores y Machine Learning

:: Cargar variables desde .env.local
for /f "usebackq tokens=1,* delims==" %%a in ("%~dp0..\.env.local") do set %%a=%%b

cd /d "C:\Users\juand\OneDrive\Escritorio\Indicadores y Machine Learning"

python scripts\actualizar_precios_railway.py >> logs\actualizar_precios.log 2>&1

echo [%date% %time%] Script finalizado >> logs\actualizar_precios.log
