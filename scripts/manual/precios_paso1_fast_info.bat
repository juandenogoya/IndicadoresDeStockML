@echo off
chcp 65001 > nul
REM ============================================================
REM  precios_paso1_fast_info.bat
REM  Alternativa al cron_paso1_precios.bat usando fast_info.
REM
REM  Endpoint: /v8/finance/quote/ (cotizaciones)
REM  En lugar de: /v8/finance/chart/ (historico OHLCV)
REM
REM  Ventaja: funciona en GH Actions y residencial sin rate limit.
REM  Limitacion: solo captura la sesion mas reciente cerrada.
REM              Para backfill de dias multiples usar cron_paso1_precios.bat.
REM
REM  Tiempo estimado: ~45-50 minutos (199 tickers x ~15s c/u)
REM  Prerequisito: correr despues de las 21:00 UTC (cierre NYSE)
REM  Paso siguiente: cron_paso2_features.bat
REM ============================================================

SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe

REM Cargar DATABASE_URL y otras variables desde .env.local
for /f "usebackq tokens=1,* delims==" %%a in ("%ROOT%.env.local") DO set %%a=%%b

echo.
echo ============================================================
echo   PASO 1 (fast_info) - Precios + Futuros
echo   Fecha : %DATE%  Hora: %TIME%
echo ============================================================
echo.
echo Estado actual de la DB:
"%PYTHON%" "%ROOT%scripts\manual\db_status.py"
echo.
echo PREREQUISITO: correr despues de las 21:00 UTC (cierre NYSE 20:00 UTC).
echo DIFERENCIA vs cron_paso1_precios.bat:
echo   - Usa endpoint de cotizaciones (menos rate-limit)
echo   - Solo captura la sesion de hoy (no backfill de dias pasados)
echo.
set /p CONFIRM="Ejecutar Paso 1 (fast_info)? (s/n): "
if /i not "%CONFIRM%"=="s" (
    echo Operacion cancelada.
    pause
    exit /b 0
)

echo.
echo [1a] Descargando precios via fast_info (199 tickers)...
echo ----------------------------------------
"%PYTHON%" "%ROOT%scripts\actualizar_precios_fast_info.py"
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Descarga de precios termino con errores. Revisar output arriba.
    echo         Considerar usar cron_paso1_precios.bat para backfill.
    pause
    exit /b 1
)

echo.
echo [1b-1e] Futuros + indicadores macro + z-scores (no criticos)...
echo ----------------------------------------
"%PYTHON%" "%ROOT%scripts\34_futuros_indices.py"
"%PYTHON%" -c "
import sys, os
sys.path.insert(0, r'%ROOT%')
try:
    from dotenv import load_dotenv
    load_dotenv(r'%ROOT%.env.local', override=True)
except: pass
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('ind_fut', pathlib.Path(r'%ROOT%scripts/35_calcular_indicadores_futuros.py'))
if spec:
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    if hasattr(m, 'main'): m.main()
" 2>nul

echo.
echo [1e] Z-scores tickers...
"%PYTHON%" -c "
import sys, os
sys.path.insert(0, r'%ROOT%')
try:
    from dotenv import load_dotenv
    load_dotenv(r'%ROOT%.env.local', override=True)
except: pass
from datetime import date
from src.data.database import get_engine
from src.utils.zscore_pipeline import init_tablas, calcular_zscore_tickers
engine = get_engine()
init_tablas(engine)
n = calcular_zscore_tickers(date.today(), engine)
print(f'  Z-scores tickers: {n} filas')
" 2>nul

echo.
echo ============================================================
echo   PASO 1 (fast_info) COMPLETADO
echo ============================================================
echo.
echo Estado post-ejecucion:
"%PYTHON%" "%ROOT%scripts\manual\db_status.py"
echo.
echo Paso siguiente: cron_paso2_features.bat
pause
