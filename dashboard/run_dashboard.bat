@echo off
REM Lanza el Dashboard (informe descriptivo por ticker) en local.
REM Plan C: LOCAL es la fuente de verdad. Aseguramos engine local quitando
REM DATABASE_URL (si estuviera seteada, get_engine caeria a Railway).
REM Corre bajo el venv del proyecto: tiene streamlit + weasyprint + pymupdf,
REM necesarios para el render y la exportacion a JPG in-process.

setlocal
pushd "%~dp0.."
set "DATABASE_URL="

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: no encuentro el venv en venv\Scripts\activate.bat
    popd
    endlocal
    exit /b 2
)

call venv\Scripts\activate.bat
streamlit run dashboard\app.py
set RC=%errorlevel%
call venv\Scripts\deactivate.bat 2>nul

popd
endlocal & exit /b %RC%
