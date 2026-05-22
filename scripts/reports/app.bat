@echo off
REM ====================================================================
REM app.bat
REM Lanza la app local (Streamlit) para generar reportes/infografias.
REM Abre el navegador en http://localhost:8501
REM
REM Para cerrarla: Ctrl+C en esta consola.
REM ====================================================================

setlocal
pushd "%~dp0..\.."

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: no encuentro el venv en venv\Scripts\activate.bat
    popd
    endlocal
    exit /b 2
)

call venv\Scripts\activate.bat
streamlit run scripts\reports\app.py
set RC=%errorlevel%
call venv\Scripts\deactivate.bat 2>nul

popd
endlocal & exit /b %RC%
