@echo off
chcp 65001 > nul
REM ============================================================
REM  compute_perfiles_carteras.bat
REM  Wrapper Windows del computo de perfiles de riesgo (carteras).
REM
REM  Corre el motor PURO Fase 1 (metricas: ATR% multi-TF + beta +
REM  drawdown) + Fase 2 (clasificacion data-driven por percentil
REM  del universo) sobre los 200 tickers y hace UPSERT en la tabla
REM  perfiles_ticker (LOCAL).
REM
REM  Cadencia MENSUAL: el perfil es una propiedad estable del
REM  instrumento, NO va en el recovery diario.
REM
REM  Default: fecha snapshot = hoy, target = LOCAL (Plan C).
REM
REM  Argumentos opcionales:
REM    --dry-run              calcula y muestra la distribucion, no escribe
REM    --fecha YYYY-MM-DD     fecha del snapshot (default hoy)
REM
REM  Ejemplos:
REM    scripts\manual\compute_perfiles_carteras.bat
REM    scripts\manual\compute_perfiles_carteras.bat --dry-run
REM    scripts\manual\compute_perfiles_carteras.bat --fecha 2026-08-06
REM ============================================================

SET "ROOT=%~dp0..\..\"
SET "PYTHON=%ROOT%venv\Scripts\python.exe"
SET "SCRIPT=%ROOT%scripts\compute_perfiles_carteras.py"
SET "LOGDIR=%ROOT%logs"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM Timestamp del log. Antes salia de `wmic`, que Windows 11 ya no incluye:
REM el for /f no devuelve nada, DT queda SIN DEFINIR y el nombre del log sale
REM literal, con la sintaxis de substring adentro. PowerShell siempre esta; y
REM si aun asi fallara, el fallback garantiza un nombre valido -- perder el
REM log por no poder fecharlo seria el peor de los dos males.
for /f "delims=" %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set "DT=%%I"
if not defined DT set "DT=sin-fecha"
SET "LOGFILE=%LOGDIR%\compute_perfiles_%DT%.log"

REM `tee` viene con Git y vive en Git\usr\bin, que NO esta en el PATH de cmd
REM (solo esta Git\cmd). Lanzado desde Git Bash funciona; con doble clic desde
REM el Explorador no, y entonces el log queda vacio sin avisar. Se agrega esa
REM carpeta al PATH solo si hace falta.
REM El `if exist` no es decorativo: `where git` devuelve DOS rutas (Git\cmd y
REM Git\mingw64\bin) y, sin expansion retardada, %PATH% se expande UNA sola vez
REM -- la segunda vuelta del for pisaba lo que agrego la primera y dejaba solo
REM la ruta que no existe.
where tee >nul 2>&1 || for /f "delims=" %%G in ('where git 2^>nul') do @if exist "%%~dpG..\usr\bin\tee.exe" set "PATH=%PATH%;%%~dpG..\usr\bin"

echo.
echo ============================================================
echo   COMPUTE PERFILES CARTERAS  [perfiles_ticker]
echo   Log: %LOGFILE%
echo ============================================================
echo.

"%PYTHON%" "%SCRIPT%" %* 2>&1 | tee "%LOGFILE%"
SET EXITCODE=%ERRORLEVEL%

echo.
echo Exit code: %EXITCODE%
exit /b %EXITCODE%
