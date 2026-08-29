@echo off
chcp 65001 > nul
REM ============================================================
REM  recovery_incremental.bat
REM  Wrapper Windows del recovery incremental de precios + futuros.
REM
REM  A diferencia de un pipeline ciego de 199 tickers:
REM    - NO descarga los 199 tickers ciegamente
REM    - Detecta cuales tienen MAX(fecha) < ultimo dia habil
REM    - Descarga SOLO los pendientes (batches de 3 por defecto)
REM    - Verifica explicitamente y reporta tickers NO completados
REM
REM  Default: target = LOCAL.
REM
REM  Argumentos opcionales:
REM    --dry-run             solo diagnostico
REM    --target railway      apuntar a Railway en lugar de local
REM    --batch-size N        tickers por lote (default 3)
REM    --max-cycles N        reintentos (default 5)
REM    --skip-futuros        solo precios, no futuros
REM    --skip-indicadores    no recalcula indicadores
REM
REM  Ejemplo de uso normal:
REM    scripts\manual\recovery_incremental.bat
REM
REM  Ejemplo dry-run para ver que tickers faltan:
REM    scripts\manual\recovery_incremental.bat --dry-run
REM ============================================================

SET "ROOT=%~dp0..\..\"
SET "PYTHON=%ROOT%venv\Scripts\python.exe"
SET "SCRIPT=%ROOT%scripts\recovery_incremental.py"
SET "LOGDIR=%ROOT%logs"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM Timestamp del log. Antes salia de `wmic`, que Windows 11 ya no incluye:
REM el for /f no devuelve nada, DT queda SIN DEFINIR y el nombre del log sale
REM literal, con la sintaxis de substring adentro. PowerShell siempre esta; y
REM si aun asi fallara, el fallback garantiza un nombre valido -- perder el
REM log por no poder fecharlo seria el peor de los dos males.
for /f "delims=" %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set "DT=%%I"
if not defined DT set "DT=sin-fecha"
SET "LOGFILE=%LOGDIR%\recovery_incremental_%DT%.log"

echo.
echo ============================================================
echo   RECOVERY INCREMENTAL  (precios + futuros + indicadores)
echo   Log: %LOGFILE%
echo ============================================================
echo.

REM `tee` viene con Git y vive en Git\usr\bin, que NO esta en el PATH de cmd
REM (solo esta Git\cmd). Lanzado desde Git Bash funciona; con doble clic desde
REM el Explorador no, y entonces el log queda vacio sin avisar. Se agrega esa
REM carpeta al PATH solo si hace falta.
REM El `if exist` no es decorativo: `where git` devuelve DOS rutas (Git\cmd y
REM Git\mingw64\bin) y, sin expansion retardada, %PATH% se expande UNA sola vez
REM -- la segunda vuelta del for pisaba lo que agrego la primera y dejaba solo
REM la ruta que no existe.
where tee >nul 2>&1 || for /f "delims=" %%G in ('where git 2^>nul') do @if exist "%%~dpG..\usr\bin\tee.exe" set "PATH=%PATH%;%%~dpG..\usr\bin"
"%PYTHON%" "%SCRIPT%" %* 2>&1 | tee "%LOGFILE%"
SET EXITCODE=%ERRORLEVEL%

echo.
echo ============================================================
if %EXITCODE%==0 (
    echo   RECOVERY COMPLETO  --  todos los pendientes resueltos
) else (
    echo   RECOVERY PARCIAL   --  ver detalle en log
)
echo   Log completo: %LOGFILE%
echo ============================================================
echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
