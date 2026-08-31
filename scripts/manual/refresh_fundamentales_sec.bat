@echo off
chcp 65001 > nul
REM ============================================================
REM  refresh_fundamentales_sec.bat
REM  Wrapper Windows del refresh de la fuente SEC XBRL.
REM
REM  Fuente PARALELA a yahooquery: las dos conviven y ningun
REM  consumidor actual lee las tablas fundamentales_sec_*.
REM  Este .bat NO reemplaza a refresh_fundamentales.bat.
REM
REM  Alcance: los ~147 tickers de region USA. Los ADR extranjeros
REM  presentan 20-F ANUAL y no tienen XBRL trimestral: quedan
REM  afuera por construccion. Ver docs/fuentes_fundamentales.md.
REM
REM  INCREMENTAL: consulta submissions (~164 KB/ticker) y solo baja
REM  companyfacts (~4 MB) si cambio el accession del ultimo 10-Q/10-K.
REM  Sin balances nuevos mueve ~24 MB en vez de ~522 MB.
REM
REM  REQUISITO: SEC_USER_AGENT en el .env. SEC devuelve 403 a los
REM  pedidos sin un User-Agent con mail de contacto:
REM      SEC_USER_AGENT=tu-mail@dominio.com IndicadoresDeStockML
REM
REM  Argumentos opcionales (se pasan tal cual al script):
REM    --dry-run              no escribe en la DB
REM    --tickers AAPL,JPM     solo esos tickers
REM    --solo-normalizar      no sale a la red, usa el cache
REM    --forzar               re-baja aunque no haya filing nuevo
REM    --desde YYYY-MM-DD     recorta los periodos
REM
REM  Para correrlo desatendido:  set REFRESH_NO_PAUSE=1
REM ============================================================

SET "ROOT=%~dp0..\..\"
SET "PYTHON=%ROOT%venv\Scripts\python.exe"
SET "SCRIPT=%ROOT%scripts\refresh_fundamentales_sec.py"
SET "LOGDIR=%ROOT%logs"

if not exist "%LOGDIR%" mkdir "%LOGDIR%"

REM Timestamp del log. Antes salia de `wmic`, que Windows 11 ya no incluye:
REM el for /f no devuelve nada, DT queda SIN DEFINIR y el nombre del log sale
REM literal, con la sintaxis de substring adentro. PowerShell siempre esta; y
REM si aun asi fallara, el fallback garantiza un nombre valido -- perder el
REM log por no poder fecharlo seria el peor de los dos males.
for /f "delims=" %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmm"') do set "DT=%%I"
if not defined DT set "DT=sin-fecha"
SET "LOGFILE=%LOGDIR%\refresh_fundamentales_sec_%DT%.log"

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
echo   REFRESH FUNDAMENTALES SEC XBRL  (fuente paralela)
echo   Log: %LOGFILE%
echo ============================================================
echo.

REM El pipe a tee PISA %ERRORLEVEL% con el de tee (siempre 0). Se captura el
REM estado real con un marcador escrito DENTRO del bloque, antes del pipe.
REM El redirect va ANTES del echo a proposito: "echo OK > f" guardaria "OK "
REM con el espacio previo al operador -- por eso se compara por substring.
SET "RCFILE=%TEMP%\refresh_fund_sec_rc.txt"
del "%RCFILE%" 2>nul
( "%PYTHON%" "%SCRIPT%" %* && >"%RCFILE%" echo OK || >"%RCFILE%" echo FAIL ) 2>&1 | tee "%LOGFILE%"
SET "REFRESH_RC="
if exist "%RCFILE%" set /p REFRESH_RC=<"%RCFILE%"
del "%RCFILE%" 2>nul

REM ============================================================
REM  PASOS DERIVADOS
REM
REM  La fuente SEC cruda no sirve sola: los multiplos necesitan ademas la
REM  serie de acciones en circulacion. Antes habia que acordarse de correr
REM  los tres comandos a mano y en orden -- el hermano de yahooquery
REM  (refresh_fundamentales.bat) ya encadenaba sus 5 pasos y este no.
REM
REM    1. refresh_acciones_circulacion : yahooquery + extension SEC validada.
REM       SALE A LA RED. Usa yfinance_lock (regla 9) y NO debe correr
REM       pre-mercado (regla 10).
REM    2. compute_sec_multiplos        : serie diaria. Recomputa DB->local,
REM       sin red. Va COMPLETO (no --incremental) porque un refresh puede
REM       traer restatements que cambian TTM viejos.
REM    3. sec_avisos --defectos --alertar : revisa la red de seguridad y avisa
REM       por Telegram SOLO si hay DEFECTOS. Va al final porque los avisos los
REM       reescribe el refresh del paso 0, y sin este paso la tabla
REM       `fundamentales_sec_avisos` vuelve a ser lo que fue toda la Fase 2:
REM       una red que nadie consulta. El defecto de revenue (87,8% de
REM       reconciliacion) estuvo senalado ahi desde el principio y lo
REM       encontramos meses despues, a mano.
REM       No manda nada si no hay defectos: un canal que avisa de lo
REM       informativo deja de leerse.
REM
REM  Solo corren si el refresh anduvo: sobre una fuente a medio escribir
REM  los derivados propagarian el error en vez de detenerse.
REM  Escape: set SEC_NO_DERIVADOS=1  (util con --solo-normalizar, que es
REM  offline, o cuando solo se quiere refrescar el crudo).
REM ============================================================
if not "%REFRESH_RC:~0,2%"=="OK" goto :sin_derivados
if defined SEC_NO_DERIVADOS goto :sin_derivados

echo.
echo ------------------------------------------------------------
echo   PASO DERIVADO 1/3 -- acciones en circulacion
echo ------------------------------------------------------------
"%PYTHON%" "%ROOT%scripts\refresh_acciones_circulacion.py" 2>&1 | tee -a "%LOGFILE%"

echo.
echo ------------------------------------------------------------
echo   PASO DERIVADO 2/3 -- multiplos diarios
echo ------------------------------------------------------------
"%PYTHON%" "%ROOT%scripts\compute_sec_multiplos.py" 2>&1 | tee -a "%LOGFILE%"

echo.
echo ------------------------------------------------------------
echo   PASO DERIVADO 3/3 -- avisos de normalizacion
echo ------------------------------------------------------------
"%PYTHON%" "%ROOT%scripts\manual\sec_avisos.py" --defectos --alertar 2>&1 | tee -a "%LOGFILE%"

:sin_derivados

echo.
echo ============================================================
if "%REFRESH_RC:~0,2%"=="OK" (
    echo   REFRESH SEC COMPLETO
) else (
    echo   REFRESH SEC CON ERRORES  --  ver detalle en log
)
echo   Log completo: %LOGFILE%
echo ============================================================
echo.

REM -- Pausa SOLO si se abrio con doble clic --
REM  1. REFRESH_NO_PAUSE definida -> nunca pausa (escape determinista para
REM     Programador de tareas / otro .bat / CI).
REM  2. Heuristica: al hacer doble clic, cmd.exe arranca con /c y el nombre
REM     del .bat queda en %cmdcmdline% -> frena para no perder la salida.
REM     OJO: invocarlo como cmd /c ruta\...bat tambien matchea; por eso
REM     existe el nivel 1.
if defined REFRESH_NO_PAUSE goto :fin
echo %cmdcmdline% | findstr /I /C:"%~nx0" >nul
if errorlevel 1 goto :fin
echo Presiona cualquier tecla para cerrar...
pause > nul
:fin
