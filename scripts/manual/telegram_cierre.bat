@echo off
REM ============================================================
REM  telegram_cierre.bat
REM  Cierre por sector de la ultima rueda, por Telegram.
REM  EN EVALUACION: no esta en la rutina diaria. Ver docs\cierre_grupos.md
REM
REM  Doble clic, sin parametros:
REM    1. muestra el mensaje en pantalla, sin enviar nada
REM    2. pregunta si se envia por Telegram
REM
REM  Con parametros los pasa directo al script, sin preguntar:
REM    scripts\manual\telegram_cierre.bat --dry-run
REM    scripts\manual\telegram_cierre.bat --rueda 2026-09-25 --dry-run
REM    scripts\manual\telegram_cierre.bat --evaluar 20
REM
REM  Sin bloques IF con parentesis: ver CLAUDE.md, "Batch .bat".
REM ============================================================

SET "ROOT=%~dp0..\..\"
SET "PYTHON=%ROOT%venv\Scripts\python.exe"
SET "SCRIPT=%ROOT%scripts\manual\telegram_cierre.py"
cd /d "%ROOT%"

if not "%~1"=="" goto :directo

echo.
echo ============================================================
echo   CIERRE POR SECTOR - vista previa, todavia no se envia
echo ============================================================
"%PYTHON%" "%SCRIPT%" --dry-run
SET EXITCODE=%ERRORLEVEL%
if %EXITCODE%==2 goto :nada
if not %EXITCODE%==0 goto :error

echo.
choice /C SN /N /M "Enviar este mensaje por Telegram? [S/N] "
if errorlevel 2 goto :no_enviado

"%PYTHON%" "%SCRIPT%"
SET EXITCODE=%ERRORLEVEL%
if not %EXITCODE%==0 goto :error
echo.
echo   ENVIADO por Telegram.
goto :fin

:directo
"%PYTHON%" "%SCRIPT%" %*
SET EXITCODE=%ERRORLEVEL%
if %EXITCODE%==2 goto :nada
if not %EXITCODE%==0 goto :error
goto :fin

:no_enviado
echo.
echo   No se envio nada.
SET EXITCODE=0
goto :fin

:nada
echo.
echo   Sin datos para la rueda: no hay nada para mandar.
goto :fin

:error
echo.
echo   ERROR: codigo %EXITCODE% - ver el detalle arriba.
goto :fin

:fin
echo.
echo Presiona una tecla para cerrar...
pause > nul
exit /b %EXITCODE%
