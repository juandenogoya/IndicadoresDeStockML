@echo off
chcp 65001 > nul
REM ============================================================
REM  refresh_earnings_historico.bat
REM  Fechas de anuncio de balances (earnings_historico) desde Alpha
REM  Vantage. Insumo de la vista 'Reaccion a balances' del dashboard y
REM  de cualquier analisis que necesite excluir los dias de balance.
REM
REM  TARGET: PostgreSQL LOCAL - Plan C.
REM
REM  CUOTA: la key free admite 25 llamadas/dia y 5/min -> el script trae
REM  hasta 20 tickers por corrida, con 13s de pausa (~4 min), y se corta
REM  limpio cuando Alpha Vantage avisa que se acabo. Es REANUDABLE: la
REM  proxima corrida sigue por los que faltan.
REM
REM  Desde el 19/9/2026 corre SOLO, como ultimo paso de rutina_diaria.bat
REM  (politica INFORMAR: nunca frena la rutina). Este .bat rehace el paso
REM  a mano, por ejemplo para vaciar un atraso mas rapido.
REM
REM  OJO al editar: nada de parentesis sin escapar adentro de un bloque
REM  IF ( ... ). Editar en modo binario, nunca con sed -i: ver CLAUDE.md.
REM ============================================================

SET ROOT=%~dp0..\..\
SET PYTHON=%ROOT%venv\Scripts\python.exe

REM Posicionarse en la raiz para que el script encuentre el .env
cd /d "%ROOT%"

echo.
echo ============================================================
echo   Fechas de balances - earnings_historico
echo   TARGET: LOCAL  ^|  Fecha : %DATE%  Hora: %TIME%
echo ============================================================
echo.
echo Cobertura actual:
"%PYTHON%" "%ROOT%scripts\refresh_earnings_historico.py" --status
echo.
echo Se traen hasta 20 tickers (tope de la key free). Tarda unos 4 minutos.
echo.
set /p CONFIRM="Ejecutar? (s/n): "
if /i not "%CONFIRM%"=="s" goto :cancelado

"%PYTHON%" "%ROOT%scripts\manual\rutina_diaria.py" paso earnings
set "RC=%ERRORLEVEL%"
echo.
if "%RC%"=="0" echo [OK] Corrida terminada. Si quedan tickers en cola, volver a correrlo manana.
if "%RC%"=="2" echo [AVISO] Terminado con avisos: ver el detalle de arriba.
if "%RC%"=="1" echo [ERROR] Termino con errores. Revisar el log de arriba.
echo.
echo Cobertura post-ejecucion:
"%PYTHON%" "%ROOT%scripts\refresh_earnings_historico.py" --status
echo.
echo Presiona cualquier tecla para cerrar...
pause > nul
exit /b %RC%

:cancelado
echo Operacion cancelada.
pause > nul
exit /b 0
