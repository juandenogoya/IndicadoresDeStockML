@echo off
chcp 65001 > nul
REM ============================================================
REM  limpiar_cookies_yfinance.bat
REM  Limpia el cache y cookies locales de yfinance.
REM
REM  Cuando usar: si yf.download() da YFRateLimitError en IP
REM  residencial (PC o celular), la cookie de rate-limit queda
REM  guardada en disco y bloquea todas las requests siguientes
REM  sin importar la red. Borrarla fuerza una sesion nueva.
REM
REM  Despues de ejecutar: cambiar de red (WiFi -> 4G o viceversa)
REM  y correr check_yfinance_download.py para verificar.
REM ============================================================

SET COOKIES_DB=%LOCALAPPDATA%\py-yfinance\cookies.db
SET CACHE_FILE=%TEMP%\yfinance.cache

echo.
echo ============================================================
echo   LIMPIAR CACHE / COOKIES  yfinance
echo ============================================================
echo.

REM Mostrar estado actual
echo Estado actual:
if exist "%COOKIES_DB%" (
    echo   [EXISTE] %COOKIES_DB%
) else (
    echo   [NO EXISTE] %COOKIES_DB%
)
if exist "%CACHE_FILE%" (
    echo   [EXISTE] %CACHE_FILE%
) else (
    echo   [NO EXISTE] %CACHE_FILE%
)
echo.

REM Borrar cookies
if exist "%COOKIES_DB%" (
    del /f /q "%COOKIES_DB%"
    if not exist "%COOKIES_DB%" (
        echo   [OK] cookies.db eliminado.
    ) else (
        echo   [ERROR] No se pudo eliminar cookies.db.
    )
) else (
    echo   [SKIP] cookies.db no existia.
)

REM Borrar cache temporal
if exist "%CACHE_FILE%" (
    del /f /q "%CACHE_FILE%"
    if not exist "%CACHE_FILE%" (
        echo   [OK] yfinance.cache eliminado.
    ) else (
        echo   [ERROR] No se pudo eliminar yfinance.cache.
    )
) else (
    echo   [SKIP] yfinance.cache no existia.
)

echo.
echo ============================================================
echo   PROXIMO PASO
echo   1. Cambiar de red: WiFi -> 4G (o viceversa)
echo   2. Verificar con: check_yfinance_download.bat
echo   3. Si da [OK], lanzar el pipeline normalmente.
echo ============================================================
echo.
pause
