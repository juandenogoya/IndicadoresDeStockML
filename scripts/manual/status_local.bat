@echo off
chcp 65001 > nul
REM ============================================================
REM  status_local.bat
REM  Estado de las tablas en el PostgreSQL LOCAL.
REM
REM  Diferencia con status.bat:
REM    - status.bat       -> consulta Railway (DATABASE_URL de .env.local)
REM    - status_local.bat -> consulta LOCAL  (DB_HOST/PORT/etc de .env)
REM
REM  Acepta los mismos parametros que db_status.py, ej:
REM    status_local.bat --dias 20
REM ============================================================
cd /d %~dp0..\..
python scripts/manual/db_status.py --target local %*
pause
