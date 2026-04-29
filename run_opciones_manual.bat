@echo off
chcp 1252 >nul
title Opciones Snapshot Manual

cd /d "%~dp0"

echo.
echo  Iniciando Opciones Snapshot Manual...
echo  Directorio: %CD%
echo.

python scripts\manual\opciones_snapshot_manual.py %*

echo.
pause
