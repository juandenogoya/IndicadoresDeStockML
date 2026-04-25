@echo off
chcp 65001 > nul
cd /d %~dp0..\..
python scripts/manual/db_status.py %*
pause
