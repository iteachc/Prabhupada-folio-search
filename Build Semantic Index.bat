@echo off
title Folio Search Index Build
cd /d "%~dp0"
echo Building the semantic index. Safe to close - resumes.
echo.
:loop
python build_index.py 999999
if errorlevel 3 (
  echo Resuming...
  goto loop
)
echo.
python build_index.py 30
echo INDEX COMPLETE AND MERGED! Restart the search app.
pause
