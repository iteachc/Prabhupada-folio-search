@echo off
title Folio Search
cd /d "%~dp0"
echo Starting Folio Search... browser opens when ready.
python serve.py
pause
