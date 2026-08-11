@echo off
chcp 65001 >nul
cd /d "%~dp0"
title StoryWriter - Speaker-Aware Voice Recorder

if not exist ".venv\Scripts\activate.bat" goto setup
if not exist ".env" goto setup
goto start

:setup
echo.
echo First run detected. Running the installer first.
echo.
call install.bat
if errorlevel 1 exit /b 1
if not exist ".venv\Scripts\activate.bat" exit /b 1

:start
call .venv\Scripts\activate.bat
echo.
echo   Starting StoryWriter. Your browser will open automatically.
echo   Press Ctrl+C in this window, or close it, to stop.
echo.
python -m app.main

echo.
echo Server stopped.
pause
