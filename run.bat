@echo off
chcp 65001 >nul
cd /d "%~dp0"
title StoryWriter - 화자 구분 음성 기록기

if not exist ".venv\Scripts\activate.bat" goto setup
if not exist ".env" goto setup
goto start

:setup
echo.
echo 처음 실행입니다. 설치를 먼저 진행합니다.
echo.
call install.bat
if errorlevel 1 exit /b 1
if not exist ".venv\Scripts\activate.bat" exit /b 1

:start
call .venv\Scripts\activate.bat
echo.
echo   StoryWriter 를 시작합니다. 브라우저가 자동으로 열립니다.
echo   종료하려면 이 창에서 Ctrl+C 를 누르거나 창을 닫으세요.
echo.
python -m app.main

echo.
echo 서버가 종료되었습니다.
pause
