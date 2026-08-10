@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ---------------------------------------------------------------
REM cmd.exe is byte-offset based. Non-ASCII text inside ( ) blocks
REM desynchronises the parser, so this script uses goto only.
REM Keep every REM / separator line pure ASCII.
REM ---------------------------------------------------------------

echo.
echo ============================================================
echo   StoryWriter - 화자 구분 음성 기록기  설치
echo ============================================================
echo.
echo 이 설치는 약 3~5GB 를 내려받습니다. 인터넷 연결이 필요합니다.
echo 한 번만 하면 됩니다. 다음부터는 run.bat 만 실행하세요.
echo.
pause
echo.

REM --- 1. Python ---------------------------------------------------
echo [1/6] Python 확인 중...
where python >nul 2>&1
if not errorlevel 1 goto py_found
echo.
echo   [X] Python 을 찾을 수 없습니다.
echo.
echo       https://www.python.org/downloads/  에서 Python 3.11 설치
echo       설치 화면 맨 아래 "Add python.exe to PATH" 를 반드시 체크하세요.
echo       설치 후 이 창을 닫고 install.bat 을 다시 실행하세요.
echo.
pause
exit /b 1

:py_found
python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>&1
if not errorlevel 1 goto py_ok
python --version
echo.
echo   [X] Python 3.10 이상이 필요합니다. 3.11 을 권장합니다.
echo.
pause
exit /b 1

:py_ok
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo       %%v  OK

REM --- 2. ffmpeg ---------------------------------------------------
echo.
echo [2/6] ffmpeg 확인 중...
where ffmpeg >nul 2>&1
if not errorlevel 1 goto ffmpeg_ok
echo       ffmpeg 가 없습니다. winget 으로 설치를 시도합니다...
winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
where ffmpeg >nul 2>&1
if not errorlevel 1 goto ffmpeg_ok
echo.
echo   [!] ffmpeg 가 아직 PATH 에 잡히지 않았습니다.
echo       이 창을 닫고 install.bat 을 다시 실행하세요.
echo.
pause
exit /b 1

:ffmpeg_ok
echo       ffmpeg OK

REM --- 3. GPU ------------------------------------------------------
echo.
echo [3/6] GPU 확인 중...
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
set "GPU_MODE=NVIDIA"
where nvidia-smi >nul 2>&1
if not errorlevel 1 goto gpu_found
set "TORCH_INDEX="
set "GPU_MODE=CPU"
echo       NVIDIA GPU 를 찾지 못했습니다. CPU 모드로 설치합니다.
echo       CPU 는 훨씬 느립니다. 1시간 녹음에 30분 이상 걸릴 수 있습니다.
goto gpu_done

:gpu_found
for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do echo       %%g

:gpu_done

REM --- 4. venv -----------------------------------------------------
echo.
echo [4/6] 가상환경 준비 중...
if exist ".venv\Scripts\activate.bat" goto venv_ready
python -m venv .venv
if not errorlevel 1 goto venv_ready
echo   [X] 가상환경 생성 실패
pause
exit /b 1

:venv_ready
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip --quiet
echo       OK

REM --- 5. packages -------------------------------------------------
echo.
echo [5/6] 패키지 설치 중... 수 분 걸립니다. 창을 닫지 마세요.
echo.
if not defined TORCH_INDEX goto torch_cpu
echo       PyTorch CUDA 버전 설치 중...
pip install torch torchaudio --index-url %TORCH_INDEX%
goto torch_check

:torch_cpu
echo       PyTorch CPU 버전 설치 중...
pip install torch torchaudio

:torch_check
if not errorlevel 1 goto torch_ok
echo   [X] PyTorch 설치 실패
pause
exit /b 1

:torch_ok
echo.
echo       나머지 패키지 설치 중...
pip install -r requirements.txt
if not errorlevel 1 goto pkg_ok
echo   [X] 패키지 설치 실패
pause
exit /b 1

:pkg_ok

REM --- 6. Hugging Face token ---------------------------------------
echo.
echo [6/6] Hugging Face 토큰 설정
echo.
if not exist ".env" goto ask_token
findstr /b /c:"HF_TOKEN=hf_" ".env" >nul 2>&1
if errorlevel 1 goto ask_token
echo       .env 에 토큰이 이미 있습니다. 건너뜁니다.
goto finish

:ask_token
echo   화자 구분 모델은 Hugging Face 토큰이 필요합니다. 무료입니다.
echo.
echo     1. https://hf.co/settings/tokens
echo        에서 read 토큰 생성
echo     2. https://huggingface.co/pyannote/speaker-diarization-community-1
echo        페이지에서 약관에 동의
echo.
echo   두 단계를 마친 뒤 토큰을 붙여넣으세요. 나중에 하려면 그냥 Enter.
echo.
set "HFTOKEN="
set /p "HFTOKEN=  토큰: "

if exist ".env" goto env_ready
copy /y ".env.example" ".env" >nul

:env_ready
if not defined HFTOKEN goto no_token
powershell -NoProfile -Command "(Get-Content -Raw -Encoding UTF8 '.env') -replace '(?m)^HF_TOKEN=.*$', 'HF_TOKEN=%HFTOKEN%' | Set-Content -NoNewline -Encoding UTF8 '.env'"
echo.
echo       토큰 저장 완료
goto token_done

:no_token
echo.
echo   [!] 토큰을 넣지 않았습니다. 전사는 되지만 화자 구분은 동작하지 않습니다.
echo       나중에 .env 파일을 메모장으로 열어 HF_TOKEN= 뒤에 붙여넣으세요.

:token_done
if not "%GPU_MODE%"=="CPU" goto finish
powershell -NoProfile -Command "(Get-Content -Raw -Encoding UTF8 '.env') -replace '(?m)^WHISPER_MODEL=.*$', 'WHISPER_MODEL=large-v3-turbo' -replace '(?m)^BATCH_SIZE=.*$', 'BATCH_SIZE=4' | Set-Content -NoNewline -Encoding UTF8 '.env'"
echo       CPU 모드에 맞춰 모델을 large-v3-turbo 로 설정했습니다.

:finish
echo.
echo ============================================================
echo   설치 완료
echo ============================================================
echo.
echo   run.bat 을 더블클릭하면 실행됩니다.
echo   첫 실행 때 AI 모델 약 3GB 를 더 내려받습니다.
echo.
pause
exit /b 0
