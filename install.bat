@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM ---------------------------------------------------------------
REM cmd.exe is byte-offset based. Non-ASCII text inside ( ) blocks
REM desynchronises the parser, so this script uses labels and jumps.
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

REM --- 1. Python 3.10-3.13 ------------------------------------------
REM whisperx requires >=3.10,<3.14. Python 3.14 is the current default
REM download on python.org, so many machines have an unusable version.
echo [1/7] Python 확인 중...
set "PYEXE="
for %%V in (3.13 3.12 3.11 3.10) do call :try_py %%V
if defined PYEXE goto py_ok

where python >nul 2>&1
if errorlevel 1 goto py_missing
python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>&1
if errorlevel 1 goto py_badver
set "PYEXE=python"
goto py_ok

:py_missing
echo.
echo   [X] Python 을 찾을 수 없습니다.
echo.
echo       아래에서 Python 3.13 을 설치하세요.
echo       https://www.python.org/downloads/release/python-31314/
echo       페이지 맨 아래 "Windows installer (64-bit)" 를 받으면 됩니다.
echo       설치 화면 맨 아래 "Add python.exe to PATH" 를 반드시 체크하세요.
echo.
pause
exit /b 1

:py_badver
python --version
echo.
echo   [X] 설치된 Python 을 쓸 수 없습니다. 3.10 ~ 3.13 이 필요합니다.
echo       whisperx 가 아직 Python 3.14 를 지원하지 않습니다.
echo       python.org 첫 화면의 최신 버전은 3.14 이니 아래 링크로 받으세요.
echo.
echo       https://www.python.org/downloads/release/python-31314/
echo.
echo       3.13 을 설치한 뒤 이 창을 닫고 install.bat 을 다시 실행하세요.
echo       기존 3.14 를 지울 필요는 없습니다. 알아서 3.13 을 찾아 씁니다.
echo.
pause
exit /b 1

:py_ok
%PYEXE% --version
echo       사용할 Python: %PYEXE%

REM --- 2. ffmpeg ---------------------------------------------------
echo.
echo [2/7] ffmpeg 확인 중...
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
echo [3/7] GPU 확인 중...
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
REM An existing venv may have been built with an unsupported Python.
echo.
echo [4/7] 가상환경 준비 중...
if not exist ".venv\Scripts\python.exe" goto venv_create
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>&1
if not errorlevel 1 goto venv_ready
echo       기존 가상환경의 Python 버전이 맞지 않습니다. 다시 만듭니다.
rmdir /s /q ".venv"

:venv_create
%PYEXE% -m venv .venv
if not errorlevel 1 goto venv_ready
echo   [X] 가상환경 생성 실패
pause
exit /b 1

:venv_ready
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip --quiet
echo       OK

REM --- 5. packages -------------------------------------------------
REM Versions are pinned on purpose. whisperx 3.8.6 requires torch~=2.8.0.
REM Installing a newer torch here makes pip downgrade it later from PyPI,
REM which on Windows is a CPU-only wheel and silently kills CUDA.
echo.
echo [5/7] 패키지 설치 중... 수 분 걸립니다. 창을 닫지 마세요.
echo.
if not defined TORCH_INDEX goto torch_cpu
echo       PyTorch 2.8.0 CUDA 버전 설치 중...
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url %TORCH_INDEX%
goto torch_check

:torch_cpu
echo       PyTorch 2.8.0 CPU 버전 설치 중...
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0

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
echo [6/7] Hugging Face 토큰 설정
echo.
if not exist ".env" goto ask_token
findstr /b /c:"HF_TOKEN=hf_" ".env" >nul 2>&1
if errorlevel 1 goto ask_token
echo       .env 에 토큰이 이미 있습니다. 건너뜁니다.
goto verify

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
if not "%GPU_MODE%"=="CPU" goto verify
powershell -NoProfile -Command "(Get-Content -Raw -Encoding UTF8 '.env') -replace '(?m)^WHISPER_MODEL=.*$', 'WHISPER_MODEL=large-v3-turbo' -replace '(?m)^BATCH_SIZE=.*$', 'BATCH_SIZE=4' | Set-Content -NoNewline -Encoding UTF8 '.env'"
echo       CPU 모드에 맞춰 모델을 large-v3-turbo 로 설정했습니다.

REM --- 7. verify ---------------------------------------------------
REM Catches the silent CPU-wheel downgrade described in step 5.
:verify
echo.
echo [7/7] 설치 검증 중...
python -c "import torch;print('      torch',torch.__version__,'/ CUDA',torch.cuda.is_available())"
if errorlevel 1 goto verify_bad
python -c "import whisperx,pyannote.audio;print('      whisperx / pyannote.audio import OK')"
if errorlevel 1 goto verify_bad
if not "%GPU_MODE%"=="NVIDIA" goto finish
python -c "import sys,torch;sys.exit(0 if torch.cuda.is_available() else 1)"
if not errorlevel 1 goto finish
echo.
echo   [!] NVIDIA GPU 는 있는데 PyTorch 가 CUDA 를 못 씁니다.
echo       CPU 전용 휠이 깔린 것입니다. 아래를 그대로 실행해 고치세요.
echo.
echo       .venv\Scripts\activate
echo       pip install --force-reinstall torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
echo.
pause
exit /b 1

:verify_bad
echo.
echo   [X] 패키지 검증 실패. 위 오류 메시지를 확인하세요.
echo.
pause
exit /b 1

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

REM --- subroutine: pick the first usable py launcher version --------
:try_py
if defined PYEXE goto :eof
py -%1 -c "import sys" >nul 2>&1
if errorlevel 1 goto :eof
set "PYEXE=py -%1"
goto :eof
