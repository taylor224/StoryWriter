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
echo   StoryWriter - Speaker-Aware Speech To Text for ChatGPT/Claude  installer
echo ============================================================
echo.
echo This downloads roughly 3-5GB and needs an internet connection.
echo You only do this once. After that, just run run.bat.
echo.
pause
echo.

REM --- 1. Python 3.10-3.13 ------------------------------------------
REM whisperx requires >=3.10,<3.14. Python 3.14 is the current default
REM download on python.org, so many machines have an unusable version.
echo [1/7] Checking Python...
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
echo   [X] Python not found.
echo.
echo       Install Python 3.13 from the link below.
echo       https://www.python.org/downloads/release/python-31314/
echo       Grab "Windows installer (64-bit)" at the bottom of the page.
echo       Be sure to check "Add python.exe to PATH" in the installer.
echo.
pause
exit /b 1

:py_badver
python --version
echo.
echo   [X] The installed Python cannot be used. 3.10 - 3.13 is required.
echo       whisperx does not support Python 3.14 yet.
echo       The front page of python.org offers 3.14, so use the link below.
echo.
echo       https://www.python.org/downloads/release/python-31314/
echo.
echo       Install 3.13, close this window, then run install.bat again.
echo       You do not need to remove 3.14 - this finds and uses 3.13.
echo.
pause
exit /b 1

:py_ok
%PYEXE% --version
echo       Using Python: %PYEXE%

REM --- 2. ffmpeg ---------------------------------------------------
echo.
echo [2/7] Checking ffmpeg...
where ffmpeg >nul 2>&1
if not errorlevel 1 goto ffmpeg_ok
echo       ffmpeg is missing. Trying to install it with winget...
winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
where ffmpeg >nul 2>&1
if not errorlevel 1 goto ffmpeg_ok
echo.
echo   [!] ffmpeg is not on PATH yet.
echo       Close this window and run install.bat again.
echo.
pause
exit /b 1

:ffmpeg_ok
echo       ffmpeg OK

REM --- 3. GPU ------------------------------------------------------
echo.
echo [3/7] Checking for a GPU...
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
set "GPU_MODE=NVIDIA"
where nvidia-smi >nul 2>&1
if not errorlevel 1 goto gpu_found
set "TORCH_INDEX="
set "GPU_MODE=CPU"
echo       No NVIDIA GPU found. Installing in CPU mode.
echo       CPU is much slower - an hour of audio can take 30+ minutes.
goto gpu_done

:gpu_found
for /f "tokens=*" %%g in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do echo       %%g

:gpu_done

REM --- 4. venv -----------------------------------------------------
REM An existing venv may have been built with an unsupported Python.
echo.
echo [4/7] Preparing the virtual environment...
if not exist ".venv\Scripts\python.exe" goto venv_create
".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>&1
if not errorlevel 1 goto venv_ready
echo       The existing venv has the wrong Python version. Recreating it.
rmdir /s /q ".venv"

:venv_create
%PYEXE% -m venv .venv
if not errorlevel 1 goto venv_ready
echo   [X] Failed to create the virtual environment
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
echo [5/7] Installing packages... this takes several minutes. Do not close this window.
echo.
if not defined TORCH_INDEX goto torch_cpu
echo       Installing PyTorch 2.8.0 (CUDA build)...
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url %TORCH_INDEX%
goto torch_check

:torch_cpu
echo       Installing PyTorch 2.8.0 (CPU build)...
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0

:torch_check
if not errorlevel 1 goto torch_ok
echo   [X] PyTorch installation failed
pause
exit /b 1

:torch_ok
echo.
echo       Installing the remaining packages...
pip install -r requirements.txt
if not errorlevel 1 goto pkg_ok
echo   [X] Package installation failed
pause
exit /b 1

:pkg_ok

REM --- 6. Hugging Face token ---------------------------------------
echo.
echo [6/7] Hugging Face token setup
echo.
if not exist ".env" goto ask_token
findstr /b /c:"HF_TOKEN=hf_" ".env" >nul 2>&1
if errorlevel 1 goto ask_token
echo       .env already has a token. Skipping.
goto verify

:ask_token
echo   The diarization model needs a Hugging Face token. It is free.
echo.
echo     1. https://hf.co/settings/tokens
echo        Create a read token there
echo     2. https://huggingface.co/pyannote/speaker-diarization-community-1
echo        Accept the terms on that page
echo.
echo   Paste the token once both steps are done, or press Enter to do it later.
echo.
set "HFTOKEN="
set /p "HFTOKEN=  Token: "

if exist ".env" goto env_ready
copy /y ".env.example" ".env" >nul

:env_ready
if not defined HFTOKEN goto no_token
powershell -NoProfile -Command "(Get-Content -Raw -Encoding UTF8 '.env') -replace '(?m)^HF_TOKEN=.*$', 'HF_TOKEN=%HFTOKEN%' | Set-Content -NoNewline -Encoding UTF8 '.env'"
echo.
echo       Token saved
goto token_done

:no_token
echo.
echo   [!] No token entered. Transcription works, but speaker separation will not.
echo       Open .env in Notepad later and paste it after HF_TOKEN=

:token_done
if not "%GPU_MODE%"=="CPU" goto verify
powershell -NoProfile -Command "(Get-Content -Raw -Encoding UTF8 '.env') -replace '(?m)^WHISPER_MODEL=.*$', 'WHISPER_MODEL=large-v3-turbo' -replace '(?m)^BATCH_SIZE=.*$', 'BATCH_SIZE=4' | Set-Content -NoNewline -Encoding UTF8 '.env'"
echo       Set the model to large-v3-turbo to suit CPU mode.

REM --- 7. verify ---------------------------------------------------
REM Catches the silent CPU-wheel downgrade described in step 5.
:verify
echo.
echo [7/7] Verifying the installation...
python -c "import torch;print('      torch',torch.__version__,'/ CUDA',torch.cuda.is_available())"
if errorlevel 1 goto verify_bad
REM torchcodec warns at import time because winget's ffmpeg is a static build
REM with no shared DLLs. Harmless: we hand pyannote a decoded waveform instead.
python -W ignore::UserWarning -c "import whisperx,pyannote.audio;print('      whisperx / pyannote.audio import OK')"
if errorlevel 1 goto verify_bad
python -W ignore::UserWarning -c "import sys;sys.path.insert(0,'.');from app import diarize;print('      audio decode path OK (torchcodec bypassed)')"
if errorlevel 1 goto verify_bad
if not "%GPU_MODE%"=="NVIDIA" goto finish
python -c "import sys,torch;sys.exit(0 if torch.cuda.is_available() else 1)"
if not errorlevel 1 goto finish
echo.
echo   [!] An NVIDIA GPU is present but PyTorch cannot use CUDA.
echo       The CPU-only wheel got installed. Run the following to fix it.
echo.
echo       .venv\Scripts\activate
echo       pip install --force-reinstall torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
echo.
pause
exit /b 1

:verify_bad
echo.
echo   [X] Package verification failed. Check the error messages above.
echo.
pause
exit /b 1

:finish
echo.
echo ============================================================
echo   Installation complete
echo ============================================================
echo.
echo   Double-click run.bat to start it.
echo   The first run downloads about 3GB more of AI models.
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
