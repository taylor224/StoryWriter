# 화자 구분 음성 기록기

로컬(Windows + NVIDIA GPU)에서 도는 음성 → 텍스트 웹 서비스.
음성 파일을 올리면 **누가 무슨 말을 했는지** 구분해서 기록하고, 한 번 이름을 지정한
화자는 **다음 파일부터 자동으로 인식**한다.

```
안차돌 : 안녕하세요 회의를 시작하겠습니다.
안캐돌 : 네 알겠습니다.
화자A : 말씀 진행하세요
화자B : 네 반갑습니다. Hello my name is Taylor.
```

---

## 빠른 시작 (Windows · 클릭 두 번)

1. [Releases](https://github.com/taylor224/StoryWriter/releases) 에서 zip 을 받아 압축을 푼다
2. **`install.bat` 더블클릭** — Python·ffmpeg 확인, GPU 자동 감지, 패키지 설치,
   Hugging Face 토큰 입력까지 한 번에 진행된다 (약 3~5GB 다운로드, 1회만)
3. **`run.bat` 더블클릭** — 브라우저가 자동으로 열린다

`run.bat` 을 먼저 눌러도 된다. 설치가 안 돼 있으면 알아서 `install.bat` 을 부른다.

> **⚠️ Python 은 반드시 [3.13](https://www.python.org/downloads/release/python-31314/) 을 받을 것.**
> python.org 첫 화면의 최신 버전은 **3.14 인데 whisperx 가 아직 지원하지 않는다.**
> 설치 화면 맨 아래 **"Add python.exe to PATH" 체크**도 잊지 말 것.
> (3.14 가 이미 깔려 있어도 지울 필요 없다. `install.bat` 이 `py` 런처로 3.13 을 찾아 쓴다.)
>
> 화자 구분에는 무료 Hugging Face 토큰이 필요하다 —
> [토큰 생성](https://hf.co/settings/tokens) 후
> [모델 페이지](https://huggingface.co/pyannote/speaker-diarization-community-1)
> 에서 약관 동의. `install.bat` 이 이 과정을 안내한다.

아래는 수동 설치 및 상세 설명이다.

---

## 어떻게 동작하나

Whisper 는 음성을 글자로 바꿀 뿐 **화자 구분을 못 한다.** 그래서 두 모델을 붙여 쓴다.

| 단계 | 사용 모델 | 하는 일 |
|---|---|---|
| 1. 전사 | WhisperX (faster-whisper `large-v3`) | 음성 → 텍스트 |
| 2. 정렬 | wav2vec2 | 단어 단위 타임스탬프 |
| 3. 화자 분리 | pyannote `speaker-diarization-community-1` | "언제 누가 말했나" 구간 + **화자별 256차원 임베딩** |
| 4. 화자 인식 | 코사인 유사도 + Hungarian 배정 | 저장된 목소리와 대조해 실명 부여 |

4번이 "한 번 저장한 화자를 자동 인식"의 정체다. 3번이 뱉는 임베딩(목소리 지문)을
SQLite 에 쌓아두고, 새 파일의 화자 임베딩과 코사인 유사도로 비교한다.

---

## 설치 (Windows)

### 1. 사전 준비

```powershell
# Python 3.13 필요. 3.14 는 whisperx 가 아직 지원하지 않는다.
py -3.13 --version

# ffmpeg
winget install Gyan.FFmpeg
# 설치 후 터미널을 새로 열고 확인
ffmpeg -version
```

### 2. 가상환경 + 패키지

```powershell
cd C:\dev\whisper
py -3.13 -m venv .venv
.venv\Scripts\activate

# PyTorch 는 버전을 고정해서 CUDA 인덱스로 먼저 설치한다. 이유는 아래 표 참고.
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 `
    --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

CUDA 인식 확인:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA GeForce RTX 4070 Ti  <- 이렇게 나와야 함
```

### 3. Hugging Face 토큰

pyannote 모델은 게이트가 걸려 있어 토큰 + 약관 동의가 필요하다. (무료)

1. https://hf.co/settings/tokens 에서 **read** 토큰 생성
2. https://huggingface.co/pyannote/speaker-diarization-community-1 접속 → 약관 동의
3. `.env.example` 을 `.env` 로 복사하고 `HF_TOKEN=hf_...` 채우기

### 4. 점검

```powershell
python scripts\smoke.py                 # 환경만 확인
python scripts\smoke.py 샘플.wav        # 실제 파이프라인까지 확인
```

첫 실행 시 모델 약 3GB 를 내려받는다 (`models\` 폴더에 캐시, 이후 오프라인 동작).

### 5. 실행

```powershell
run.bat
```

브라우저에서 http://127.0.0.1:8000 이 열린다.

---

## 설치 (macOS · Apple Silicon)

**GPU 없이도 돈다.** 다만 Mac 에서는 **GPU 를 거의 못 쓴다.**

| 단계 | Mac GPU (Metal/MPS) | 실제 동작 |
|---|---|---|
| 전사 (WhisperX → CTranslate2) | **미지원** — CUDA·CPU 전용 | CPU. Apple Accelerate 로 최적화는 됨 |
| 화자 분리 (pyannote) | 되지만 일부 연산자 미구현 + 타임스탬프 어긋남 보고 | 기본 CPU, `DEVICE=mps` 로 옵트인 |
| 단어 정렬 (wav2vec2) | 가능 | `DEVICE=mps` 시 GPU |

```bash
brew install ffmpeg python@3.13       # 3.14 는 whisperx 미지원

python3.13 -m venv .venv
source .venv/bin/activate
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0   # CUDA 인덱스 없이
pip install -r requirements.txt

cp .env.example .env                  # HF_TOKEN 채우기
```

`.env` 를 Mac 용으로:

```ini
WHISPER_MODEL=large-v3-turbo   # CPU 전사는 large-v3 가 너무 느리다
DEVICE=auto                    # = cpu. GPU 실험하려면 mps
COMPUTE_TYPE=auto              # = int8
BATCH_SIZE=4
```

```bash
python scripts/smoke.py        # 어떤 디바이스로 도는지 출력해 준다
./run.sh
```

`DEVICE=mps` 는 화자 분리·정렬만 GPU 로 보낸다. 전사는 어차피 CPU 다.
pyannote 의 MPS 커널이 불완전해 **타임스탬프가 어긋난 사례**가 보고돼 있으니,
켰다면 결과에서 화자 경계가 밀리지 않는지 꼭 확인할 것. 이상하면 `DEVICE=cpu`.

---

## 사용법

### 전사

1. 메인 화면에 음성 파일을 끌어다 놓는다
2. **결과 이름** — 기본값은 오늘 날짜. 중복이면 `-2`, `-3` 이 붙는다
3. **최대 화자 수** — 참석 인원을 알면 꼭 넣는다. 정확도가 크게 오른다
4. **용어사전** — 고유명사·회사 용어를 넣으면 인식 정확도가 오른다. 등록된 화자
   이름은 자동으로 함께 전달된다
5. `전사 시작`

결과는 `data\results\<이름>.txt` 와 `<이름>.json` 두 개로 저장된다.

### 화자 등록 (핵심)

**방법 A — 전사 결과에서 지정 (권장)**

결과 페이지 상단 `화자 지정` 에서 `화자A` 옆에 이름을 넣고 저장.
그 순간 목소리 지문이 등록되고 txt 가 다시 만들어진다. **재전사 없음, 1초.**
다음에 올리는 파일부터는 그 사람이 자동으로 실명 표시된다.

**방법 B — 샘플 음성 미리 등록**

`화자 관리` 페이지에서 화자를 만들고, 그 사람만 말하는 10~30초 녹음을 올린다.

같은 사람의 지문이 여러 개 쌓일수록 인식률이 올라간다 (화자당 최대 10개 보관).

---

## 출력 파일

**`<이름>.txt`** — 연속된 같은 화자 발화는 한 줄로 합쳐진다

```
안차돌 : 안녕하세요 회의를 시작하겠습니다.
안캐돌 : 네 알겠습니다.
```

**`<이름>.json`** — 타임스탬프, 단어 단위 정렬, 화자 임베딩, 세그먼트 원본.
이 파일이 있어서 **화자 이름만 바꿔 txt 를 다시 만들 수 있다** (재전사 불필요).

`data\uploads\` 에는 업로드 원본과 변환된 `<이름>.16k.wav` 가 함께 남는다.
1시간 회의면 wav 만 약 115MB 이므로, 디스크가 부족하면 오래된 파일을 지우면 된다
(결과 txt·json 은 `data\results\` 에 따로 있으므로 지워도 무방하다).

---

## 버전 고정 이유

전부 최신을 쓰지는 **못한다.** whisperx 가 상한을 정한다. (기준일 2026-08-10)

| 패키지 | 최신 | 이 프로젝트 | 왜 |
|---|---|---|---|
| Python | 3.14.7 | **3.13** | whisperx 가 `<3.14` 요구 |
| torch / torchaudio | 2.13.0 | **2.8.0** | whisperx 가 `~=2.8.0` 요구 |
| torchvision | 0.26.0 | **0.23.0** | whisperx 가 `~=0.23.0` 요구 |
| torchcodec | 0.15.0 | **0.7.x** | whisperx `<0.8` ∩ pyannote `>=0.7`. torch 2.8 과 ABI 가 맞는 범위이기도 하다 |
| whisperx | 3.8.6 | 3.8.6 | ✅ 최신 |
| pyannote.audio | 4.0.7 | 4.0.7 | ✅ 최신 |
| ctranslate2 | 4.8.1 | 최신 | ✅ 자동 |
| faster-whisper | 1.2.1 | 최신 | ✅ 자동 |
| fastapi / uvicorn | 0.141.1 / 0.52.1 | 최신 | ✅ 자동 |
| numpy / scipy | 2.5.2 / 1.18.0 | 최신 | ✅ 자동 |

**torch 버전 고정이 특히 중요하다.** 최신 torch 를 먼저 깔면, 뒤이어 whisperx 를
설치할 때 pip 이 `torch~=2.8.0` 을 맞추려고 torch 를 되돌린다. 이때 CUDA 인덱스가
아니라 PyPI 기본 인덱스에서 받으므로 Windows 에서는 **CPU 전용 휠(230MB)** 이 깔린다.
설치는 성공한 것처럼 보이지만 GPU 를 전혀 쓰지 못한다.
`install.bat` 은 마지막 단계에서 `torch.cuda.is_available()` 로 이 상황을 검사한다.

---

## 튜닝

`.env` 에서 조정한다.

| 값 | 기본 | 의미 |
|---|---|---|
| `MATCH_THRESHOLD` | `0.60` | 코사인 유사도 임계값. **다른 사람을 같은 사람으로 착각하면 올리고(0.65~0.70), 같은 사람을 못 알아보면 내린다(0.50~0.55)** |
| `MATCH_MARGIN` | `0.05` | 1등과 2등 점수 차. 이보다 작으면 애매하다고 보고 매칭 포기 |
| `MIN_SPEECH_SEC` | `5.0` | 총 발화가 이보다 짧은 화자는 자동 인식하지 않음 |
| `BATCH_SIZE` | `8` | VRAM 부족(OOM) 시 `4` 로 |
| `UNLOAD_BETWEEN_STAGES` | `false` | `true` 면 단계마다 모델을 내려 VRAM 을 아낀다 (느려짐) |

튜닝 근거는 결과 페이지의 `자동 인식 · 유사도 0.xx` 배지에서 얻는다.
오인식이 났을 때 그 값이 임계값 바로 위였다면 임계값을 올리면 된다.

---

## 성능

**RTX 4070 Ti (12GB)** — 1시간 오디오 기준

- 전사 + 정렬 + 화자 분리 합계 **5~10분**
- VRAM 약 8~9GB (`large-v3` fp16 4.7GB + wav2vec2 1GB + pyannote 2.5GB)
- 훨씬 느리면 CPU 로 떨어진 것 → `python -c "import torch; print(torch.cuda.is_available())"`

**Apple Silicon (M1 Max 급, CPU)** — 1시간 오디오 기준 **예상치**

| 모델 | 전사 | 화자분리 | 합계 |
|---|---|---|---|
| `large-v3` | 20~30분 | 10~20분 | **30~50분** |
| `large-v3-turbo` | 8~12분 | 10~20분 | **20~30분** |

Mac 은 `large-v3-turbo` 를 권장한다. 단어오류율 차이가 평균 0.4%p 수준인데
속도는 5~8배다. 메모리는 int8 기준 2GB 안팎이라 16GB 램에서도 무리 없다.
(위 수치는 공개 벤치마크 기반 추정이며 이 저장소에서 실측한 값은 아니다.)

---

## 트러블슈팅

| 증상 | 원인 / 해결 |
|---|---|
| `No matching distribution found for whisperx` + `Requires-Python >=3.10,<3.14` 목록 | Python 3.14 를 쓰고 있다. [3.13](https://www.python.org/downloads/release/python-31314/) 설치 후 `.venv` 폴더를 지우고 `install.bat` 재실행 |
| 설치는 됐는데 GPU 를 안 씀 | 최신 torch 가 whisperx 때문에 PyPI CPU 휠로 되돌려진 것. `pip install --force-reinstall torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128` |
| `Could not locate cudnn_ops64_9.dll` | torch 2.4 미만. CUDA 휠로 재설치 |
| `torch.cuda.is_available()` 가 False | CPU 전용 휠이 깔림. `pip uninstall torch torchaudio` 후 CUDA 인덱스로 재설치 |
| 화자 분리 모델 로드 실패 (401/403/None) | HF 토큰 누락 또는 모델 페이지 약관 미동의 |
| `ffmpeg 를 찾을 수 없습니다` | `winget install Gyan.FFmpeg` 후 터미널 재시작 |
| CUDA out of memory | `BATCH_SIZE=4`, 그래도 안 되면 `UNLOAD_BETWEEN_STAGES=true` |
| `ValueError: unsupported device mps` | CTranslate2 는 MPS 를 못 쓴다. 코드가 전사만 CPU 로 돌리도록 처리하므로 `DEVICE=mps` 여도 정상 — 이 오류가 뜨면 `DEVICE=cpu` |
| Mac 에서 화자 경계가 밀림 | `DEVICE=mps` 의 pyannote MPS 버그. `DEVICE=cpu` 로 |
| Mac 에서 너무 느림 | `WHISPER_MODEL=large-v3-turbo`, `BATCH_SIZE=4` |
| 화자를 못 알아봄 | `MATCH_THRESHOLD` 를 낮추거나, 같은 사람 샘플을 더 등록 |
| 다른 사람을 같은 사람으로 인식 | `MATCH_THRESHOLD` 를 올린다 |

---

## 한계

- **동시 발화는 한 명만 기록된다.** Whisper 가 겹치는 목소리 중 우세한 쪽만 전사한다
- **화자 6명 이상이면 정확도가 떨어진다.** 업로드 시 최대 화자 수를 지정하면 개선된다
- **짧은 발화(5초 미만)는 자동 인식하지 않는다.** 임베딩이 불안정해서 오인식 위험이 크다

---

## 구조

```
app/
  config.py     설정 (.env) — 다른 모듈보다 먼저 import 되어야 함 (HF_HOME 설정)
  db.py         SQLite: 화자 / 보이스프린트 / 작업 / 설정
  audio.py      ffmpeg 로 16kHz mono wav 변환
  asr.py        WhisperX 전사 + wav2vec2 정렬 (모델 상주)
  diarize.py    pyannote 화자 분리 + 임베딩 + 세그먼트 화자 배정
  matching.py   코사인 유사도 매칭 / 등록
  render.py     txt·json 생성, 재생성, 화자A/B 라벨링
  pipeline.py   전체 흐름 + CLI
  jobs.py       단일 워커 백그라운드 큐
  main.py       FastAPI 라우트
scripts/
  smoke.py      설치 점검 (환경 확인 + 실제 파이프라인 1회 실행)
data/
  uploads/  results/  samples/  app.db
models/         HF 모델 캐시
whisper/        openai/whisper 원본 클론 (참고용, 미사용)
run.bat         원클릭 실행 (.env 의 HOST/PORT 사용, 브라우저 자동 실행)
```

CLI 로도 쓸 수 있다:

```powershell
python -m app.pipeline 회의.mp3 --name 2026-08-10 --max-speakers 4
```
