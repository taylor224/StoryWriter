# Speaker-Aware Voice Recorder

A local (Windows + NVIDIA GPU) speech-to-text web service. Upload an audio file
and it writes down **who said what** — and once you name a speaker, they are
**recognized automatically in every later file**.

```
Alex Kim : Hi everyone, let's get started.
Dana Park : Sounds good.
Speaker A : Go ahead.
Speaker B : Thanks. 안녕하세요, 반갑습니다.
```

Pile the transcripts into Claude or ChatGPT and you get a searchable personal
archive — [Record and search your life](#record-and-search-your-life)

---

## Quick start (Windows, two double-clicks)

1. Download the zip from [Releases](https://github.com/taylor224/StoryWriter/releases) and unpack it
2. **Double-click `install.bat`** — checks Python and ffmpeg, detects your GPU,
   installs everything and walks you through the Hugging Face token
   (about 3-5GB downloaded, once)
3. **Double-click `run.bat`** — your browser opens automatically

You can start with `run.bat` too. If nothing is installed it calls `install.bat` for you.

> **⚠️ Install [Python 3.13](https://www.python.org/downloads/release/python-31314/), not the newest.**
> The front page of python.org offers **3.14, which whisperx does not support yet.**
> Do not miss **"Add python.exe to PATH"** at the bottom of the installer.
> (You do not have to remove an existing 3.14 — `install.bat` finds 3.13 via the `py` launcher.)
>
> Speaker separation needs a free Hugging Face token —
> [create one](https://hf.co/settings/tokens), then accept the terms on the
> [model page](https://huggingface.co/pyannote/speaker-diarization-community-1).
> `install.bat` walks you through it.

Manual installation and the details are below.

---

## How it works

Whisper turns speech into text but **cannot tell speakers apart.** So two models
are combined.

| Stage | Model | What it does |
|---|---|---|
| 0. Silence removal | frame RMS energy | Cuts long silences and builds a **map back to original times** (applies to 1, 2 and 3) |
| 0.5 Chunking | — | Splits long files at silence and runs 1-3 per chunk |
| 1. Transcription | WhisperX (faster-whisper `large-v3-turbo`) | audio -> text |
| 2. Alignment | wav2vec2 | word-level timestamps |
| 3. Diarization | pyannote `speaker-diarization-community-1` | "who spoke when" plus a **256-dim embedding per speaker** |
| 3.5 Stitching | cosine similarity + Hungarian assignment | Merges the per-chunk speaker labels into one set |
| 3.6 Re-merging | cosine + simultaneous-speech disproof | Rejoins one person who got split into several speakers |
| 3.7 Hallucination filter | rule-based | Drops sentences Whisper invented |
| 4. Recognition | cosine similarity + Hungarian assignment | Matches against saved voices and applies real names |

Stage 4 is what "enroll a speaker once, recognize them forever" actually is. The
embeddings (voiceprints) from stage 3 accumulate in SQLite, and each new file's
speaker embeddings are compared against them by cosine similarity.

### Silence removal and timestamps

Stage 0 applies to **all of** 1, 2 and 3. Even so, **result timestamps are always
on the original audio's clock.**

Using the trimmed waveform directly would slide everything after a cut earlier by
however much was removed. So a table (`vad.Timeline`) records what was kept, and
every stage maps its results back to original coordinates right before returning
them.

- **Transcription and alignment** — every segment and word time is mapped back
  as soon as alignment finishes
- **Diarization** — pyannote's speaker turns are mapped back, and any turn that
  crosses a cut is **split.** Mapping only its endpoints would make that speaker
  own the silence we removed and steal words from whoever is on the far side of it

Playback and click-to-seek on the result page use the original file, so nothing
desyncs.

The biggest time saving is in **diarization.** whisperx already skips silence with
its own internal VAD, but pyannote's segmentation sweeps the whole file with a
sliding window, so silence costs exactly as much as speech there. On a recording
that is 40% silence, diarization drops by nearly 40% too.

Silence detection needs no extra model — just 20ms frame RMS. Because the noise
floor differs per file, the threshold is not an absolute dB: it sits between that
file's own quiet side (bottom 10%) and speaking side (top 5%). If those are less
than 12dB apart the evidence is too weak and **nothing is cut** — leaving the
original alone beats chopping off someone's words.

Turn it off with `TRIM_SILENCE=false` in `.env`, or `--no-trim` on the CLI.

### Audio filters — read this before turning one on

**"Cleaner to the ear" does not mean "easier to transcribe."** Whisper was trained
on 680k hours of noisy real-world audio and is already robust to ordinary noise.
Aggressive denoising creates artifacts it never saw in training, and **accuracy
often gets worse.** That is why the default is `off`.

There are still situations where they help, so they are available via
`AUDIO_FILTER` in `.env`.

| Value | What it does | When |
|---|---|---|
| `off` | nothing | the default |
| `voice` | strips low rumble, lifts up whoever sits far from the mic | one person in the room is distant. Leaves the spectrum alone, so relatively safe |
| `denoise` | the above plus FFT denoising | constant fan or white noise. **Higher risk** |
| `declip` | repairs clipping | recordings that came out distorted |

You can also pass an ffmpeg filter string directly (`highpass=f=120,dynaudnorm`).

**If you turn one on, compare the results with and without.** Changing the filter
invalidates the cache and re-transcribes, so running the same file under two names
and reading them side by side is the easy way.

#### This one is always on — the highpass for silence detection

Air conditioning, projector fans and desk vibration sit below 100Hz. You barely
hear them, but they raise the RMS across the board and **defeat silence detection
entirely.** Measured:

```
without a 45Hz hum  ->  31.1s of silence removed
with a 45Hz hum     ->   0.0s removed   (9.1dB range, under the 12dB guard)
   same file + 80Hz highpass  ->  31.1s removed
```

So 80Hz and below is stripped **only for the decision** (`TRIM_HIGHPASS_HZ`). The
filtered waveform is used to judge and then discarded — **what goes to
transcription and diarization is the untouched original**, for the reason above.

### Long files are processed in chunks

Files longer than `CHUNK_SEC` (3600s by default) are split. This is not only about
speed — **pyannote splits one person into several speakers** on very long files,
because over 10 hours both the voice and the mic situation drift.

- **Boundaries land in the middle of silence** — cutting mid-word gets that word
  half-recognized on both sides and makes diarization wobble at the seam. It
  reuses the regions silence removal already found
- **A chunk Timeline carries original coordinates** — a timestamp from inside a
  chunk is original-clock after one restore. There is no chunk index threaded
  around and no offsets added anywhere
- **Speakers are stitched back by voice** (`stitch.py`) — pyannote numbers from
  `SPEAKER_00` inside every chunk, so cosine similarity of the embeddings finds
  the same person again. Assignment is 1:1 so two labels in one chunk can never
  fuse into one person
- **The cache is per chunk** — dying on chunk 7 does not re-run chunks 1-6

If one person comes out as several, lower `STITCH_THRESHOLD` (0.55); if different
people get merged, raise it (0.75).

### When one person is split into several speakers

pyannote splits one person into several speakers often: their tone shifts, they
move away from the mic, or the file is long. So at the end **the final speakers
are compared once more and merged** (`stitch.collapse`). This applies to short
files that were never chunked too.

Unlike stitching there is no 1:1 constraint here. That is what catches splits
pyannote made **within one chunk**, as well as links stitching missed because of
its 1:1 assignment.

Overruling pyannote needs a safety net, but **raising the threshold is not it** —
that would make pairs stitching already missed unmergeable forever. Instead:

- **A pair that ever spoke at the same time is never merged.** If they talked over
  each other they cannot be one person — the only hard disproof there is. It comes
  from pyannote's overlap-inclusive annotation (not the overlap-free one, which
  has that information stripped out)
- **Disproofs are inherited on merge.** Fold A into B and anyone who talked over B
  is also not A
- **Average linkage** (centroid to centroid). Maximum linkage would chain A-B-C
  together when A~B and B~C are close but A~C is not

What was merged and why appears on the result page and in `merged_speakers` in the json.

If one person is still split, lower `MERGE_THRESHOLD` (0.55-0.60); if different
people get merged, raise it (0.72-0.78). `0` disables the stage.

**The surest fix is naming them yourself.** Give two speakers **the same name** in
the assignment panel and they become one person, with their transcript lines
joined. Every one of those voices is then enrolled as that person's voiceprint,
so later files recognize them better.

### Filtering out hallucinations

In stretches that are pure noise, Whisper emits sentences that were common in its
training data.

```
uh uh uh uh uh uh uh
This is Kim Sung-hyun, MBC News
Thanks for watching!
Subtitles by the Amara.org community
```

Silence removal gets rid of many such stretches, but noisy ones are never judged
silent, so there is a second screen on the text side (`cleanup.py`).

**The rule: never delete based on the text alone.** "This is Kim Sung-hyun, MBC
News" is a real utterance if you are transcribing the news, and "see you next
time" ends half the meetings ever recorded. Looking like boilerplate is not
evidence. So there are two verdicts.

| Verdict | Condition | Example |
|---|---|---|
| **drop** | same word or syllable is 60%+ of the text | `uh uh uh uh uh uh` — no content, so nothing is lost even if real |
| **drop** | boilerplate **+ evidence** | `Thanks for watching` with an alignment score of 0.05 |
| **flag** | boilerplate, no evidence | the audio does match the words, so it stays |
| **flag** | evidence only | audio and words disagree, but it is not boilerplate |

"Evidence" means something that came from the audio:

- mean word confidence from the aligner is below `HALLUCINATION_MIN_SCORE`
- absurdly few characters for the duration — filling 30 seconds of silence with
  one sentence is the classic hallucination (normal speech runs 4-8 chars/sec)

**Both the deletions and the keeps are recorded.** They show on the result page,
and `dropped` / `suspect` in the json hold the text, the time and the reason.
Click the line to hear it and decide for yourself. `DROP_SUSPECT=true` removes
them all; `DROP_HALLUCINATION=false` turns the whole thing off.

Some phrases were **deliberately left out** of the pattern list: bare "thank you",
"like", "subscribe", "notification settings", "see you next time". Whisper
hallucinates all of them, but they are also the most common things actually said
in a meeting. The list lives in `BOILERPLATE` in `app/cleanup.py`.

### Hearing just one part of the transcript

Click what someone said and **only that span plays.** Click again to stop.

The server cuts that clip (`/api/results/{name}/clip`). It does not ship the whole
wav, because a 10-hour recording is 1.1GB and nothing would play until the browser
had all of it. Since the file is already 16kHz mono PCM, the relevant bytes get a
fresh header and no ffmpeg is involved — the sound starts the instant you click.

Because timestamps are on the original clock, **playback lines up whether or not
silence was cut or the file was chunked.** You need that to judge a hallucination
by ear.

---

## Installation (Windows)

### 1. Prerequisites

```powershell
# Python 3.13 required. whisperx does not support 3.14 yet.
py -3.13 --version

# ffmpeg
winget install Gyan.FFmpeg
# open a new terminal afterwards and check
ffmpeg -version
```

### 2. Virtual environment and packages

```powershell
cd C:\dev\whisper
py -3.13 -m venv .venv
.venv\Scripts\activate

# Install PyTorch first, pinned, from the CUDA index. See the table below for why.
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 `
    --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

Confirm CUDA is visible:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True NVIDIA GeForce RTX 4070 Ti  <- this is what you want
```

### 3. Hugging Face token

The pyannote model is gated, so it needs a token and accepted terms. (Free.)

1. Create a **read** token at https://hf.co/settings/tokens
2. Open https://huggingface.co/pyannote/speaker-diarization-community-1 and accept the terms
3. Copy `.env.example` to `.env` and fill in `HF_TOKEN=hf_...`

### 4. Check

```powershell
python scripts\smoke.py                 # environment only
python scripts\smoke.py sample.wav      # also run the real pipeline
```

The first run downloads about 3GB of models (cached in `models\`, offline afterwards).

### 5. Run

```powershell
run.bat
```

Your browser opens at http://127.0.0.1:8000.

---

## Installation (macOS, Apple Silicon)

**It runs without a GPU.** That said, on a Mac **the GPU is barely usable.**

| Stage | Mac GPU (Metal/MPS) | What actually happens |
|---|---|---|
| Transcription (WhisperX -> CTranslate2) | **unsupported** — CUDA and CPU only | CPU, optimized through Apple Accelerate |
| Diarization (pyannote) | works, but some operators are missing and timestamps have drifted | CPU by default, opt in with `DEVICE=mps` |
| Alignment (wav2vec2) | works | GPU when `DEVICE=mps` |

```bash
brew install ffmpeg python@3.13       # whisperx does not support 3.14

python3.13 -m venv .venv
source .venv/bin/activate
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0   # no CUDA index
pip install -r requirements.txt

cp .env.example .env                  # fill in HF_TOKEN
```

A Mac-friendly `.env`:

```ini
WHISPER_MODEL=large-v3-turbo   # large-v3 is far too slow on CPU
DEVICE=auto                    # = cpu. Use mps to experiment with the GPU
COMPUTE_TYPE=auto              # = int8
BATCH_SIZE=4
```

```bash
python scripts/smoke.py        # prints which device everything runs on
./run.sh
```

`DEVICE=mps` only moves diarization and alignment to the GPU; transcription stays
on CPU regardless. pyannote's MPS kernels are incomplete and there are **reports of
drifting timestamps**, so if you enable it, check that speaker boundaries have not
shifted. If anything looks off, go back to `DEVICE=cpu`.

---

## Usage

### Transcribing

1. Drop an audio file onto the main page
2. **Result name** — defaults to today's date. Duplicates get `-2`, `-3` appended
3. **Language** — **pin it if you know it** (see below)
4. **Maximum speakers** — if you know how many attended, enter it. Accuracy improves a lot
5. **Glossary** — proper nouns and in-house jargon improve accuracy. Enrolled
   speaker names are passed along automatically. Unchecking the box sends nothing
6. `Start transcription`

### When the output looks wrong

**Text in a language nobody spoke / content nobody said.**
Almost always **language detection failure.** Whisper *always* decodes in the
language it was given, so misreading Korean audio as English makes it produce
plausible English out of nothing. The same happens when it tries to transcribe silence.

- **Pin the language.** That alone fixes most cases
- If you use auto-detect, check `auto-detected, confidence 0.xx` at the top of the
  result page. Low confidence or a visible warning means you should pin it and re-run

> Auto-detect samples several points across the audio. whisperx's default behavior
> looks at **only the first 30 seconds**, so a recording that opens with silence or
> an English greeting sends the entire file into the wrong language. Sampling is
> still not perfect, so pin the language when you know it.

**Participant names or terms nobody said show up.**
Uncheck the glossary box and re-run. Batched inference applies the prompt to every
30-second window, so it can bleed through.

**The same sentence repeats forever.**
That is a hallucination over silence or noise. Trimming the dead air at the start
and end reduces it.

Results are saved as `data\results\<name>.txt` and `<name>.json`.

### Enrolling speakers (the important part)

**Option A — from a result (recommended)**

On the result page, type a name next to `Speaker A` under `Assign speakers` and
save. The voiceprint is enrolled at that moment and the txt is rebuilt.
**No re-transcription, about a second.** Every file you upload afterwards shows
that person by name.

**Option B — enroll a voice sample first**

Create a speaker on the `Speakers` page and upload 10-30 seconds of only that
person talking.

Recognition improves as more voiceprints accumulate for the same person (up to 10
kept per speaker).

---

### Resuming a failed job

Each stage's intermediate result lives in `data\cache\<name>\`. When a job fails,
a **`Resume`** button appears in the list along with which stages will be reused.

```
meeting  [error]  Resume  reuses audio conversion · transcription · word alignment
```

Transcription is the expensive stage. If diarization died on GPU memory, lower
`BATCH_SIZE` in `.env` and retry — it picks up **from diarization.**

The cache is only used when the inputs match. Changing the language, glossary or
speaker counts, or uploading a different file under the same name, recomputes from
the affected stage onward. On success the cache is cleared.

The CLI resumes by default; `--no-resume` turns it off.

---

## Output files

**`<name>.txt`** — consecutive lines from the same speaker are joined

```
Alex Kim : Hi everyone, let's get started.
Dana Park : Sounds good.
```

**`<name>.json`** — timestamps, word-level alignment, speaker embeddings and the
raw segments. This file is why **you can rename a speaker and rebuild the txt**
without re-transcribing anything.

`data\uploads\` keeps both the uploaded original and the converted
`<name>.16k.wav`. Those are scratch files; deleting them does not affect results.

An hour of audio is about 115MB of wav alone, so clear out old files if disk space
runs short (the txt and json live separately in `data\results\`).

---

## Record and search your life

Transcribe-and-forget means you never read them again. Put the text piling up in
`data\results\` **into Claude or ChatGPT and it becomes a personal archive you can
search and question.**

Meetings, calls, lectures, interviews, even notes you talk to yourself — all of it works.

### 1. Accumulate

`<name>.txt` is already in a paste-friendly shape. The speaker name is on the front
of every line, so the LLM keeps track of who said what.

| Service | How |
|---|---|
| **Claude** | Create a [Project](https://claude.ai/projects) and keep adding the txt files to its knowledge. Every conversation inside that project can see everything you have accumulated |
| **ChatGPT** | Upload the files to a Project, or attach them to a conversation |

The default result name is the date (`2026-08-10`), so chronological order comes
for free. Naming them **date + title** (`2026-08-10 team sprint retro`) makes them
far easier to find later.

Assigning speaker names (the enrollment feature) sharply improves search quality.
Left as `Speaker A`, nothing can answer "what did Alex say about this?".

### 2. Ask

```
Summarize the decisions from this meeting as a table with owners and deadlines.

Find every mention of "pricing policy" across the last three months,
in chronological order, and tell me how the position changed.

Collect everything Alex agreed to do, and keep only the items with no
mention of completion yet.

Was anything deferred last week that also went untouched this week?

Pick the three topics I repeated most in this month's meetings.
```

Especially useful for weekly retros, handover documents and project histories.

### 3. Before you upload anything

**Meeting transcripts contain names, contact details, contract terms and
unreleased information.** The moment you upload them to an external service they
are stored on that company's servers, and deleting them later does not undo what
was already processed.

- Confirm the content is something you are allowed to upload
- If other people are on the recording, get **consent for both the recording and
  the external upload**
- Strip the awkward parts before uploading, or keep sensitive transcripts local only

---

## Why versions are pinned

Not everything can be the latest. **whisperx sets the ceiling.** (As of 2026-08-10.)

| Package | Latest | This project | Why |
|---|---|---|---|
| Python | 3.14.7 | **3.13** | whisperx requires `<3.14` |
| torch / torchaudio | 2.13.0 | **2.8.0** | whisperx requires `~=2.8.0` |
| torchvision | 0.26.0 | **0.23.0** | whisperx requires `~=0.23.0` |
| torchcodec | 0.15.0 | **0.7.x** | whisperx `<0.8` ∩ pyannote `>=0.7`. Also the range that is ABI-compatible with torch 2.8 |
| whisperx | 3.8.6 | 3.8.6 | ✅ latest |
| pyannote.audio | 4.0.7 | 4.0.7 | ✅ latest |
| ctranslate2 | 4.8.1 | latest | ✅ automatic |
| faster-whisper | 1.2.1 | latest | ✅ automatic |
| fastapi / uvicorn | 0.141.1 / 0.52.1 | latest | ✅ automatic |
| numpy / scipy | 2.5.2 / 1.18.0 | latest | ✅ automatic |

**Pinning torch matters most.** Install the newest torch first and, when whisperx
goes in afterwards, pip downgrades it to satisfy `torch~=2.8.0` — pulling from the
default PyPI index rather than the CUDA one, which on Windows means the
**CPU-only wheel (230MB)**. The install looks successful while the GPU goes
completely unused. `install.bat` catches exactly this in its final step with
`torch.cuda.is_available()`.

---

## Tuning

Everything is set in `.env`.

| Setting | Default | Meaning |
|---|---|---|
| `TRIM_SILENCE` | `true` | Remove silence before transcription. Timestamps are mapped back to the original clock |
| `TRIM_MIN_SILENCE_SEC` | `0.8` | Silence shorter than this is kept as part of speech |
| `TRIM_PAD_SEC` | `0.25` | Headroom around each utterance. **Raise it if first sounds get clipped** |
| `TRIM_SENSITIVITY` | `0.30` | Where the line sits between noise floor and speech. **Lower it (0.20) if speech is cut, raise it (0.40) if silence is not** |
| `TRIM_HIGHPASS_HZ` | `80` | Low-cut applied only when **judging** silence. The waveform sent to the models is untouched |
| `AUDIO_FILTER` | `off` | `voice` / `denoise` / `declip`. **Always compare results with and without** |
| `CHUNK_SEC` | `3600` | Files longer than this are processed in chunks. `0` disables it |
| `STITCH_THRESHOLD` | `0.65` | Same-person threshold across chunks. **Lower it (0.55) if one person splits into several, raise it (0.75) if several fuse into one** |
| `MERGE_THRESHOLD` | `0.65` | Re-merging over-split speakers. **Lower it (0.55) if one person is still split, raise it (0.75) if different people fuse**. `0` disables it |
| `WHISPER_MODEL` | `large-v3-turbo` | Use `large-v3` when accuracy matters most (several times slower) |
| `DROP_HALLUCINATION` | `true` | Remove sentences Whisper invented |
| `HALLUCINATION_MIN_SCORE` | `0.30` | Alignment-confidence floor. `0` drops just this piece of evidence |
| `DROP_SUSPECT` | `false` | `true` also removes what would only have been flagged. **Real speech can be lost** |
| `MATCH_THRESHOLD` | `0.60` | Cosine threshold. **Raise it (0.65-0.70) if people get confused for each other, lower it (0.50-0.55) if someone is not recognized** |
| `MATCH_MARGIN` | `0.05` | Gap between first and second place. Below this the match is abandoned as too close to call |
| `MIN_SPEECH_SEC` | `5.0` | Speakers with less total speech than this are not auto-recognized |
| `BATCH_SIZE` | `8` | Drop to `4` on out-of-memory |
| `UNLOAD_BETWEEN_STAGES` | `false` | `true` unloads models between stages to save VRAM (slower) |

The evidence for tuning is on the result page: the `auto-recognized · similarity
0.xx` badge. If a misidentification happened just above your threshold, raise it.

---

## Performance

**RTX 4070 Ti (12GB)** — per hour of audio

- Transcription + alignment + diarization, **5-10 minutes total**
- About 5-6GB VRAM (`large-v3-turbo` fp16 1.6GB + wav2vec2 1GB + pyannote 2.5GB).
  `large-v3` pushes it to 8-9GB
- Much slower than that means it fell back to CPU →
  `python -c "import torch; print(torch.cuda.is_available())"`

**Apple Silicon (M1 Max class, CPU)** — per hour of audio, **estimated**

| Model | Transcription | Diarization | Total |
|---|---|---|---|
| `large-v3` | 20-30 min | 10-20 min | **30-50 min** |
| `large-v3-turbo` | 8-12 min | 10-20 min | **20-30 min** |

`large-v3-turbo` is the default. Word error differs by about 0.4pp on average while
being 5-8x faster. Memory sits around 2GB at int8, so 16GB of RAM is plenty.
(These figures are estimates from public benchmarks, not measured in this repo.)

**What turbo costs you:** it is a distilled model whose decoder went from 32 layers
to 4, so it **falls into repetition and hallucination more easily.** Looping the
same phrase or inventing sentences over silence happens more often than with
`large-v3`. This default assumes silence removal and the hallucination filter are
on (both are, by default). If odd sentences keep appearing and the filter does not
catch them, go back to `WHISPER_MODEL=large-v3`.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No matching distribution found for whisperx` + a `Requires-Python >=3.10,<3.14` list | You are on Python 3.14. Install [3.13](https://www.python.org/downloads/release/python-31314/), delete the `.venv` folder and re-run `install.bat` |
| Installed fine but the GPU is unused | The newest torch got downgraded to the PyPI CPU wheel because of whisperx. `pip install --force-reinstall torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128` |
| Text in a language nobody spoke | Language detection failed. **Pin the language** and re-run → [details](#when-the-output-looks-wrong) |
| First syllable keeps getting cut off | Silence removal trimmed too tightly. `TRIM_PAD_SEC=0.4`, or `TRIM_SILENCE=false` |
| Quiet recording but nothing gets faster | A high noise floor blocked the silence decision. Raise `TRIM_SENSITIVITY` to `0.40` |
| `uh uh uh uh` or broadcast sign-offs appear | Whisper hallucination. `DROP_HALLUCINATION=true` (default) removes them. If one persists, add the phrase to `BOILERPLATE` in `app/cleanup.py` |
| Something real got deleted | Check the removed list on the result page, then set `DROP_HALLUCINATION=false` or `HALLUCINATION_MIN_SCORE=0` |
| One person comes out as several | Lower `MERGE_THRESHOLD` to `0.55`. On long files lower `STITCH_THRESHOLD` too |
| Different people merged into one | Raise `MERGE_THRESHOLD` and `STITCH_THRESHOLD` to `0.75` |
| Names or terms nobody said appear | Uncheck the glossary box and re-run |
| `OSError: [WinError 1314] A required privilege is not held by the client` | Windows symlink permissions. **Delete the `models` folder and run again** (v0.1.4+ avoids symlinks). If it persists, move the project off the Desktop to a path with no OneDrive sync such as `C:\StoryWriter`, or turn on Developer Mode under Settings > Privacy & security > For developers |
| `torchcodec is not installed correctly` / `Could not load libtorchcodec` | **Safe to ignore.** winget's ffmpeg is a static build without the shared DLLs torchcodec wants. This project hands pyannote a decoded waveform, so torchcodec is never used |
| `Could not locate cudnn_ops64_9.dll` | torch below 2.4. Reinstall from the CUDA index |
| `torch.cuda.is_available()` is False | The CPU-only wheel got installed. `pip uninstall torch torchaudio`, then reinstall from the CUDA index |
| Diarization model fails to load (401/403/None) | Missing HF token, or the model page terms were never accepted |
| `ffmpeg not found` | `winget install Gyan.FFmpeg`, then restart the terminal |
| CUDA out of memory | `BATCH_SIZE=4`, then `UNLOAD_BETWEEN_STAGES=true` if that is not enough |
| `ValueError: unsupported device mps` | CTranslate2 cannot use MPS. The code already keeps transcription on CPU, so `DEVICE=mps` is fine — if you see this error, use `DEVICE=cpu` |
| Speaker boundaries drift on Mac | The pyannote MPS bug under `DEVICE=mps`. Switch to `DEVICE=cpu` |
| Too slow on Mac | `WHISPER_MODEL=large-v3-turbo`, `BATCH_SIZE=4` |
| A speaker is not recognized | Lower `MATCH_THRESHOLD`, or enroll more samples of that person |
| Two people recognized as one | Raise `MATCH_THRESHOLD` |

---

## Limitations

- **Only one voice survives simultaneous speech.** Whisper transcribes whichever
  voice dominates. Speaker assignment uses pyannote's overlap-free output
  (`exclusive_speaker_diarization`), so boundaries stay clean, but the buried
  utterance itself cannot be recovered
- **Accuracy drops past six speakers.** Specifying the maximum speaker count at
  upload time helps
- **Utterances under 5 seconds are not auto-recognized.** The embedding is too
  unstable and the misidentification risk is high
- **Silence removal is energy-based, so noise weakens it.** With constant air
  conditioning or keyboard noise the evidence gets thin and it uses the original
  untouched (it will not cut wrongly). Conversely a very quiet whisper can read as
  silence, so use `TRIM_SILENCE=false` for those recordings
- **Cutting silence can make speaker boundaries wobble slightly at the seams.**
  `TRIM_PAD_SEC` of headroom stays around every utterance so the real impact is
  small, but if speaker separation matters unusually much, run it once with
  `TRIM_SILENCE=false` and compare

---

## Layout

```
app/
  config.py     settings (.env) — must be imported before anything else (sets HF_HOME)
  db.py         SQLite: speakers / voiceprints / jobs / settings
  audio.py      ffmpeg conversion to 16kHz mono wav + ranged wav reads
  vad.py        silence removal + chunking + the map back to original times (Timeline)
  asr.py        WhisperX transcription + wav2vec2 alignment (models stay resident)
  diarize.py    pyannote diarization + embeddings + speaker assignment
  stitch.py     merging per-chunk speaker labels + re-merging over-split speakers
  cleanup.py    filtering out Whisper hallucinations
  matching.py   cosine similarity matching / enrollment
  render.py     txt and json output, regeneration, Speaker A/B labelling
  pipeline.py   the whole flow + CLI
  jobs.py       single-worker background queue
  main.py       FastAPI routes
scripts/
  smoke.py      installation check (environment + one real pipeline run)
  selfcheck.py  silence removal, chunking, stitching, merging and the hallucination
                filter (no model or GPU needed)
data/
  uploads/  results/  samples/  app.db
models/         HF model cache
whisper/        a clone of openai/whisper (reference only, unused)
run.bat         one-click launch (uses HOST/PORT from .env, opens the browser)
```

It also works from the CLI:

```powershell
python -m app.pipeline meeting.mp3 --name 2026-08-10 --max-speakers 4
```
