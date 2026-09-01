# breeze-tts-zhtw — Breeze TTS 2 for Traditional Chinese

A 繁體中文-first wrapper around [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
(open-weight zh/en TTS with voice clone, voice design and inline vocal events).

What this adds on top of upstream:

- **Web UI** — paste a paragraph, drop a reference clip, preview, download WAV.
- **繁 → 簡 before synthesis** — type Traditional, we convert with OpenCC (live preview of
  what will be spoken). Kept ON by default: in A/B listening tests on the same seed the
  Simplified-fed output was clearly better (an ASR round-trip shows no difference, so judge by ear).
  Applies to vocal-event tags too (`[嘆氣]` → `[叹气]`).
- **Reference clip auto-transcription** — upload a clip with no transcript and the
  逐字稿 field is filled by Breeze-ASR-25 (Whisper-large-v2 fine-tune, Traditional output),
  either via an external ASR service or in-process.
- **Vocal-event chips** — one click inserts `[笑]`, `(sigh)`, … at the cursor.
- **Standalone Docker image** — one container, GPU, weights mounted read-only.
- **Windows-native fallback** — works around two Windows-only problems (see below).

Upstream lives in `upstream/breeze-tts` as a git submodule and is **never modified**;
everything of ours is under `app/`, `scripts/` and the Docker files.

## Layout

```
app/server.py           FastAPI server (UI + /api/tts + /api/transcribe_ref + /api/convert)
app/breeze_runtime.py   our load_runtime (replaces upstream's; adds Windows-safe loading)
app/ui/index.html       single-file UI
scripts/                download_models.ps1, patch_checkpoint.py
upstream/breeze-tts     git submodule → breezeblue-ai/breeze-tts (pinned)
Dockerfile, docker-compose.yml
```

## Quick start (Docker, recommended)

```bash
# HTTPS / gh also work: gh repo clone romanticamaj/breeze-tts-zhtw -- --recurse-submodules
git clone --recurse-submodules git@github.com:romanticamaj/breeze-tts-zhtw.git
cd breeze-tts-zhtw

# 1) Weights live OUTSIDE the repo (≈7.2 GB + 3.1 GB). Skip this if you already have them.
pwsh scripts/download_models.ps1          # → C:\ai-models\breeze-tts-2 and C:\ai-models\breeze-asr

# 2) Build + run. First build 10–15 min from cold (≈3 GB torch download), image ≈ 13.5 GB.
docker compose up -d --build
```

**How the weights get in:** `docker-compose.yml` bind-mounts two host directories read-only —
`C:/ai-models/breeze-tts-2 → /models/breeze-tts-2` and `C:/ai-models/breeze-asr → /models/breeze-asr`.
If yours live elsewhere, put the host paths in a `.env` next to the compose file before `up`:

```
BREEZE_TTS2_WEIGHTS=D:/models/breeze-tts-2
BREEZE_ASR_WEIGHTS=D:/models/breeze-asr
BREEZE_PORT=7772
```

Sanity check of a complete download: `breeze-tts-2/` contains `model-00001-of-00002.safetensors`,
`model-00002-of-00002.safetensors`, `config.json`, `tokenizer.json` and an `audio_tokenizer/` folder;
`breeze-asr/` contains `model.safetensors` and `config.json`.

Open http://localhost:7772. Model load takes ≈ 1 min on NVMe (up to ~3 min on slower disks);
until then `/health` answers **HTTP 503 `{"status":"loading"}`** (so `curl -f` fails on purpose) and
the UI shows 「模型載入中…」. When ready `/health` reports device, VRAM, `fast_path` and `asr_mode`.

Requirements: NVIDIA GPU ≥ 12 GB (7.2 GB resident + 3.1 GB while ASR is loaded),
Docker with the NVIDIA container runtime, CUDA 12.8-capable driver (RTX 50-series OK).
Already cloned without `--recurse-submodules`? Run `git submodule update --init`.

**Build/log noise that is expected and harmless:**

- Build: the flash-attn step prints `Preparing metadata (setup.py): finished with status 'error'`
  followed by `>> flash-attn wheel not available … fast path disabled`. Intended fallback; `/health`
  will show `fast_path: false`. (Also two `Running pip as the 'root' user` warnings.)
- Runtime: `The tokenizer you are loading … incorrect regex pattern … fix_mistral_regex` (upstream
  tokenizer config; output is unaffected), `SoX could not be found` (torchaudio backend probe — we
  decode with libsndfile, not SoX), `flash-attn is not installed`, and FastAPI's `on_event is deprecated`.

Day-to-day:

```bash
docker compose stop            # free the GPU
docker compose up -d           # start again (no rebuild)
docker compose up -d --build   # after changing app/ or the Dockerfile
docker logs -f breeze-tts-zhtw
```

## 使用方式（UI）

1. **要合成的段落** — 直接打繁體。勾著「繁 → 簡 自動轉換」（預設開），下方會即時預覽實際送給模型的簡體字。
   想加聲音表情就點文字框下方的 chips（`[笑]`、`[叹气]`、`(sigh)`…），會插在游標處。
2. **聲音克隆（參考音檔）** — 拖一段乾淨人聲（5–15 秒、單一說話者）進去。
   wav / mp3 / flac / ogg 直接可用；m4a / aac（如 iPhone 語音備忘錄）會自動轉檔
   （裝了 ffmpeg 就用 ffmpeg，macOS 上退回內建的 afconvert）。上傳後會自動用 Breeze ASR
   產生逐字稿填入「參考音檔的逐字稿」，請核對一下再往下（逐字稿越準、克隆越像）。
   不上傳參考音檔也能生成，改由「進階設定 → 語音指示」用文字描述音色（voice design）。
3. **進階設定** — `CFG scale`：1.0 純克隆最自然；用語音指示（voice design / direction）時官方建議 4。
   `Seed`：同輸入＋同 seed 結果可重現；不滿意就換個數字重抽。滑鼠移到 ⓘ 有說明。
4. 按 **產生語音** → 出現播放器可試聽，**⬇ 下載 WAV** 匯出（24 kHz、16-bit、單聲道）。
   輸出檔同時保存在 `outputs/`（最近 200 個）。

## Windows-native run (no Docker)

```powershell
py -3.10 -m venv .venv
.venv\Scripts\pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r upstream\breeze-tts\requirements.txt -r requirements.txt
python scripts\patch_checkpoint.py C:\ai-models\breeze-tts-2
.venv\Scripts\python app\server.py       # http://localhost:7772
```

Two Windows-only issues are handled automatically:

1. The checkpoint's `config.json` asks the text encoder for `flash_attention_2`
   (no Windows wheel) → `scripts/patch_checkpoint.py` switches it to `sdpa`.
2. transformers' mmap shard loading segfaults on Windows (torch 2.9/2.11, safetensors
   0.7/0.8) → `app/breeze_runtime.py` preloads the state dict via `safe_open().get_tensor`
   when `sys.platform == "win32"` (force with `BREEZE_SAFE_LOAD=1/0`).

## macOS (Apple Silicon) run

```bash
python3.12 -m venv .venv                 # 3.10–3.12; or: uv venv --python 3.12 .venv
.venv/bin/pip install torch==2.9.1 torchaudio==2.9.1
.venv/bin/pip install -r upstream/breeze-tts/requirements.txt -r requirements.txt
hf download BreezeBlue/Breeze-TTS-2 --local-dir ~/ai-models/breeze-tts-2
hf download MediaTek-Research/Breeze-ASR-25 --local-dir ~/ai-models/breeze-asr
.venv/bin/python scripts/patch_checkpoint.py ~/ai-models/breeze-tts-2
.venv/bin/python app/server_mac.py       # http://localhost:7772
```

`app/server_mac.py` runs the eager path on MPS (`/health` shows `device: mps:0`);
the fast path stays CUDA-only. Weights default to `~/ai-models/…` — override with
`BREEZE_TTS2_MODEL_PATH` / `BREEZE_ASR_MODEL_PATH`, or move both at once with
`BREEZE_MAC_MODELS_ROOT`. The in-process ASR runs on MPS too. Measured RTF ≈ 3–10
on an M5 Pro (48 GB); expect audio to take a few times its duration to generate.

## Configuration (env)

| Variable | Default | Meaning |
|---|---|---|
| `BREEZE_TTS2_MODEL_PATH` | `C:\ai-models\breeze-tts-2` | TTS weights (`/models/breeze-tts-2` in Docker) |
| `BREEZE_ASR_MODEL_PATH` | `C:\ai-models\breeze-asr` | Breeze-ASR-25 for the in-process fallback |
| `BREEZE_ASR_SERVICE_URL` | `""` | External Breeze ASR base URL (e.g. `http://127.0.0.1:8765`). Empty = always in-process |
| `BREEZE_ASR_IDLE_UNLOAD_SEC` | `600` | Unload the in-process ASR after idle |
| `BREEZE_TTS2_PORT` / `HOST` | `7772` / `0.0.0.0` | Bind |
| `BREEZE_TTS2_FAST` | `0` | Upstream fast path (Linux + flash-attn wheel, ~14.4 GB VRAM) |

Compose-level: `BREEZE_PORT` (host port), `BREEZE_TTS2_WEIGHTS`, `BREEZE_ASR_WEIGHTS` (host paths to mount).

## API

- `POST /api/tts` — form: `text`, `ref_audio` (file, optional), `ref_text`, `instruction`,
  `convert_t2s` (`1`/`0`), `cfg_scale`, `seed` → `{url, duration_sec, rtf, spoken_text, …}`
- `POST /api/transcribe_ref` — form: `ref_audio` → `{text, source: "asr-service"|"local"}`
- `POST /api/convert` — form: `text` → `{converted, changed}`
- `GET /api/audio/<id>.wav?download=1`
- `GET /health`

One generation at a time (single GPU); a second concurrent `/api/tts` gets HTTP 409.

```bash
curl -X POST http://localhost:7772/api/tts \
  -F "text=大家好，歡迎收聽今天的節目。[笑] 我們開始吧！" \
  -F "ref_audio=@voice.wav" -F "ref_text=參考音檔裡實際說的話"
```

## Notes

- Vocal events documented upstream: `[笑] [叹气] [咳嗽] [清嗓子]` / `(laugh) (sigh) (cough) (clears throat)`.
  They are plain text, not tokens; other tags are experimental.
- `cfg_scale` 1.0 for plain cloning; upstream recommends 4 for voice design / voice direction.
- Measured speed without the fast path: RTF ≈ 2.2–2.7 on an RTX 5070 Ti (upstream's 0.32 needs flash-attn + 24 GB).
- Taiwanese/閩南語 loan words (e.g. 蚵仔煎) may be mispronounced regardless of script.

## License

This repository combines components under **different** licenses — the model weights' terms are
the restrictive ones, so read them before using this for anything beyond personal research:

| Component | License | Notes |
|---|---|---|
| This repo's own code (`app/`, `scripts/`, Docker files) | [Apache-2.0](LICENSE) | Matches upstream's code license. |
| Upstream code `upstream/breeze-tts` | [Apache-2.0](upstream/breeze-tts/LICENSE) | © BreezeBlue; used unmodified as a git submodule. |
| **Breeze-TTS-2 model weights** | [BreezeBlue Research and Non-Commercial License v1.0](upstream/breeze-tts/MODEL_LICENSE) | **Research / non-commercial only. Not open source. Commercial use requires a separate written license from BreezeBlue (Resonia, Inc.).** Weights are therefore never baked into the Docker image or committed here — download them yourself and accept the terms on Hugging Face. |
| Breeze-ASR-25 weights | [Apache-2.0](https://huggingface.co/MediaTek-Research/Breeze-ASR-25) | © MediaTek Research. |
| OpenCC data | Apache-2.0 | via `opencc-python-reimplemented`. |

Practical consequence: **voice output from this tool is bound by the non-commercial model license.**
Do not use it in paid products, ads, or client work without a commercial license from BreezeBlue.
Also do not clone a voice you do not have the speaker's permission to clone.
