# CLAUDE.md — breeze-tts-zhtw

繁體中文-first wrapper around Breeze-TTS-2 (voice clone TTS). Read README.md first; this file
covers what a coding agent needs beyond it.

## Hard rules

- **Never edit anything under `upstream/breeze-tts`** — it is a pinned git submodule
  (breezeblue-ai/breeze-tts @ ca632ce). All our code lives in `app/`, `scripts/`, and the
  Docker files. If upstream behavior must change, wrap or replace it in `app/` (see
  `app/breeze_runtime.py`, which replaces upstream's `load_runtime`).
- **Model weights never enter the repo or the Docker image.** Breeze-TTS-2 weights are
  research/non-commercial (see README License table); they are bind-mounted at runtime.
- Keep endpoints in `app/server.py` as sync `def` (not `async def`): GPU work is synchronous
  and would block the event loop, making `/health` time out during generation.

## Architecture in one minute

`app/server.py` (FastAPI, port 7772) loads the 3B TTS model once at startup (~7.2 GB VRAM,
bf16, eager attention) and serves `app/ui/index.html` (single file, no build step).
Reference-clip transcription (`/api/transcribe_ref`): if `BREEZE_ASR_SERVICE_URL` is set and
healthy, forward to it; else lazy-load Breeze-ASR-25 in-process (fp16 ~3.1 GB, unloads after
`BREEZE_ASR_IDLE_UNLOAD_SEC`). TTS and in-process ASR share `_request_lock` — one GPU job at
a time; concurrent `/api/tts` returns 409. 繁→簡 via OpenCC happens server-side before
synthesis (`convert_t2s=1`, default on — user A/B-listened and Simplified-fed output was
clearly better; do not flip this default based on text-level metrics).

## Windows quirks (both handled in code — do not "clean up")

1. Checkpoint `config.json` requests `flash_attention_2` for the text encoder; no Windows
   wheel exists → `scripts/patch_checkpoint.py` rewrites it to `sdpa` (idempotent).
2. transformers' mmap shard loading segfaults on Windows (torch 2.9/2.11, safetensors
   0.7/0.8, both this model and Whisper) → `app/breeze_runtime.py` preloads the full state
   dict via `safe_open().get_tensor` and calls `from_pretrained(None, config=…, state_dict=…)`
   when `sys.platform == "win32"` (`BREEZE_SAFE_LOAD=1/0` to force). Linux uses the plain path.
   Note: `from_pretrained(None)` skips `generation_config.json` — Whisper's is loaded explicitly.

## Build / run / verify

- `docker compose up -d --build` (10–15 min cold; image ~13.5 GB). Compose project name is
  `breeze-tts-zhtw` regardless of checkout directory — two clones address the same container.
- Health: `curl http://127.0.0.1:7772/health` → 503 `{"status":"loading"}` while loading
  (~1 min), then `{"status":"ok", device, vram, fast_path, asr_mode, …}`.
- Smoke test after changes: POST `/api/convert` (t2s), `/api/tts` with and without
  `ref_audio`+`ref_text`, `/api/transcribe_ref` with a short wav; expect `rtf` ≈ 2–3.
- Expected build/log noise (flash-attn "error" then fallback, `fix_mistral_regex`, SoX,
  `on_event` deprecation) is listed in README — don't chase it.

## Performance expectations

RTF ≈ 2.2–2.7 on RTX 5070 Ti (eager). Upstream's advertised RTF 0.32 needs the fast path:
Linux + flash-attn wheel + ~14.4 GB VRAM (`BREEZE_TTS2_FAST=1`). macOS: `app/server_mac.py`
runs the eager path on MPS (RTF ≈ 3–10 on an M5 Pro) by monkeypatching `resolve_device` and
tolerating `FastBreezeStreamingRuntime`'s CUDA-only check before importing server.py — the
submodule stays untouched. The fast path stays CUDA-only; Docker on macOS still has no GPU.
