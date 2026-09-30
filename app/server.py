"""Breeze TTS 2 for Traditional Chinese — voice-clone web UI + JSON API.

Wraps the upstream ``breeze_infer`` runtime (git submodule at upstream/breeze-tts)
with a 繁體中文-friendly workflow:
  * GET  /            — web UI (app/ui/index.html): paragraph + reference voice,
                        繁→簡 conversion before synthesis, vocal-event chips,
                        preview + download.
  * GET  /health      — 200 {"status":"ok", device, vram...} once the model is loaded.
  * POST /api/convert — OpenCC t2s preview of the text that would be spoken.
  * POST /api/tts     — multipart form -> full WAV saved under outputs/,
                        returns JSON with an /api/audio/<id>.wav URL.
  * GET  /api/audio/{name} — serve/download a generated WAV.
  * POST /api/transcribe_ref — auto-transcribe an uploaded reference clip
                        (external Breeze ASR service if configured and up,
                        else in-process Breeze-ASR-25 with idle unload).

Environment (all optional):
  BREEZE_TTS2_MODEL_PATH   weights dir (default C:\\ai-models\\breeze-tts-2)
  BREEZE_TTS2_HOST / PORT  bind address (0.0.0.0 / 7772)
  BREEZE_TTS2_FAST=1       upstream fast path (Linux + flash-attn, ~14.4GB VRAM)
  BREEZE_ASR_SERVICE_URL   external Breeze ASR base URL; empty = never probe
  BREEZE_ASR_MODEL_PATH    Breeze-ASR-25 dir for in-process fallback
  BREEZE_ASR_IDLE_UNLOAD_SEC / BREEZE_ASR_DEVICE
"""

from __future__ import annotations

import os
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from opencc import OpenCC

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
UPSTREAM = ROOT / "upstream" / "breeze-tts"
if not (UPSTREAM / "breeze_infer").is_dir():
    raise SystemExit(
        f"upstream submodule missing at {UPSTREAM} — run: git submodule update --init"
    )
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(APP_DIR))

from breeze_runtime import load_runtime, safe_state_dict_load_needed  # noqa: E402
from breeze_infer.runtime import (  # noqa: E402
    resolve_device,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs  # noqa: E402
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig  # noqa: E402
from models.warmup_profile import load_warmup_profile  # noqa: E402

MODEL_PATH = Path(os.environ.get("BREEZE_TTS2_MODEL_PATH", r"C:\ai-models\breeze-tts-2"))
PORT = int(os.environ.get("BREEZE_TTS2_PORT", "7772"))
HOST = os.environ.get("BREEZE_TTS2_HOST", "0.0.0.0")
OUTPUT_DIR = ROOT / "outputs"
UPLOAD_DIR = ROOT / "uploads"
UI_INDEX = APP_DIR / "ui" / "index.html"
MAX_NEW_TOKENS = 1500
MAX_SEQ_LEN = 2048
REPETITION_PENALTY = 1.1
DEFAULT_CFG_SCALE = 1.0
DEFAULT_INSTRUCTION = "Speak clearly and naturally."
MAX_OUTPUTS_KEPT = 200
# Upstream "fast path" (CUDA graphs + compile, Linux only, ~14.4GB VRAM).
# Off by default; on a 16GB card it cannot coexist with the in-process ASR.
FAST_ALL = os.environ.get("BREEZE_TTS2_FAST", "0") == "1"
FAST_CONFIG = UPSTREAM / "configs" / "fast.json"

# Reference-audio auto-transcription (fills the 逐字稿 field in the UI).
# 1) If BREEZE_ASR_SERVICE_URL is set and its /health is up, forward to it.
# 2) Otherwise lazy-load Breeze-ASR-25 (Whisper-large-v2, fp16 ~3.1GB)
#    in-process; it sits beside the ~7.2GB TTS model and is unloaded after idle.
ASR_SERVICE_URL = os.environ.get("BREEZE_ASR_SERVICE_URL", "")
ASR_MODEL_PATH = Path(os.environ.get("BREEZE_ASR_MODEL_PATH", r"C:\ai-models\breeze-asr"))
ASR_IDLE_UNLOAD_SEC = int(os.environ.get("BREEZE_ASR_IDLE_UNLOAD_SEC", "600"))
ASR_DEVICE = os.environ.get("BREEZE_ASR_DEVICE", "cuda:0")  # "cpu" for testing
ASR_MAX_REF_SEC = 120  # reference clips should be short; refuse absurd uploads

app = FastAPI(title="Breeze TTS 2 (zh-TW)")
_cc_t2s = OpenCC("t2s")
_request_lock = threading.Lock()
_state: dict = {"ready": False, "error": None, "runtime": None}
_asr: dict = {"model": None, "processor": None, "last_used": 0.0}
_asr_lock = threading.Lock()


def _load_model() -> None:
    try:
        tokenizer, model, audio_tokenizer = load_runtime(
            MODEL_PATH,
            device=resolve_device(),
            attn_implementation="eager",
        )
        update_generation_config_for_breeze(model)
        config = FastStreamingConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            max_seq_len=MAX_SEQ_LEN,
            repetition_penalty=REPETITION_PENALTY,
            fast_all=True if FAST_ALL else None,
        )
        runtime = FastBreezeStreamingRuntime(
            model, audio_tokenizer, config, tokenizer=tokenizer
        )
        if runtime.fast_enabled:
            from dataclasses import replace

            profile = load_warmup_profile(FAST_CONFIG)
            profile = replace(profile, codec_chunk_frames=runtime.codec_chunk_frames)
            manifest = runtime.warmup_from_profile(profile)
            print(f"fast warmup: {manifest['total_elapsed_ms']:.2f} ms", flush=True)
        _state.update(
            tokenizer=tokenizer,
            model=model,
            audio_tokenizer=audio_tokenizer,
            runtime=runtime,
            ready=True,
        )
        print(f"model loaded from {MODEL_PATH}, sample_rate={runtime.sample_rate}", flush=True)
    except Exception as exc:  # surface load failures via /health
        _state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"MODEL LOAD FAILED: {_state['error']}", flush=True)


@app.on_event("startup")
def _startup() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    threading.Thread(target=_load_model, daemon=True).start()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_INDEX, media_type="text/html")


@app.get("/health")
def health() -> JSONResponse:
    if _state.get("error"):
        return JSONResponse({"status": "error", "detail": _state["error"]}, status_code=500)
    if not _state["ready"]:
        return JSONResponse({"status": "loading"}, status_code=503)
    import torch

    model = _state["model"]
    device = str(next(model.parameters()).device)
    info = {
        "status": "ok",
        "sample_rate": _state["runtime"].sample_rate,
        "device": device,
        "fast_path": bool(getattr(_state["runtime"], "fast_enabled", False)),
        "asr_mode": "service+local" if ASR_SERVICE_URL else "local-only",
        "asr_local_loaded": _asr["model"] is not None,
    }
    if device.startswith("cuda"):
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 2**30, 2)
        info["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / 2**30, 2)
    return JSONResponse(info)


@app.post("/api/convert")
def convert(text: str = Form(...)) -> JSONResponse:
    converted = _cc_t2s.convert(text)
    return JSONResponse({"converted": converted, "changed": converted != text})


def _prune_outputs() -> None:
    wavs = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in wavs[MAX_OUTPUTS_KEPT:]:
        old.unlink(missing_ok=True)


# NOTE: plain `def` (not async) on purpose — the GPU work is synchronous, and a
# blocking `async def` would freeze the event loop so /health times out during
# generation and the service manager flags the service as "error".
@app.post("/api/tts")
def tts(
    text: str = Form(...),
    ref_text: str = Form(""),
    instruction: str = Form(""),
    convert_t2s: str = Form("1"),
    cfg_scale: float = Form(DEFAULT_CFG_SCALE),
    seed: int = Form(42),
    ref_audio: UploadFile | None = File(None),
) -> JSONResponse:
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Model is still loading.")
    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is empty.")
    if not np.isfinite(cfg_scale) or cfg_scale <= 0:
        raise HTTPException(status_code=400, detail="cfg_scale must be > 0.")

    ref_text = ref_text.strip()
    has_ref = ref_audio is not None and bool(ref_audio.filename)
    if has_ref != bool(ref_text):
        raise HTTPException(
            status_code=400,
            detail="Reference audio and its transcript must be provided together.",
        )

    spoken_text = _cc_t2s.convert(text) if convert_t2s == "1" else text
    # vocal-event brackets stay as-is; OpenCC only maps characters.

    if not _request_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another generation is running — try again shortly.")
    try:
        job_id = uuid.uuid4().hex[:12]
        reference_path: Path | None = None
        if has_ref:
            import io

            suffix = Path(ref_audio.filename).suffix or ".wav"
            payload = ref_audio.file.read()
            if not payload:
                raise HTTPException(status_code=400, detail="Reference audio is empty.")
            try:
                sf.info(io.BytesIO(payload))
            except Exception:
                # Upstream loads the reference with soundfile too — normalise
                # anything libsndfile can't open (m4a/aac/…) to WAV up front.
                payload = _transcode_to_wav(payload, suffix)
                suffix = ".wav"
            reference_path = UPLOAD_DIR / f"ref_{job_id}{suffix}"
            reference_path.write_bytes(payload)

        request = {
            "id": job_id,
            "text": spoken_text,
            "instruction": instruction.strip() or DEFAULT_INSTRUCTION,
            "speaker": "S0",
        }
        template_name = "tts_instruction"
        if reference_path is not None:
            request["ref_audio_path"] = str(reference_path)
            request["ref_text"] = _cc_t2s.convert(ref_text) if convert_t2s == "1" else ref_text
            template_name = "ref_edit_tata"

        set_all_seeds(seed)
        inputs = prepare_inputs(
            _state["tokenizer"],
            _state["audio_tokenizer"],
            _state["model"],
            [request],
            get_template(template_name),
            guidance_scale=cfg_scale,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )

        runtime = _state["runtime"]
        out_path = OUTPUT_DIR / f"breeze_{job_id}.wav"
        started = time.perf_counter()
        frames = 0
        with sf.SoundFile(
            out_path, mode="w", samplerate=runtime.sample_rate, channels=1, subtype="PCM_16"
        ) as f:
            for chunk in runtime.iter_audio_chunks(inputs, request_id=job_id):
                audio = np.clip(np.asarray(chunk.audio, dtype=np.float32), -1.0, 1.0)
                f.write(audio)
                frames += len(audio)
        elapsed = time.perf_counter() - started
        duration = frames / runtime.sample_rate
        _prune_outputs()
        return JSONResponse(
            {
                "id": job_id,
                "url": f"/api/audio/{out_path.name}",
                "sample_rate": runtime.sample_rate,
                "duration_sec": round(duration, 2),
                "elapsed_sec": round(elapsed, 2),
                "rtf": round(elapsed / duration, 3) if duration else None,
                "spoken_text": spoken_text,
                "converted": spoken_text != text,
            }
        )
    finally:
        if has_ref and reference_path is not None:
            reference_path.unlink(missing_ok=True)
        _request_lock.release()


# ---------------------------------------------------------------- ASR helpers
def _asr_service_up() -> bool:
    import urllib.request

    if not ASR_SERVICE_URL:  # empty → standalone mode: never probe
        return False
    try:
        with urllib.request.urlopen(f"{ASR_SERVICE_URL}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _asr_via_service(payload: bytes, filename: str) -> str:
    import json
    import urllib.request

    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"{ASR_SERVICE_URL}/transcribe?language=zh",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read()).get("text", "").strip()


def _transcode_to_wav(payload: bytes, suffix: str) -> bytes:
    """Decode formats libsndfile can't open (m4a/aac/…) to PCM16 WAV.

    Uses ffmpeg when available (any platform), else macOS's built-in
    afconvert. Sample rate and channel count are preserved — both callers
    downmix/resample themselves.
    """
    import shutil
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / f"in{suffix or '.bin'}"
        dst = Path(td) / "out.wav"
        src.write_bytes(payload)
        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-y", "-i", str(src), "-c:a", "pcm_s16le", str(dst)]
        elif sys.platform == "darwin":
            cmd = ["afconvert", "-f", "WAVE", "-d", "LEI16", str(src), str(dst)]
        else:
            raise HTTPException(
                status_code=400,
                detail="無法解碼音檔（請用 wav / mp3 / flac / ogg，"
                "或安裝 ffmpeg 以支援 m4a/aac）。",
            )
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0 or not dst.is_file():
            tail = proc.stderr.decode(errors="replace")[-300:]
            raise HTTPException(status_code=400, detail=f"無法解碼音檔：{tail}")
        return dst.read_bytes()


def _decode_to_16k(payload: bytes, filename: str = "") -> np.ndarray:
    import io

    import torch
    import torchaudio.functional as AF

    try:
        audio, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    except Exception:
        # libsndfile can't open it (e.g. iPhone 語音備忘錄的 m4a) — transcode.
        wav = _transcode_to_wav(payload, Path(filename).suffix)
        audio, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != 16000:
        mono = AF.resample(torch.from_numpy(mono), sr, 16000).numpy()
    return mono


def _asr_unload_if_idle() -> None:
    with _asr_lock:
        if _asr["model"] is not None and time.time() - _asr["last_used"] >= ASR_IDLE_UNLOAD_SEC:
            import torch

            _asr["model"] = None
            _asr["processor"] = None
            torch.cuda.empty_cache()
            print("local ASR unloaded after idle", flush=True)


def _asr_local(audio16k: np.ndarray) -> str:
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    with _asr_lock:
        if _asr["model"] is None:
            print(f"loading local Breeze ASR from {ASR_MODEL_PATH}", flush=True)
            dtype = torch.float16 if ASR_DEVICE.startswith("cuda") else torch.float32
            _asr["processor"] = WhisperProcessor.from_pretrained(ASR_MODEL_PATH)
            if safe_state_dict_load_needed():
                # Same Windows workaround as breeze_runtime.py.
                from transformers import GenerationConfig, WhisperConfig

                from breeze_runtime import load_safetensors_state_dict

                state_dict = load_safetensors_state_dict(ASR_MODEL_PATH)
                config = WhisperConfig.from_pretrained(ASR_MODEL_PATH)
                model = WhisperForConditionalGeneration.from_pretrained(
                    None, config=config, state_dict=state_dict, dtype=dtype
                )
                del state_dict
                # from_pretrained(None) skips generation_config.json — load it
                # explicitly so Whisper's language/task ids and suppress lists are right.
                model.generation_config = GenerationConfig.from_pretrained(ASR_MODEL_PATH)
            else:
                model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_PATH, dtype=dtype)
            _asr["model"] = model.to(ASR_DEVICE).eval()
        processor, model = _asr["processor"], _asr["model"]
        _asr["last_used"] = time.time()

        texts = []
        step = 30 * 16000
        for start in range(0, len(audio16k), step):
            chunk = audio16k[start : start + step]
            feats = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features
            feats = feats.to(ASR_DEVICE, dtype=model.dtype)
            with torch.no_grad():
                ids = model.generate(feats, language="zh", task="transcribe", max_new_tokens=440)
            text = processor.decode(ids[0], skip_special_tokens=True).strip()
            if text:
                texts.append(text)
        _asr["last_used"] = time.time()
        threading.Timer(ASR_IDLE_UNLOAD_SEC + 5, _asr_unload_if_idle).start()
        return " ".join(texts)


@app.post("/api/transcribe_ref")
def transcribe_ref(ref_audio: UploadFile = File(...)) -> JSONResponse:
    payload = ref_audio.file.read()
    if not payload:
        raise HTTPException(status_code=400, detail="Reference audio is empty.")
    filename = Path(ref_audio.filename or "ref.wav").name
    started = time.perf_counter()

    audio16k = _decode_to_16k(payload, filename)
    duration = len(audio16k) / 16000
    if duration > ASR_MAX_REF_SEC:
        raise HTTPException(
            status_code=400, detail=f"參考音檔 {duration:.0f}s 太長，請剪到 {ASR_MAX_REF_SEC}s 以內。"
        )

    source = "asr-service"
    text = ""
    if _asr_service_up():
        try:
            text = _asr_via_service(payload, filename)
        except Exception as exc:
            print(f"ASR service failed, falling back to local: {exc}", flush=True)
            source = "local"
    else:
        source = "local"
    if source == "local":
        if not ASR_MODEL_PATH.is_dir():
            raise HTTPException(
                status_code=503,
                detail="Breeze ASR 服務未啟動，且本機找不到 Breeze ASR 模型。",
            )
        # Serialise with TTS generation so the two never contend for the GPU.
        with _request_lock:
            text = _asr_local(audio16k)

    return JSONResponse(
        {
            "text": text,
            "source": source,
            "duration_sec": round(duration, 2),
            "elapsed_sec": round(time.perf_counter() - started, 2),
        }
    )


@app.get("/api/audio/{name}")
def audio(name: str, download: int = 0):
    if "/" in name or "\\" in name or ".." in name or not name.endswith(".wav"):
        raise HTTPException(status_code=400, detail="bad name")
    path = OUTPUT_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    headers = {"Content-Disposition": f'attachment; filename="{name}"'} if download else None
    return FileResponse(path, media_type="audio/wav", headers=headers)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
