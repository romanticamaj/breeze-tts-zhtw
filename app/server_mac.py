"""macOS (Apple Silicon) launcher — run app/server.py on MPS (or CPU).

Upstream is CUDA-only in exactly two places that matter for the eager path:

  1. ``breeze_infer.runtime.resolve_device()`` falls back to "cpu", never "mps".
  2. ``FastBreezeStreamingRuntime.__init__`` raises
     "fast streaming requires a CUDA device" as its second-to-last statement —
     after every attribute is already set — even when all fast_* flags are off
     and only the eager path will run.

Per CLAUDE.md the submodule is never modified, so this launcher monkeypatches
both before importing app/server.py. The rest of the eager path is
device-generic: CUDA-graph capture code is only reached with fast flags on,
and the ``torch.cuda.*`` helpers (empty_cache / manual_seed_all guards) no-op
without CUDA.

Usage:  .venv/bin/python app/server_mac.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODELS_ROOT = Path(os.environ.get("BREEZE_MAC_MODELS_ROOT", Path.home() / "ai-models"))

# Defaults point at the Windows paths; override for this machine before
# server.py reads them at import time.
os.environ.setdefault("BREEZE_TTS2_MODEL_PATH", str(MODELS_ROOT / "breeze-tts-2"))
os.environ.setdefault("BREEZE_ASR_MODEL_PATH", str(MODELS_ROOT / "breeze-asr"))
# Whisper runs fine on MPS; server.py picks fp32 for any non-CUDA device.
os.environ.setdefault("BREEZE_ASR_DEVICE", "mps")
# Any op MPS is missing falls back to CPU instead of aborting the request.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

sys.path.insert(0, str(ROOT / "upstream" / "breeze-tts"))
sys.path.insert(0, str(APP_DIR))

import torch  # noqa: E402

import breeze_infer.runtime as _rt  # noqa: E402
from models.fast_streaming import FastBreezeStreamingRuntime  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def _resolve_device_mac(explicit_device: str | None = None) -> str:
    return explicit_device or DEVICE


_rt.resolve_device = _resolve_device_mac

_orig_init = FastBreezeStreamingRuntime.__init__


def _init_mac(self, *args, **kwargs):
    try:
        _orig_init(self, *args, **kwargs)
    except RuntimeError as exc:
        if "fast streaming requires a CUDA device" not in str(exc):
            raise
        if self.fast_enabled:
            raise  # the fast path really does need CUDA graphs
        # __init__ had already fully initialized self; redo the one check
        # that sat after the device check.
        if self.config.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0") from None


FastBreezeStreamingRuntime.__init__ = _init_mac

if os.environ.get("BREEZE_TTS2_FAST", "0") == "1":
    raise SystemExit("BREEZE_TTS2_FAST=1 needs CUDA graphs — not available on macOS.")

import server  # noqa: E402  (binds the patched resolve_device)

if __name__ == "__main__":
    import uvicorn

    print(f"breeze-tts-zhtw (mac) — device={DEVICE}", flush=True)
    uvicorn.run(server.app, host=server.HOST, port=server.PORT, log_level="info")
