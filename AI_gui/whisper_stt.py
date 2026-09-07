#!/usr/bin/env python3
"""Local Whisper speech-to-text with Korean/English auto detection.

The browser records one utterance at a time as 16 kHz mono PCM and posts it
here. We return both the transcript and the language it was spoken in, so the
GUI can answer in whichever language the operator used.
"""

from __future__ import annotations

import ctypes
import glob
import io
import os
import struct
import sysconfig
import threading
import time
import wave
from typing import Any

# ctranslate2 wheels link against the CUDA 12 runtime, but this environment's
# torch pulls in CUDA 13 (libcublas.so.13). Loading the CUDA 12 copies from
# nvidia-cublas-cu12/nvidia-cudnn-cu13 into the global namespace before
# faster_whisper imports satisfies ctranslate2 without an LD_LIBRARY_PATH
# wrapper around the launcher. Failures here are not fatal: the engine falls
# back to CPU.
_CUDA_PRELOAD_PATTERNS = (
    "nvidia/cublas/lib/libcublas.so.12",
    "nvidia/cublas/lib/libcublasLt.so.12",
    "nvidia/cudnn/lib/libcudnn*.so.9",
)


def preload_cuda_libraries() -> list[str]:
    """Best-effort dlopen of the CUDA 12 libraries ctranslate2 needs."""
    loaded: list[str] = []
    site_packages = sysconfig.get_paths()["purelib"]
    for pattern in _CUDA_PRELOAD_PATTERNS:
        for path in sorted(glob.glob(os.path.join(site_packages, pattern))):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                loaded.append(os.path.basename(path))
            except OSError:
                pass
    return loaded


def decode_wav(data: bytes) -> tuple[Any, float]:
    """Decode 16-bit PCM WAV bytes into mono float32 samples at 16 kHz."""
    import numpy as np

    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError, struct.error) as exc:
        # A truncated or non-WAV upload must read as a bad request, not as a
        # crash in the recognition thread.
        raise ValueError(
            f"not a readable 16-bit PCM WAV ({str(exc) or type(exc).__name__})"
        ) from exc

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM, got {sample_width * 8}-bit")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        # Linear resample; the browser already targets 16 kHz, so this is a
        # safety net rather than the normal path.
        target_length = int(round(samples.size * 16000 / rate))
        if target_length <= 0:
            return samples[:0], 0.0
        source_index = np.linspace(0, samples.size - 1, target_length, dtype=np.float32)
        samples = np.interp(source_index, np.arange(samples.size), samples).astype(np.float32)
    return samples, samples.size / 16000.0


class WhisperSTT:
    """Lazily loaded faster-whisper model shared across requests."""

    def __init__(self, config: dict[str, Any], logger: Any = None) -> None:
        settings = dict(config.get("stt", {}))
        self.enabled = bool(settings.get("enabled", True))
        self.model_name = str(settings.get("model", "small"))
        self.requested_device = str(settings.get("device", "cuda"))
        self.requested_compute_type = str(settings.get("compute_type", "int8_float16"))
        self.cpu_threads = int(settings.get("cpu_threads", 8))
        self.beam_size = int(settings.get("beam_size", 1))
        self.vad_filter = bool(settings.get("vad_filter", True))
        self.no_speech_threshold = float(settings.get("no_speech_threshold", 0.6))
        self.min_avg_logprob = float(settings.get("min_avg_logprob", -1.0))
        self.max_audio_seconds = float(settings.get("max_audio_seconds", 15.0))
        self.min_audio_seconds = float(settings.get("min_audio_seconds", 0.20))
        self.languages = [str(item) for item in settings.get("languages", ["ko", "en"])]
        self.initial_prompt = dict(settings.get("initial_prompt", {}))
        self.logger = logger

        self.device = self.requested_device
        self.compute_type = self.requested_compute_type
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._state = "disabled" if not self.enabled else "idle"
        self._detail = ""

    # -- state ---------------------------------------------------------------

    def _log(self, message: str, level: str = "info") -> None:
        if self.logger is not None:
            self.logger(message, level)

    def _set_state(self, state: str, detail: str = "") -> None:
        with self._state_lock:
            self._state = state
            self._detail = detail

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "state": self._state,
                "detail": self._detail,
                "model": self.model_name,
                "device": self.device,
                "compute_type": self.compute_type,
                "languages": list(self.languages),
            }

    # -- model ---------------------------------------------------------------

    def warm_up_async(self) -> None:
        if not self.enabled:
            return
        threading.Thread(target=self._warm_up, name="whisper-warmup", daemon=True).start()

    def _warm_up(self) -> None:
        try:
            import numpy as np

            model = self._ensure_model()
            # A short silent buffer forces the encoder/decoder graphs to build
            # so the operator's first real command is not the slow one.
            with self._model_lock:
                list(
                    model.transcribe(
                        np.zeros(16000, dtype=np.float32),
                        language=self.languages[0],
                        beam_size=1,
                    )[0]
                )
            self._set_state("ready")
            self._log(f"Whisper STT ready ({self.model_name} · {self.device} · {self.compute_type}).")
        except Exception as exc:  # pragma: no cover - environment specific
            self._set_state("error", str(exc))
            self._log(f"Whisper STT unavailable: {exc}", "error")

    def _ensure_model(self) -> Any:
        with self._model_lock:
            if self._model is not None:
                return self._model
            self._set_state("loading")
            preload_cuda_libraries()
            from faster_whisper import WhisperModel

            attempts = [(self.requested_device, self.requested_compute_type)]
            if self.requested_device != "cpu":
                attempts.append(("cpu", "int8"))

            last_error: Exception | None = None
            for device, compute_type in attempts:
                try:
                    model = WhisperModel(
                        self.model_name,
                        device=device,
                        compute_type=compute_type,
                        cpu_threads=self.cpu_threads,
                    )
                except Exception as exc:
                    last_error = exc
                    self._log(
                        f"Whisper STT could not start on {device}/{compute_type}: {exc}",
                        "warning",
                    )
                    continue
                if device != self.requested_device:
                    self._log(
                        f"Whisper STT fell back to {device}/{compute_type}; recognition will be slower.",
                        "warning",
                    )
                self.device = device
                self.compute_type = compute_type
                self._model = model
                return model
            raise RuntimeError(last_error or "Whisper model could not be loaded")

    # -- transcription -------------------------------------------------------

    def _decode(self, model: Any, audio: Any, language: str | None) -> tuple[str, Any]:
        segments, info = model.transcribe(
            audio,
            language=language,
            beam_size=self.beam_size,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=self.vad_filter,
            initial_prompt=self.initial_prompt.get(language) if language else None,
        )
        # Whisper invents filler ("you", "감사합니다") when handed room tone, and
        # an always-open microphone hands it plenty. Drop segments the model
        # itself is unsure about rather than acting on them.
        kept = [
            segment.text.strip()
            for segment in segments
            if getattr(segment, "no_speech_prob", 0.0) <= self.no_speech_threshold
            and getattr(segment, "avg_logprob", 0.0) >= self.min_avg_logprob
        ]
        return " ".join(part for part in kept if part).strip(), info

    def _preferred_language(self, info: Any) -> str | None:
        """Pick the most likely of the configured languages, ignoring the rest."""
        probabilities = getattr(info, "all_language_probs", None)
        if not probabilities:
            return None
        allowed = {language: 0.0 for language in self.languages}
        for language, probability in probabilities:
            if language in allowed:
                allowed[language] = probability
        best = max(allowed, key=lambda key: allowed[key])
        return best if allowed[best] > 0.0 else None

    def transcribe(self, wav_bytes: bytes, language: str | None = None) -> dict[str, Any]:
        if not self.enabled:
            return {"ok": False, "reason": "disabled"}

        started = time.time()
        audio, duration = decode_wav(wav_bytes)
        if duration < self.min_audio_seconds:
            return {"ok": False, "reason": "too_short", "audio_seconds": duration}
        if duration > self.max_audio_seconds:
            audio = audio[: int(self.max_audio_seconds * 16000)]
            duration = self.max_audio_seconds

        model = self._ensure_model()
        forced = language if language in self.languages else None

        with self._model_lock:
            text, info = self._decode(model, audio, forced)
            detected = forced or getattr(info, "language", None)
            probability = float(getattr(info, "language_probability", 0.0) or 0.0)

            if forced is None:
                preferred = self._preferred_language(info)
                # Whisper occasionally lands on a language we do not support
                # (Japanese for Korean, Dutch for English). Redo the pass with
                # the best supported language rather than returning nonsense.
                if preferred and preferred != detected:
                    text, info = self._decode(model, audio, preferred)
                    detected = preferred
                    probability = float(getattr(info, "language_probability", 0.0) or 0.0)

        self._set_state("ready")
        if detected not in self.languages:
            detected = self.languages[0]
        return {
            "ok": True,
            "text": text,
            "language": detected,
            "language_probability": probability,
            "audio_seconds": round(duration, 3),
            "elapsed_ms": int((time.time() - started) * 1000),
        }
