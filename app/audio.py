"""Decode arbitrary uploaded audio into the 16 kHz mono float32 Whisper expects."""
from __future__ import annotations

import io
import logging
import shutil
import subprocess

import numpy as np

log = logging.getLogger(__name__)

TARGET_SR = 16_000


def _resample(audio: np.ndarray, source_sr: int) -> np.ndarray:
    if source_sr == TARGET_SR:
        return audio
    # Linear interpolation. Good enough for speech at these rates and avoids
    # pulling in librosa/scipy just for this.
    duration = audio.shape[0] / source_sr
    target_len = int(round(duration * TARGET_SR))
    return np.interp(
        np.linspace(0.0, duration, target_len, endpoint=False),
        np.linspace(0.0, duration, audio.shape[0], endpoint=False),
        audio,
    ).astype("float32")


def _decode_with_ffmpeg(data: bytes) -> np.ndarray:
    """Handles webm/opus/m4a — what browsers actually record."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "cannot decode this audio format without ffmpeg. Install it "
            "(brew install ffmpeg) or upload WAV/FLAC instead."
        )
    process = subprocess.run(
        ["ffmpeg", "-nostdin", "-threads", "1", "-i", "pipe:0",
         "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SR), "pipe:1"],
        input=data,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0 or not process.stdout:
        raise RuntimeError(f"ffmpeg failed to decode audio: {process.stderr[-500:].decode(errors='replace')}")
    return np.frombuffer(process.stdout, dtype="float32").copy()


def decode(data: bytes) -> np.ndarray:
    """Bytes of any common audio container -> mono float32 @ 16 kHz."""
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:  # unsupported container (webm, m4a, mp3 on some builds)
        log.debug("soundfile could not read audio (%s); falling back to ffmpeg", exc)
        return _decode_with_ffmpeg(data)

    mono = audio.mean(axis=1)  # downmix
    return _resample(mono, sample_rate)


def duration_seconds(audio: np.ndarray) -> float:
    return float(audio.shape[0]) / TARGET_SR
