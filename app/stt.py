"""Thai speech-to-text backed by Typhoon Whisper Turbo."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app import config
from app.audio import TARGET_SR

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    language: str
    duration: float | None = None


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TransformersSTT:
    """Runs the HF checkpoint directly. Portable: CUDA, Apple MPS, or plain CPU."""

    def __init__(self, cfg: config.STTConfig):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.cfg = cfg
        self.device = _resolve_device(cfg.device)
        # fp16 on GPU/MPS, fp32 on CPU — fp16 on CPU is slower, not faster.
        dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32

        log.info("loading %s on %s (%s)", cfg.model_id, self.device, dtype)
        # `dtype` replaced `torch_dtype` in transformers 4.56; hence the floor in
        # requirements-thai.txt.
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            cfg.model_id, dtype=dtype, low_cpu_mem_usage=True
        ).to(self.device)
        processor = AutoProcessor.from_pretrained(cfg.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=dtype,
            device=self.device,
            chunk_length_s=cfg.chunk_length_s,
        )

    def transcribe(self, audio: np.ndarray | str | Path) -> Transcript:
        payload = (
            {"array": audio, "sampling_rate": TARGET_SR}
            if isinstance(audio, np.ndarray)
            else str(audio)
        )
        result = self.pipe(
            payload,
            batch_size=self.cfg.batch_size,
            generate_kwargs={"language": self.cfg.language, "task": "transcribe"},
        )
        return Transcript(text=result["text"].strip(), language=self.cfg.language)


class FasterWhisperSTT:
    """CTranslate2 backend. Faster on CPU, but needs a converted model directory.

    Convert once with:
        ct2-transformers-converter --model scb10x/typhoon-whisper-turbo \\
            --output_dir models/typhoon-whisper-turbo-ct2 --quantization int8
    then set STT_MODEL to that directory.
    """

    def __init__(self, cfg: config.STTConfig):
        from faster_whisper import WhisperModel

        self.cfg = cfg
        device = _resolve_device(cfg.device)
        # CTranslate2 has no MPS backend; Apple Silicon falls back to CPU int8.
        if device == "mps":
            device = "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        log.info("loading %s via faster-whisper on %s", cfg.model_id, device)
        self.model = WhisperModel(cfg.model_id, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray | str | Path) -> Transcript:
        segments, info = self.model.transcribe(
            audio if isinstance(audio, np.ndarray) else str(audio),
            language=self.cfg.language,
            vad_filter=True,
            beam_size=5,
        )
        text = "".join(segment.text for segment in segments).strip()
        return Transcript(text=text, language=info.language, duration=info.duration)


@lru_cache(maxsize=1)
def get_stt() -> TransformersSTT | FasterWhisperSTT:
    """Load the model once and reuse it — loading takes tens of seconds."""
    cfg = config.stt
    if cfg.backend == "faster-whisper":
        return FasterWhisperSTT(cfg)
    if cfg.backend == "transformers":
        return TransformersSTT(cfg)
    raise ValueError(f"unknown STT_BACKEND: {cfg.backend!r}")
