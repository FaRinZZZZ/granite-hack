"""Thai text-to-speech via edge-tts.

Piper (used by the original code) has no usable Thai voice, and the bundled
piper.exe is a Windows binary that cannot run on macOS or Linux.
"""
from __future__ import annotations

import logging
import re

import edge_tts

from app import config

log = logging.getLogger(__name__)

THAI_VOICES = {
    "female": "th-TH-PremwadeeNeural",
    "male": "th-TH-NiwatNeural",
}

# Markdown and emoji get read out literally ("ดาวจันทร์..." for **), so strip them.
_MARKDOWN = re.compile(r"[*_`#>|~\[\]]+")
_EMOJI = re.compile(
    "[\U0001f300-\U0001faff\U00002600-\U000027bf\U0001f1e6-\U0001f1ff️]+"
)


def clean_for_speech(text: str) -> str:
    text = _MARKDOWN.sub("", text)
    text = _EMOJI.sub("", text)
    text = re.sub(r"https?://\S+", "ลิงก์", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


async def synthesize(text: str, voice: str | None = None) -> bytes:
    """Return MP3 bytes for the given text."""
    cleaned = clean_for_speech(text)
    if not cleaned:
        raise ValueError("nothing to speak")

    cfg = config.tts
    communicate = edge_tts.Communicate(
        cleaned,
        voice=THAI_VOICES.get(voice or "", voice or cfg.voice),
        rate=cfg.rate,
        volume=cfg.volume,
        pitch=cfg.pitch,
    )

    audio = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])

    if not audio:
        raise RuntimeError("edge-tts returned no audio")
    return bytes(audio)


async def list_thai_voices() -> list[dict]:
    voices = await edge_tts.list_voices()
    return [v for v in voices if v.get("Locale", "").startswith("th")]
