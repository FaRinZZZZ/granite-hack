"""Central configuration, read from environment (.env)."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ROOT / "storage"))


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class STTConfig:
    # Whisper Large v3 Turbo fine-tuned on ~11k hours of Thai. Note the org is
    # typhoon-ai, not scb10x — the model card text still says scb10x, but that repo 404s.
    model_id: str = os.getenv("STT_MODEL", "typhoon-ai/typhoon-whisper-turbo")
    # "transformers" runs the HF checkpoint directly (works on MPS/CPU/CUDA).
    # "faster-whisper" needs a CTranslate2-converted copy; point STT_MODEL at that directory.
    backend: str = os.getenv("STT_BACKEND", "transformers")
    device: str = os.getenv("STT_DEVICE", "auto")
    language: str = os.getenv("STT_LANGUAGE", "th")
    # Whisper is trained on 30s windows; longer clips need chunking.
    chunk_length_s: int = int(os.getenv("STT_CHUNK_LENGTH_S", "30"))
    batch_size: int = int(os.getenv("STT_BATCH_SIZE", "8"))


@dataclass
class LLMConfig:
    api_key: str = os.getenv("TYPHOON_API_KEY", "")
    base_url: str = os.getenv("TYPHOON_BASE_URL", "https://api.opentyphoon.ai/v1")
    model: str = os.getenv("LLM_MODEL", "typhoon-v2.5-30b-a3b-instruct")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.5"))
    # Thai burns roughly 2-3x the tokens English does; the old 69-token cap truncated
    # answers mid-sentence.
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "800"))
    top_p: float = float(os.getenv("LLM_TOP_P", "0.9"))
    # How many past user/assistant turns to replay. Keeps context from growing forever.
    history_turns: int = int(os.getenv("LLM_HISTORY_TURNS", "6"))


@dataclass
class RAGConfig:
    # BGE-M3 is multilingual and handles Thai well. The old IBM SLATE 125M was
    # English-only, which silently destroys retrieval on a Thai knowledge base.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "auto")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))
    # Cosine similarity below this is treated as "not in the knowledge base".
    min_score: float = float(os.getenv("RAG_MIN_SCORE", "0.35"))
    index_dir: Path = field(default_factory=lambda: INDEX_DIR)
    data_dir: Path = field(default_factory=lambda: DATA_DIR)


@dataclass
class TTSConfig:
    # th-TH-PremwadeeNeural (female) / th-TH-NiwatNeural (male)
    voice: str = os.getenv("TTS_VOICE", "th-TH-PremwadeeNeural")
    rate: str = os.getenv("TTS_RATE", "+0%")
    volume: str = os.getenv("TTS_VOLUME", "+0%")
    pitch: str = os.getenv("TTS_PITCH", "+0Hz")


SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    """คุณคือผู้ช่วยตอบคำถามภาษาไทยของ Granite Supercenter

กติกาที่ต้องทำตามอย่างเคร่งครัด:
1. ตอบโดยอ้างอิงจาก "ข้อมูลอ้างอิง" ที่ให้มาเท่านั้น
2. ถ้าข้อมูลอ้างอิงไม่มีคำตอบ ให้บอกตรง ๆ ว่า "ขออภัยค่ะ ไม่มีข้อมูลเรื่องนี้ในระบบ" ห้ามเดาหรือแต่งขึ้นเอง
3. ตอบเป็นภาษาไทยเสมอ
4. คำตอบจะถูกนำไปอ่านออกเสียง จึงต้องสั้น กระชับ 1-3 ประโยค เป็นประโยคพูดที่อ่านออกเสียงได้ลื่น
5. ห้ามใส่ markdown, bullet, emoji, หรือสัญลักษณ์พิเศษใด ๆ ในคำตอบ""",
)

stt = STTConfig()
llm = LLMConfig()
rag = RAGConfig()
tts = TTSConfig()
DEBUG = _bool("DEBUG", False)
