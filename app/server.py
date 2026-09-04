"""FastAPI service: Thai speech in, RAG answer out, Thai speech back.

    POST /transcribe   audio            -> transcript
    POST /chat         text             -> answer + sources
    POST /chat/stream  text             -> SSE token stream
    POST /speak        text             -> audio/mpeg
    POST /voice        audio            -> the whole loop in one call
"""
from __future__ import annotations

import base64
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from app import audio as audio_utils
from app import config, llm, rag, stt, tts

logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm everything at boot so the first real request isn't a 40-second stall.
    log.info("warming up models…")
    try:
        await run_in_threadpool(stt.get_stt)
        await run_in_threadpool(rag.get_store)
        log.info("ready")
    except Exception as exc:
        log.warning("warm-up incomplete: %s", exc)
    yield


app = FastAPI(title="Thai Voice RAG Assistant", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before exposing this beyond localhost
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str = "default"


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1)
    voice: str | None = None


def _sources(hits: list[rag.Hit]) -> list[dict]:
    return [
        {
            "source": h.chunk.source,
            "page": h.chunk.page,
            "score": round(h.score, 4),
            "preview": h.chunk.text[:200],
        }
        for h in hits
    ]


async def _read_audio(file: UploadFile):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty audio upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"audio exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
    try:
        return await run_in_threadpool(audio_utils.decode, data)
    except Exception as exc:
        raise HTTPException(400, f"could not decode audio: {exc}") from exc


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "stt_model": config.stt.model_id,
        "stt_backend": config.stt.backend,
        "llm_model": config.llm.model,
        "embedding_model": config.rag.embedding_model,
        "tts_voice": config.tts.voice,
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    samples = await _read_audio(file)
    engine = await run_in_threadpool(stt.get_stt)
    result = await run_in_threadpool(engine.transcribe, samples)
    return {
        "text": result.text,
        "language": result.language,
        "duration": audio_utils.duration_seconds(samples),
    }


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        answer, hits = await llm.answer(request.session_id, request.message)
    except (FileNotFoundError, RuntimeError) as exc:
        # Missing index or missing API key — a setup problem, not a bad request.
        raise HTTPException(503, str(exc)) from exc
    return {"answer": answer, "sources": _sources(hits), "session_id": request.session_id}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def events():
        try:
            async for delta in llm.stream_answer(request.session_id, request.message):
                yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
        except Exception as exc:
            log.exception("stream failed")
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/speak")
async def speak(request: SpeakRequest):
    try:
        mp3 = await tts.synthesize(request.text, request.voice)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return Response(content=mp3, media_type="audio/mpeg")


@app.post("/voice")
async def voice(file: UploadFile = File(...), session_id: str = "default"):
    """Full turn: speech -> transcript -> RAG answer -> speech."""
    samples = await _read_audio(file)
    engine = await run_in_threadpool(stt.get_stt)
    transcript = await run_in_threadpool(engine.transcribe, samples)

    if not transcript.text:
        raise HTTPException(422, "no speech detected in the audio")

    try:
        answer, hits = await llm.answer(session_id, transcript.text)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(503, str(exc)) from exc

    try:
        mp3 = await tts.synthesize(answer)
        audio_b64 = base64.b64encode(mp3).decode()
    except Exception as exc:
        log.warning("TTS failed, returning text only: %s", exc)
        audio_b64 = None

    return {
        "transcript": transcript.text,
        "answer": answer,
        "sources": _sources(hits),
        "audio_base64": audio_b64,
        "audio_mime": "audio/mpeg",
        "session_id": session_id,
    }


@app.post("/session/reset")
async def session_reset(session_id: str = "default"):
    llm.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}


@app.get("/voices")
async def voices():
    return await tts.list_thai_voices()
