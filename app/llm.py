"""Typhoon LLM (OpenTyphoon API) + the RAG conversation loop."""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from functools import lru_cache

from openai import AsyncOpenAI

from app import config
from app.rag import Hit, format_context, get_store

log = logging.getLogger(__name__)

NO_ANSWER = "ขออภัยค่ะ ไม่มีข้อมูลเรื่องนี้ในระบบ"


@lru_cache(maxsize=1)
def get_client() -> AsyncOpenAI:
    if not config.llm.api_key:
        raise RuntimeError(
            "TYPHOON_API_KEY is not set. Get a key from https://playground.opentyphoon.ai "
            "and put it in .env"
        )
    # OpenTyphoon speaks the OpenAI wire format, so the official SDK works unchanged.
    return AsyncOpenAI(api_key=config.llm.api_key, base_url=config.llm.base_url)


@dataclass
class Turn:
    role: str
    content: str


@dataclass
class ChatSession:
    """One conversation. History is bounded so context can't grow without limit —
    the old ResponseGenerator appended every full RAG prompt to a list it never
    trimmed, which eventually blew past the context window."""

    session_id: str
    history: list[Turn] = field(default_factory=list)

    def retrieval_query(self) -> str:
        """Follow-ups like 'แล้วราคาล่ะ' are meaningless alone, so fold in the
        previous user turn. Cheaper and lower-latency than an LLM condense step."""
        users = [t.content for t in self.history if t.role == "user"]
        return " ".join(users[-2:]) if len(users) > 1 else (users[-1] if users else "")

    def messages(self, context: str) -> list[dict]:
        recent = self.history[-(config.llm.history_turns * 2) :]
        msgs: list[dict] = [{"role": "system", "content": config.SYSTEM_PROMPT}]
        # Everything except the final user turn goes in as plain conversation.
        for turn in recent[:-1]:
            msgs.append({"role": turn.role, "content": turn.content})
        question = recent[-1].content if recent else ""
        msgs.append(
            {
                "role": "user",
                "content": f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {question}",
            }
        )
        return msgs


_SESSIONS: dict[str, ChatSession] = {}


def get_session(session_id: str) -> ChatSession:
    return _SESSIONS.setdefault(session_id, ChatSession(session_id=session_id))


def reset_session(session_id: str) -> None:
    _SESSIONS.pop(session_id, None)


def retrieve(session: ChatSession) -> list[Hit]:
    cfg = config.rag
    return get_store().search(session.retrieval_query(), cfg.top_k, cfg.min_score)


async def answer(session_id: str, question: str) -> tuple[str, list[Hit]]:
    """Non-streaming answer. Returns (text, sources)."""
    session = get_session(session_id)
    session.history.append(Turn("user", question))
    hits = retrieve(session)

    if not hits:
        # Nothing above the similarity floor — refuse rather than let the model invent.
        session.history.append(Turn("assistant", NO_ANSWER))
        return NO_ANSWER, []

    response = await get_client().chat.completions.create(
        model=config.llm.model,
        messages=session.messages(format_context(hits)),
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        top_p=config.llm.top_p,
    )
    text = (response.choices[0].message.content or "").strip()
    session.history.append(Turn("assistant", text))
    return text, hits


async def stream_answer(session_id: str, question: str) -> AsyncIterator[str]:
    """Token stream. Lets the caller start TTS on the first sentence instead of
    waiting for the whole reply."""
    session = get_session(session_id)
    session.history.append(Turn("user", question))
    hits = retrieve(session)

    if not hits:
        session.history.append(Turn("assistant", NO_ANSWER))
        yield NO_ANSWER
        return

    stream = await get_client().chat.completions.create(
        model=config.llm.model,
        messages=session.messages(format_context(hits)),
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        top_p=config.llm.top_p,
        stream=True,
    )

    parts: list[str] = []
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            parts.append(delta)
            yield delta

    session.history.append(Turn("assistant", "".join(parts).strip()))
