"""Thai-aware RAG: document loading, chunking, BGE-M3 embeddings, FAISS retrieval.

Deliberately does not use LangChain. The original code depended on
`langchain.vectorstores` / `ConversationalRetrievalChain`, both of which have been
removed from modern LangChain, and the abstraction bought nothing here.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app import config

log = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
META_FILE = "meta.json"


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None = None


@dataclass
class Hit:
    chunk: Chunk
    score: float


# ---------------------------------------------------------------- loading


def _read_pdf(path: Path) -> list[tuple[str, int]]:
    """Return [(page_text, page_number)]. Prefers PyMuPDF — pypdf mangles Thai
    glyph ordering on a lot of real-world PDFs."""
    try:
        import pymupdf

        with pymupdf.open(path) as doc:
            return [(page.get_text("text"), i + 1) for i, page in enumerate(doc)]
    except ImportError:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(page.extract_text() or "", i + 1) for i, page in enumerate(reader.pages)]


def load_documents(data_dir: Path) -> list[Chunk]:
    """Load every .pdf, .txt and .md under data_dir into page-level chunks."""
    raw: list[Chunk] = []
    if not data_dir.exists():
        raise FileNotFoundError(f"data directory not found: {data_dir}")

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        source = str(path.relative_to(data_dir))

        if suffix == ".pdf":
            for text, page in _read_pdf(path):
                if text.strip():
                    raw.append(Chunk(text=text, source=source, page=page))
        elif suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                raw.append(Chunk(text=text, source=source))

    if not raw:
        raise ValueError(f"no readable .pdf/.txt/.md files in {data_dir}")
    log.info("loaded %d pages from %s", len(raw), data_dir)
    return raw


# ---------------------------------------------------------------- chunking

# Thai has no spaces between words, so splitting on whitespace produces garbage.
# Split on structural breaks instead, then fall back to a hard character cut.
_BREAKS = re.compile(r"(?<=[\n。．.!?！？;；])|(?<=ๆ\s)")


def _normalise(text: str) -> str:
    text = text.replace("​", "").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    # An overlap >= chunk_size would make the hard-cut loop below never advance.
    overlap = max(0, min(overlap, chunk_size // 2))
    text = _normalise(text)
    if len(text) <= chunk_size:
        return [text] if text else []

    pieces = [p for p in _BREAKS.split(text) if p and p.strip()]
    chunks: list[str] = []
    buf = ""

    for piece in pieces:
        # A single piece longer than the window gets hard-cut.
        while len(piece) > chunk_size:
            if buf:
                chunks.append(buf.strip())
                buf = ""
            chunks.append(piece[:chunk_size].strip())
            piece = piece[chunk_size - overlap :]

        if len(buf) + len(piece) <= chunk_size:
            buf += piece
        else:
            chunks.append(buf.strip())
            buf = (buf[-overlap:] if overlap else "") + piece

    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if c]


def chunk_documents(pages: list[Chunk], cfg: config.RAGConfig) -> list[Chunk]:
    out: list[Chunk] = []
    for page in pages:
        for piece in split_text(page.text, cfg.chunk_size, cfg.chunk_overlap):
            out.append(Chunk(text=piece, source=page.source, page=page.page))
    log.info("split into %d chunks", len(out))
    return out


# ---------------------------------------------------------------- embeddings


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    cfg = config.rag
    device = _resolve_device(cfg.embedding_device)
    log.info("loading embeddings %s on %s", cfg.embedding_model, device)
    return SentenceTransformer(cfg.embedding_model, device=device)


def embed(texts: list[str], is_query: bool = False) -> np.ndarray:
    """L2-normalised embeddings, so inner product == cosine similarity."""
    model = get_embedder()
    vectors = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=not is_query and len(texts) > 64,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype="float32")


# ---------------------------------------------------------------- store


class VectorStore:
    def __init__(self, index, chunks: list[Chunk]):
        self.index = index
        self.chunks = chunks

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "VectorStore":
        import faiss

        vectors = embed([c.text for c in chunks])
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index, chunks)

    def save(self, directory: Path) -> None:
        import faiss

        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        (directory / CHUNKS_FILE).write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        (directory / META_FILE).write_text(
            json.dumps(
                {
                    "embedding_model": config.rag.embedding_model,
                    "chunk_size": config.rag.chunk_size,
                    "chunk_overlap": config.rag.chunk_overlap,
                    "count": len(self.chunks),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("saved index (%d chunks) to %s", len(self.chunks), directory)

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        import faiss

        index_path = directory / INDEX_FILE
        if not index_path.exists():
            raise FileNotFoundError(
                f"no index at {directory}. Build one first: python -m scripts.ingest"
            )

        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        if meta.get("embedding_model") != config.rag.embedding_model:
            raise ValueError(
                f"index was built with {meta.get('embedding_model')!r} but "
                f"EMBEDDING_MODEL is {config.rag.embedding_model!r}. Re-run the ingest script."
            )

        index = faiss.read_index(str(index_path))
        raw = json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))
        return cls(index, [Chunk(**c) for c in raw])

    def search(self, query: str, top_k: int, min_score: float) -> list[Hit]:
        vector = embed([query], is_query=True)
        scores, indices = self.index.search(vector, min(top_k, len(self.chunks)))
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_score:
                continue
            hits.append(Hit(chunk=self.chunks[idx], score=float(score)))
        return hits


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    return VectorStore.load(config.rag.index_dir)


def build_index(data_dir: Path | None = None, index_dir: Path | None = None) -> VectorStore:
    cfg = config.rag
    pages = load_documents(data_dir or cfg.data_dir)
    chunks = chunk_documents(pages, cfg)
    store = VectorStore.build(chunks)
    store.save(index_dir or cfg.index_dir)
    get_store.cache_clear()
    return store


def format_context(hits: list[Hit]) -> str:
    """Render retrieved chunks for the prompt, numbered so the model can cite them."""
    blocks = []
    for i, hit in enumerate(hits, 1):
        where = f"{hit.chunk.source}" + (f" หน้า {hit.chunk.page}" if hit.chunk.page else "")
        blocks.append(f"[{i}] ({where})\n{hit.chunk.text}")
    return "\n\n".join(blocks)
