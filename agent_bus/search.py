from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from typing import Any, Literal

from agent_bus.db import AgentBusDB

SearchMode = Literal["fts", "semantic", "hybrid"]

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 1200  # characters
DEFAULT_CHUNK_OVERLAP = 200  # characters


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_index: int
    start_char: int
    end_char: int
    text: str


def sha256_hex(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def chunk_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    raw = text
    n = len(raw)
    if n == 0:
        return []

    chunks: list[TextChunk] = []
    start = 0
    idx = 0

    while start < n:
        end = min(n, start + chunk_size)

        # Prefer to split on paragraph boundaries to preserve semantics.
        if end < n:
            boundary = raw.rfind("\n\n", start, end)
            if boundary == -1:
                boundary = raw.rfind("\n", start, end)
            if boundary != -1 and boundary > start + int(chunk_size * 0.6):
                end = boundary

        chunk = raw[start:end].strip()
        if chunk:
            chunks.append(TextChunk(chunk_index=idx, start_char=start, end_char=end, text=chunk))
            idx += 1

        if end >= n:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def _semantic_deps():
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    return np, SentenceTransformer


@lru_cache(maxsize=2)
def _load_sentence_transformer(model: str):
    deps = _semantic_deps()
    if deps is None:
        raise RuntimeError(
            "Semantic search dependencies not installed. Install with: uv sync --extra semantic"
        )
    _, SentenceTransformer = deps
    return SentenceTransformer(model)


def _embed_query(model: str, query: str):
    np, _ = _semantic_deps()  # type: ignore[assignment]
    st = _load_sentence_transformer(model)
    emb = st.encode([query], normalize_embeddings=True)
    vec = np.asarray(emb[0], dtype=np.float32)
    return vec


def _cosine_scores(query_vec, candidate_rows: list[dict[str, Any]]):
    np, _ = _semantic_deps()  # type: ignore[assignment]
    if not candidate_rows:
        return np.array([], dtype=np.float32)

    dims = int(candidate_rows[0]["dims"])
    vectors = [
        np.frombuffer(c["vector"], dtype=np.float32, count=dims)  # type: ignore[arg-type]
        for c in candidate_rows
    ]
    mat = np.vstack(vectors)
    return mat @ query_vec


def _semantic_best_by_message(
    db: AgentBusDB,
    *,
    query: str,
    model: str,
    topic_id: str | None,
    message_ids: list[str] | None,
) -> dict[str, dict[str, Any]]:
    query_vec = _embed_query(model, query)
    candidates = db.list_chunk_embedding_candidates(
        model=model, topic_id=topic_id, message_ids=message_ids
    )
    if not candidates:
        return {}

    scores = _cosine_scores(query_vec, candidates)

    best: dict[str, dict[str, Any]] = {}
    for cand, score in zip(candidates, scores, strict=True):
        mid = cand["message_id"]
        s = float(score)
        existing = best.get(mid)
        if existing is None or s > float(existing["semantic_score"]):
            best[mid] = {**cand, "semantic_score": s}
    return best


def _semantic_snippet(db: AgentBusDB, *, message_id: str, start_char: int, end_char: int) -> str:
    msg = db.get_message_by_id(message_id=message_id)
    raw = msg.content_markdown
    start = max(0, min(len(raw), start_char))
    end = max(0, min(len(raw), end_char))
    if end <= start:
        return raw[:200].replace("\n", " ")
    snippet = raw[start:end].strip().replace("\n", " ")
    if len(snippet) > 240:
        snippet = snippet[:240] + "…"
    return snippet


def search_messages(
    db: AgentBusDB,
    *,
    query: str,
    mode: SearchMode = "hybrid",
    topic_id: str | None = None,
    limit: int = 20,
    model: str = DEFAULT_EMBEDDING_MODEL,
    fts_candidates: int = 100,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if fts_candidates <= 0:
        raise ValueError("fts_candidates must be > 0")

    warnings: list[str] = []

    if mode == "fts":
        return db.search_messages_fts(query=query, topic_id=topic_id, limit=limit), warnings

    if mode not in {"semantic", "hybrid"}:
        raise ValueError("mode must be one of: fts, semantic, hybrid")

    # Semantic or hybrid: require deps, but keep hybrid usable even if deps are missing.
    if _semantic_deps() is None:
        if mode == "hybrid":
            warnings.append(
                "semantic_deps_missing: install with `uv sync --extra semantic` for reranking"
            )
            return db.search_messages_fts(query=query, topic_id=topic_id, limit=limit), warnings
        raise RuntimeError("Semantic search dependencies not installed.")

    if mode == "semantic":
        best = _semantic_best_by_message(
            db, query=query, model=model, topic_id=topic_id, message_ids=None
        )
        ranked = sorted(best.values(), key=lambda r: float(r["semantic_score"]), reverse=True)[
            :limit
        ]
        out: list[dict[str, Any]] = []
        for r in ranked:
            out.append(
                {
                    "topic_id": r["topic_id"],
                    "topic_name": r["topic_name"],
                    "message_id": r["message_id"],
                    "seq": r["seq"],
                    "sender": r["sender"],
                    "message_type": r["message_type"],
                    "created_at": r["created_at"],
                    "semantic_score": r["semantic_score"],
                    "snippet": _semantic_snippet(
                        db,
                        message_id=r["message_id"],
                        start_char=int(r["start_char"]),
                        end_char=int(r["end_char"]),
                    ),
                }
            )
        return out, warnings

    # Hybrid: lexical candidate set (FTS) + semantic rerank if embeddings exist.
    try:
        fts = db.search_messages_fts(query=query, topic_id=topic_id, limit=fts_candidates)
    except RuntimeError as e:
        warnings.append(f"fts_unavailable: {e}")
        fts = []

    if not fts:
        # No lexical hits: fall back to pure semantic.
        warnings.append("hybrid_fallback_semantic: no lexical matches")
        return search_messages(
            db,
            query=query,
            mode="semantic",
            topic_id=topic_id,
            limit=limit,
            model=model,
            fts_candidates=fts_candidates,
        )

    message_ids = [r["message_id"] for r in fts]
    best = _semantic_best_by_message(
        db, query=query, model=model, topic_id=topic_id, message_ids=message_ids
    )

    merged: list[dict[str, Any]] = []
    for r in fts:
        mid = r["message_id"]
        best_row = best.get(mid)
        if best_row is not None:
            semantic_score = float(best_row["semantic_score"])
            snippet = _semantic_snippet(
                db,
                message_id=mid,
                start_char=int(best_row["start_char"]),
                end_char=int(best_row["end_char"]),
            )
        else:
            semantic_score = None
            snippet = r["snippet"]

        merged.append(
            {
                "topic_id": r["topic_id"],
                "topic_name": r["topic_name"],
                "message_id": mid,
                "seq": r["seq"],
                "sender": r["sender"],
                "message_type": r["message_type"],
                "created_at": r["created_at"],
                "fts_rank": r["rank"],
                "semantic_score": semantic_score,
                "snippet": snippet,
            }
        )

    def _sort_key(x: dict[str, Any]) -> tuple[float, float]:
        s = x.get("semantic_score")
        semantic = -float(s) if s is not None else 1e9
        return (semantic, float(x.get("fts_rank", 0.0)))

    merged.sort(key=_sort_key)
    top = merged[:limit]

    if best and any(r.get("semantic_score") is None for r in top):
        warnings.append("hybrid_partial_semantic: some results not embedded (run embeddings index)")

    if not best:
        warnings.append(
            "hybrid_no_embeddings: run `agent-bus cli embeddings index` for semantic reranking"
        )

    return top, warnings
