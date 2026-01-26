from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal, cast

from agent_bus import _core  # ty: ignore[unresolved-import]
from agent_bus.db import AgentBusDB, DBBusyError

SearchMode = Literal["fts", "semantic", "hybrid"]

# Default embedding model (downloaded on first use).
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
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


def _embed_query(model: str, query: str):
    import numpy as np

    emb = _core.embed_texts([query], model=model)[0]
    return np.asarray(emb, dtype=np.float32)


def _cosine_scores(query_vec, candidate_rows: list[dict[str, Any]]):
    import numpy as np

    if not candidate_rows:
        return np.array([], dtype=np.float32)

    dims = int(candidate_rows[0]["dims"])
    vectors = [np.frombuffer(c["vector"], dtype=np.float32, count=dims) for c in candidate_rows]
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
    candidates = db.list_chunk_embedding_candidates(
        model=model, topic_id=topic_id, message_ids=message_ids
    )
    if not candidates:
        return {}

    query_vec = _embed_query(model, query)
    scores = _cosine_scores(query_vec, candidates)

    best: dict[str, dict[str, Any]] = {}
    for cand, score in zip(candidates, scores, strict=True):
        mid = cand["message_id"]
        s = float(score)
        existing = best.get(mid)
        if existing is None or s > float(existing["semantic_score"]):
            best[mid] = {**cand, "semantic_score": s}
    return best


def _get_message_content(
    db: AgentBusDB, *, message_id: str, cache: dict[str, str]
) -> str:  # pragma: no cover (simple cache)
    raw = cache.get(message_id)
    if raw is None:
        raw = db.get_message_by_id(message_id=message_id).content_markdown
        cache[message_id] = raw
    return raw


def _semantic_snippet(
    db: AgentBusDB,
    *,
    message_id: str,
    start_char: int,
    end_char: int,
    content_cache: dict[str, str],
) -> str:
    raw = _get_message_content(db, message_id=message_id, cache=content_cache)
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
    include_content: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if fts_candidates <= 0:
        raise ValueError("fts_candidates must be > 0")
    if not isinstance(include_content, bool):
        raise ValueError("include_content must be a bool")

    warnings: list[str] = []

    if mode == "fts":
        return (
            db.search_messages_fts(
                query=query, topic_id=topic_id, limit=limit, include_content=include_content
            ),
            warnings,
        )

    if mode not in {"semantic", "hybrid"}:
        raise ValueError("mode must be one of: fts, semantic, hybrid")

    if mode == "semantic":
        content_cache: dict[str, str] = {}
        best = _semantic_best_by_message(
            db,
            query=query,
            model=model,
            topic_id=topic_id,
            message_ids=None,
        )
        ranked = sorted(best.values(), key=lambda r: float(r["semantic_score"]), reverse=True)[
            :limit
        ]
        out: list[dict[str, Any]] = []
        for r in ranked:
            mid = cast(str, r["message_id"])
            content = _get_message_content(db, message_id=mid, cache=content_cache)
            out.append(
                {
                    "topic_id": r["topic_id"],
                    "topic_name": r["topic_name"],
                    "message_id": mid,
                    "seq": r["seq"],
                    "sender": r["sender"],
                    "message_type": r["message_type"],
                    "created_at": r["created_at"],
                    "semantic_score": r["semantic_score"],
                    "snippet": _semantic_snippet(
                        db,
                        message_id=mid,
                        start_char=int(r["start_char"]),
                        end_char=int(r["end_char"]),
                        content_cache=content_cache,
                    ),
                    **({"content_markdown": content} if include_content else {}),
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
            include_content=include_content,
        )

    message_ids = [r["message_id"] for r in fts]
    try:
        best = _semantic_best_by_message(
            db, query=query, model=model, topic_id=topic_id, message_ids=message_ids
        )
    except DBBusyError:
        raise
    except Exception as e:  # pragma: no cover (best-effort hybrid)
        warnings.append(f"hybrid_semantic_failed: {e}")
        best = {}

    merged: list[dict[str, Any]] = []
    content_cache: dict[str, str] = {}
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
                content_cache=content_cache,
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

    if include_content:
        for item in top:
            mid = item["message_id"]
            item["content_markdown"] = _get_message_content(db, message_id=mid, cache=content_cache)

    if best and any(r.get("semantic_score") is None for r in top):
        warnings.append("hybrid_partial_semantic: some results not embedded (run embeddings index)")

    if not best:
        warnings.append(
            "hybrid_no_embeddings: run `agent-bus cli embeddings index` for semantic reranking"
        )

    return top, warnings
