from __future__ import annotations

import threading
import uuid
from contextlib import suppress
from typing import Any

from agent_bus.common import env_int, env_str
from agent_bus.db import AgentBusDB, DBBusyError, SchemaMismatchError
from agent_bus.search import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    chunk_text,
    sha256_hex,
)


def embedding_model() -> str:
    return env_str("AGENT_BUS_EMBEDDING_MODEL", default=DEFAULT_EMBEDDING_MODEL)


def embedding_chunk_size() -> int:
    return env_int("AGENT_BUS_EMBEDDING_CHUNK_SIZE", default=DEFAULT_CHUNK_SIZE, min_value=1)


def embedding_chunk_overlap() -> int:
    return env_int("AGENT_BUS_EMBEDDING_CHUNK_OVERLAP", default=DEFAULT_CHUNK_OVERLAP, min_value=0)


def autoindex_enabled() -> bool:
    return env_int("AGENT_BUS_EMBEDDINGS_AUTOINDEX", default=1, min_value=0) != 0


def _worker_loop(
    *,
    db: AgentBusDB,
    worker_id: str,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    lock_ttl_seconds: int,
    error_retry_seconds: int,
    max_attempts: int,
    poll_seconds: float,
    stop_event: threading.Event,
) -> None:
    import numpy as np

    from agent_bus import _core  # ty: ignore[unresolved-import]

    while not stop_event.is_set():
        try:
            jobs = db.claim_embedding_jobs(
                model=model,
                limit=batch_size,
                worker_id=worker_id,
                lock_ttl_seconds=lock_ttl_seconds,
                error_retry_seconds=error_retry_seconds,
                max_attempts=max_attempts,
            )
        except (SchemaMismatchError, ValueError):
            return
        except DBBusyError:
            stop_event.wait(poll_seconds)
            continue
        except Exception:  # pragma: no cover
            stop_event.wait(poll_seconds)
            continue

        if not jobs:
            stop_event.wait(poll_seconds)
            continue

        for job in jobs:
            if stop_event.is_set():
                return
            message_id = str(job["message_id"])
            topic_id = str(job["topic_id"])

            try:
                msg = db.get_message_by_id(message_id=message_id)
            except (ValueError, DBBusyError):
                # Message removed or DB temporarily unavailable; drop the job and move on.
                with suppress(DBBusyError):
                    db.complete_embedding_job(message_id=message_id, model=model)
                continue

            content = msg.content_markdown
            content_hash = sha256_hex(content)

            try:
                state = db.get_embedding_state(message_id=message_id, model=model)
                if (
                    state is not None
                    and state["content_hash"] == content_hash
                    and int(state["chunk_size"]) == chunk_size
                    and int(state["chunk_overlap"]) == chunk_overlap
                ):
                    db.complete_embedding_job(message_id=message_id, model=model)
                    continue

                chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if not chunks:
                    db.complete_embedding_job(message_id=message_id, model=model)
                    continue

                texts = [c.text for c in chunks]
                embs = _core.embed_texts(texts, model=model)
                arr = np.asarray(embs, dtype=np.float32)
                dims = int(arr.shape[1])

                payload: list[dict[str, Any]] = []
                for c, vec in zip(chunks, arr, strict=True):
                    payload.append(
                        {
                            "chunk_index": c.chunk_index,
                            "start_char": c.start_char,
                            "end_char": c.end_char,
                            "text_hash": sha256_hex(c.text),
                            "vector": vec.tobytes(),
                        }
                    )

                db.upsert_embeddings(
                    message_id=message_id,
                    model=model,
                    topic_id=topic_id,
                    content_hash=content_hash,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    dims=dims,
                    chunks=payload,
                )
                db.complete_embedding_job(message_id=message_id, model=model)
            except DBBusyError:
                # Leave the job claimed; it can be reclaimed after TTL if needed.
                stop_event.wait(poll_seconds)
                continue
            except Exception as e:  # pragma: no cover
                with suppress(DBBusyError):
                    db.fail_embedding_job(message_id=message_id, model=model, error=str(e))

        # Avoid a tight loop when there is always work (and allow other DB users to make progress).
        stop_event.wait(0.01)


_worker_lock = threading.Lock()
_worker_thread: threading.Thread | None = None
_stop_event: threading.Event | None = None


def start_background_embedding_worker(db: AgentBusDB) -> None:
    global _worker_thread, _stop_event
    if not autoindex_enabled():
        return

    with _worker_lock:
        if _worker_thread is not None and _worker_thread.is_alive():
            return

        model = embedding_model()
        chunk_size = embedding_chunk_size()
        chunk_overlap = embedding_chunk_overlap()
        if chunk_overlap >= chunk_size:
            return

        batch_size = env_int("AGENT_BUS_EMBEDDINGS_WORKER_BATCH_SIZE", default=5, min_value=1)
        lock_ttl_seconds = env_int(
            "AGENT_BUS_EMBEDDINGS_LOCK_TTL_SECONDS", default=300, min_value=1
        )
        error_retry_seconds = env_int(
            "AGENT_BUS_EMBEDDINGS_ERROR_RETRY_SECONDS", default=30, min_value=0
        )
        max_attempts = env_int("AGENT_BUS_EMBEDDINGS_MAX_ATTEMPTS", default=5, min_value=1)

        poll_ms = env_int("AGENT_BUS_EMBEDDINGS_POLL_MS", default=250, min_value=10)
        poll_seconds = poll_ms / 1000.0

        stop_event = threading.Event()
        worker_id = uuid.uuid4().hex[:8]
        t = threading.Thread(
            target=_worker_loop,
            kwargs={
                "db": db,
                "worker_id": worker_id,
                "model": model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "batch_size": batch_size,
                "lock_ttl_seconds": lock_ttl_seconds,
                "error_retry_seconds": error_retry_seconds,
                "max_attempts": max_attempts,
                "poll_seconds": poll_seconds,
                "stop_event": stop_event,
            },
            name="agent-bus-embeddings-worker",
            daemon=True,
        )
        t.start()
        _worker_thread = t
        _stop_event = stop_event


def stop_background_embedding_worker() -> None:  # pragma: no cover
    global _stop_event
    if _stop_event is not None:
        _stop_event.set()
