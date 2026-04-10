from __future__ import annotations

import threading
import time
from collections.abc import Callable
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


def leader_ttl_seconds() -> int:
    return env_int("AGENT_BUS_EMBEDDINGS_LEADER_TTL_SECONDS", default=30, min_value=1)


def leader_heartbeat_seconds() -> int:
    return env_int("AGENT_BUS_EMBEDDINGS_LEADER_HEARTBEAT_SECONDS", default=10, min_value=1)


EmbedFn = Callable[[list[str], str], list[list[float]]]
ProgressFn = Callable[[int, int, int, int], None]
HeartbeatFn = Callable[[], None]
EMBEDDING_BACKEND_SIGNATURE = "fastembed-v1"


def embedding_content_hash(content: str) -> str:
    return sha256_hex(f"{EMBEDDING_BACKEND_SIGNATURE}\0{content}")


def _index_message(
    *,
    db: AgentBusDB,
    embed_fn: EmbedFn,
    message_id: str,
    topic_id: str,
    content: str,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    import numpy as np

    content_hash = embedding_content_hash(content)
    state = db.get_embedding_state(message_id=message_id, model=model)
    if (
        state is not None
        and state["content_hash"] == content_hash
        and int(state["chunk_size"]) == chunk_size
        and int(state["chunk_overlap"]) == chunk_overlap
    ):
        return "skipped"

    chunks = chunk_text(content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return "skipped"

    texts = [c.text for c in chunks]
    embs = embed_fn(texts, model)
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
    return "indexed"


def index_message_rows(
    *,
    db: AgentBusDB,
    rows: list[dict[str, Any]],
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_fn: EmbedFn | None = None,
    progress: ProgressFn | None = None,
    progress_every: int = 10,
    heartbeat: HeartbeatFn | None = None,
    heartbeat_every_seconds: float = 10.0,
) -> dict[str, int]:
    from agent_bus import _core  # ty: ignore[unresolved-import]

    if embed_fn is None:

        def embed_fn(texts: list[str], selected_model: str) -> list[list[float]]:
            return _core.embed_texts(texts, model=selected_model)

    indexed = 0
    skipped = 0
    processed = 0
    total = len(rows)
    last_heartbeat = time.monotonic()

    if progress is not None:
        progress(0, total, indexed, skipped)

    for row in rows:
        if heartbeat is not None and (time.monotonic() - last_heartbeat) >= heartbeat_every_seconds:
            heartbeat()
            last_heartbeat = time.monotonic()

        result = _index_message(
            db=db,
            embed_fn=embed_fn,
            message_id=str(row["message_id"]),
            topic_id=str(row["topic_id"]),
            content=str(row["content_markdown"]),
            model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if result == "indexed":
            indexed += 1
        else:
            skipped += 1

        processed += 1
        if progress is not None and (
            processed == total or (progress_every > 0 and processed % progress_every == 0)
        ):
            progress(processed, total, indexed, skipped)

    if heartbeat is not None and total > 0:
        heartbeat()
    return {
        "processed": processed,
        "indexed": indexed,
        "skipped": skipped,
    }


def _worker_loop(
    *,
    db: AgentBusDB,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    lock_ttl_seconds: int,
    error_retry_seconds: int,
    max_attempts: int,
    poll_seconds: float,
    leader_ttl_seconds: int,
    leader_heartbeat_seconds: int,
    stop_event: threading.Event,
) -> None:
    from agent_bus import _core  # ty: ignore[unresolved-import]

    def embed_fn(texts: list[str], selected_model: str) -> list[list[float]]:
        return _core.embed_texts(texts, model=selected_model)

    next_heartbeat_at = 0.0
    has_self_lease = False
    idle_sleep_seconds = poll_seconds
    idle_sleep_cap = max(poll_seconds, 1.0)

    def _sleep_with_backoff() -> None:
        nonlocal idle_sleep_seconds
        stop_event.wait(idle_sleep_seconds)
        idle_sleep_seconds = min(idle_sleep_seconds * 2, idle_sleep_cap)

    def _reset_backoff() -> None:
        nonlocal idle_sleep_seconds
        idle_sleep_seconds = poll_seconds

    def _release_self_lease() -> None:
        nonlocal has_self_lease, next_heartbeat_at
        if not has_self_lease:
            return
        with suppress(Exception):
            db.release_embedding_leader_self()
        has_self_lease = False
        next_heartbeat_at = 0.0

    def _maybe_heartbeat() -> bool:
        nonlocal next_heartbeat_at
        if time.monotonic() < next_heartbeat_at:
            return True
        if not db.renew_embedding_leader_self(ttl_seconds=leader_ttl_seconds):
            return False
        next_heartbeat_at = time.monotonic() + leader_heartbeat_seconds
        return True

    try:
        while not stop_event.is_set():
            try:
                if not db.has_ready_embedding_jobs(
                    model=model,
                    limit=batch_size,
                    lock_ttl_seconds=lock_ttl_seconds,
                    error_retry_seconds=error_retry_seconds,
                    max_attempts=max_attempts,
                ):
                    # Only release after we have actually held the self-lease.
                    _release_self_lease()
                    _sleep_with_backoff()
                    continue

                jobs = db.claim_embedding_jobs_if_leader(
                    model=model,
                    limit=batch_size,
                    lock_ttl_seconds=lock_ttl_seconds,
                    error_retry_seconds=error_retry_seconds,
                    max_attempts=max_attempts,
                    leader_ttl_seconds=leader_ttl_seconds,
                    leader_heartbeat_seconds=leader_heartbeat_seconds,
                )
            except (SchemaMismatchError, ValueError):
                return
            except DBBusyError:
                _sleep_with_backoff()
                continue
            except Exception:  # pragma: no cover
                _sleep_with_backoff()
                continue

            if not jobs:
                # The Rust helper already handled any required cleanup for the empty batch.
                has_self_lease = False
                next_heartbeat_at = 0.0
                _sleep_with_backoff()
                continue

            has_self_lease = True
            _reset_backoff()
            next_heartbeat_at = time.monotonic() + leader_heartbeat_seconds

            for job in jobs:
                if stop_event.is_set():
                    return
                if not _maybe_heartbeat():
                    _release_self_lease()
                    _sleep_with_backoff()
                    break

                message_id = str(job["message_id"])
                topic_id = str(job["topic_id"])

                try:
                    msg = db.get_message_by_id(message_id=message_id)
                except (ValueError, DBBusyError):
                    # Message removed or DB temporarily unavailable; drop the job and move on.
                    with suppress(DBBusyError):
                        db.complete_embedding_job(message_id=message_id, model=model)
                    continue

                try:
                    _index_message(
                        db=db,
                        embed_fn=embed_fn,
                        message_id=message_id,
                        topic_id=topic_id,
                        content=msg.content_markdown,
                        model=model,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
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
    finally:
        _release_self_lease()


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
        leader_ttl = leader_ttl_seconds()
        heartbeat_seconds = leader_heartbeat_seconds()
        if heartbeat_seconds >= leader_ttl:
            heartbeat_seconds = max(1, leader_ttl // 2)
        error_retry_seconds = env_int(
            "AGENT_BUS_EMBEDDINGS_ERROR_RETRY_SECONDS", default=30, min_value=0
        )
        max_attempts = env_int("AGENT_BUS_EMBEDDINGS_MAX_ATTEMPTS", default=5, min_value=1)

        poll_ms = env_int("AGENT_BUS_EMBEDDINGS_POLL_MS", default=250, min_value=10)
        poll_seconds = poll_ms / 1000.0

        stop_event = threading.Event()
        t = threading.Thread(
            target=_worker_loop,
            kwargs={
                "db": db,
                "model": model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "batch_size": batch_size,
                "lock_ttl_seconds": lock_ttl_seconds,
                "error_retry_seconds": error_retry_seconds,
                "max_attempts": max_attempts,
                "poll_seconds": poll_seconds,
                "leader_ttl_seconds": leader_ttl,
                "leader_heartbeat_seconds": heartbeat_seconds,
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
