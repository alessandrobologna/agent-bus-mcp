from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_bus import _core
from agent_bus.embedding_worker import _worker_loop


class _WorkerDB:
    def __init__(
        self,
        *,
        ready_sequence: list[bool],
        claim_sequence: list[list[dict[str, object]]] | None = None,
        expected_limit: int = 1,
    ) -> None:
        self.ready_sequence = list(ready_sequence)
        self.claim_sequence = list(claim_sequence or [])
        self.expected_limit = expected_limit
        self.peek_calls = 0
        self.claim_calls = 0
        self.renew_calls = 0
        self.release_calls = 0
        self.completed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.upserted: list[str] = []

    def has_ready_embedding_jobs(
        self,
        *,
        model: str,
        limit: int,
        lock_ttl_seconds: int,
        error_retry_seconds: int,
        max_attempts: int,
    ) -> bool:
        assert model == "unit-test-model"
        assert limit == self.expected_limit
        assert lock_ttl_seconds > 0
        assert error_retry_seconds >= 0
        assert max_attempts > 0
        self.peek_calls += 1
        if not self.ready_sequence:
            return False
        return self.ready_sequence.pop(0)

    def claim_embedding_jobs_if_leader(
        self,
        *,
        model: str,
        limit: int,
        lock_ttl_seconds: int,
        error_retry_seconds: int,
        max_attempts: int,
        leader_ttl_seconds: int,
        leader_heartbeat_seconds: int,
    ) -> list[dict[str, object]]:
        assert model == "unit-test-model"
        assert limit == self.expected_limit
        assert lock_ttl_seconds > 0
        assert error_retry_seconds >= 0
        assert max_attempts > 0
        assert leader_ttl_seconds > 0
        assert leader_heartbeat_seconds > 0
        self.claim_calls += 1
        if not self.claim_sequence:
            return []
        return self.claim_sequence.pop(0)

    def renew_embedding_leader_self(self, *, ttl_seconds: int) -> bool:
        assert ttl_seconds > 0
        self.renew_calls += 1
        return True

    def release_embedding_leader_self(self) -> bool:
        self.release_calls += 1
        return True

    def get_message_by_id(self, *, message_id: str) -> SimpleNamespace:
        return SimpleNamespace(content_markdown=f"content {message_id}")

    def get_embedding_state(self, *, message_id: str, model: str) -> None:
        assert model == "unit-test-model"
        return None

    def upsert_embeddings(
        self,
        *,
        message_id: str,
        model: str,
        topic_id: str,
        content_hash: str,
        chunk_size: int,
        chunk_overlap: int,
        dims: int,
        chunks: list[dict[str, object]],
    ) -> None:
        self.upserted.append(message_id)
        assert model == "unit-test-model"
        assert topic_id == "topic-1"
        assert content_hash
        assert chunk_size > 0
        assert chunk_overlap >= 0
        assert dims >= 0
        assert isinstance(chunks, list)

    def complete_embedding_job(self, *, message_id: str, model: str) -> None:
        assert model == "unit-test-model"
        self.completed.append(message_id)

    def fail_embedding_job(self, *, message_id: str, model: str, error: str) -> None:
        assert model == "unit-test-model"
        self.failed.append((message_id, error))


class _StopAfterWaits:
    def __init__(self, stop_after: int) -> None:
        assert stop_after > 0
        self._stop_after = stop_after
        self._set = False
        self.waits: list[float] = []

    def is_set(self) -> bool:
        return self._set

    def wait(self, timeout: float) -> bool:
        assert timeout >= 0
        self.waits.append(timeout)
        if len(self.waits) >= self._stop_after:
            self._set = True
        return True


class _Monotonic:
    def __init__(self, values: list[float]) -> None:
        self._values = list(values)

    def __call__(self) -> float:
        if self._values:
            return self._values.pop(0)
        return 10.0


def test_idle_worker_peeks_read_only_without_releasing_when_it_has_no_lease() -> None:
    db = _WorkerDB(ready_sequence=[False])
    stop_event = _StopAfterWaits(stop_after=1)

    _worker_loop(
        db=db,
        model="unit-test-model",
        chunk_size=100,
        chunk_overlap=0,
        batch_size=1,
        lock_ttl_seconds=30,
        error_retry_seconds=0,
        max_attempts=3,
        poll_seconds=0.01,
        leader_ttl_seconds=30,
        leader_heartbeat_seconds=10,
        stop_event=stop_event,  # type: ignore[arg-type]
    )

    assert db.peek_calls == 1
    assert db.claim_calls == 0
    assert db.renew_calls == 0
    assert db.release_calls == 0


def test_worker_renews_self_lease_while_processing_and_releases_when_drained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _WorkerDB(
        ready_sequence=[True, False],
        claim_sequence=[
            [
                {"message_id": "msg-1", "topic_id": "topic-1", "attempts": 1},
                {"message_id": "msg-2", "topic_id": "topic-1", "attempts": 1},
            ]
        ],
        expected_limit=2,
    )
    stop_event = _StopAfterWaits(stop_after=2)
    monkeypatch.setattr(
        "agent_bus.embedding_worker.time.monotonic",
        _Monotonic([0.0, 0.1, 1.1, 1.2, 1.3]),
    )
    monkeypatch.setattr(
        "agent_bus.embedding_worker._index_message",
        lambda **_: "indexed",
    )

    _worker_loop(
        db=db,
        model="unit-test-model",
        chunk_size=100,
        chunk_overlap=0,
        batch_size=2,
        lock_ttl_seconds=30,
        error_retry_seconds=0,
        max_attempts=3,
        poll_seconds=0.01,
        leader_ttl_seconds=30,
        leader_heartbeat_seconds=1,
        stop_event=stop_event,  # type: ignore[arg-type]
    )

    assert db.peek_calls == 2
    assert db.claim_calls == 1
    assert db.renew_calls >= 1
    assert db.release_calls == 1
    assert db.completed == ["msg-1", "msg-2"]
    assert db.failed == []


def test_worker_idle_backoff_never_drops_below_configured_poll_interval() -> None:
    db = _WorkerDB(ready_sequence=[False, False])
    stop_event = _StopAfterWaits(stop_after=2)

    _worker_loop(
        db=db,
        model="unit-test-model",
        chunk_size=100,
        chunk_overlap=0,
        batch_size=1,
        lock_ttl_seconds=30,
        error_retry_seconds=0,
        max_attempts=3,
        poll_seconds=5.0,
        leader_ttl_seconds=30,
        leader_heartbeat_seconds=10,
        stop_event=stop_event,  # type: ignore[arg-type]
    )

    assert stop_event.waits == [5.0, 5.0]


def test_worker_does_not_release_again_when_claim_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _WorkerDB(
        ready_sequence=[True, False],
        claim_sequence=[[]],
    )
    stop_event = _StopAfterWaits(stop_after=2)
    monkeypatch.setattr(
        "agent_bus.embedding_worker.time.monotonic",
        _Monotonic([0.0, 0.1, 1.1]),
    )

    _worker_loop(
        db=db,
        model="unit-test-model",
        chunk_size=100,
        chunk_overlap=0,
        batch_size=1,
        lock_ttl_seconds=30,
        error_retry_seconds=0,
        max_attempts=3,
        poll_seconds=0.01,
        leader_ttl_seconds=30,
        leader_heartbeat_seconds=1,
        stop_event=stop_event,  # type: ignore[arg-type]
    )

    assert db.peek_calls == 2
    assert db.claim_calls == 1
    assert db.renew_calls == 0
    assert db.release_calls == 0


def test_embedding_core_rejects_e5_small_v2_alias() -> None:
    with pytest.raises(ValueError, match="intfloat/multilingual-e5-small"):
        _core.embed_texts(["hello"], model="intfloat/e5-small-v2")
