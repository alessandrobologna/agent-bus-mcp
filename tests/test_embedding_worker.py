from __future__ import annotations

import pytest

from agent_bus import _core
from agent_bus.embedding_worker import _worker_loop


class _IdleWorkerDB:
    def __init__(self) -> None:
        self.owner: str | None = None
        self.idle_release_seen = False

    def claim_embedding_leader(self, worker_id: str, ttl_seconds: int) -> bool:
        assert ttl_seconds > 0
        if self.owner in (None, worker_id):
            self.owner = worker_id
            return True
        return False

    def release_embedding_leader(self, worker_id: str) -> bool:
        if self.owner == worker_id:
            self.owner = None
            return True
        return False

    def claim_embedding_jobs(
        self,
        *,
        model: str,
        limit: int,
        worker_id: str,
        lock_ttl_seconds: int,
        error_retry_seconds: int,
        max_attempts: int,
    ) -> list[dict[str, object]]:
        assert model == "unit-test-model"
        assert limit == 1
        assert worker_id == "worker-1"
        assert lock_ttl_seconds > 0
        assert error_retry_seconds >= 0
        assert max_attempts > 0
        return []


class _StopAfterIdleWait:
    def __init__(self, db: _IdleWorkerDB) -> None:
        self._db = db
        self._set = False

    def is_set(self) -> bool:
        return self._set

    def wait(self, timeout: float) -> bool:
        assert timeout >= 0
        self._db.idle_release_seen = self._db.owner is None
        self._set = True
        return True


def test_idle_worker_releases_leader_before_sleeping() -> None:
    db = _IdleWorkerDB()
    stop_event = _StopAfterIdleWait(db)

    _worker_loop(
        db=db,
        worker_id="worker-1",
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

    assert db.idle_release_seen is True


def test_embedding_core_rejects_e5_small_v2_alias() -> None:
    with pytest.raises(ValueError, match="intfloat/multilingual-e5-small"):
        _core.embed_texts(["hello"], model="intfloat/e5-small-v2")
