"""Real-time ingestion progress events over Redis pub/sub.

The Celery worker publishes JSON progress events to a per-repo channel
(``progress:{repo_id}``) as it moves through the indexing pipeline. The FastAPI
``/repos/{id}/progress`` WebSocket subscribes to that channel and streams events
to the browser.

Each event also gets stashed at ``progress:last:{repo_id}`` (short TTL) so a
client connecting mid-run immediately receives the latest snapshot — pub/sub has
no backlog of its own.
"""

import json
import time

CHANNEL_PREFIX = "progress"
SNAPSHOT_PREFIX = "progress:last"
SNAPSHOT_TTL = 3600  # seconds — long enough to cover a reconnect, not forever

# Each phase maps to a slice of the overall 0-100% bar so a single "percent"
# value advances monotonically across the whole pipeline. Embedding is the
# long pole, so it owns the widest band.
_PHASE_BANDS = {
    "cloning":   (0, 10),
    "diffing":   (10, 15),
    "chunking":  (15, 35),
    "embedding": (35, 80),
    "upserting": (80, 98),
    "done":      (100, 100),
    "error":     (0, 0),
}

TERMINAL_PHASES = {"done", "error"}


def channel_for(repo_id) -> str:
    return f"{CHANNEL_PREFIX}:{repo_id}"


def snapshot_key(repo_id) -> str:
    return f"{SNAPSHOT_PREFIX}:{repo_id}"


def is_terminal(raw: str) -> bool:
    """True if a serialized event marks the end of the run (done/error)."""
    try:
        return json.loads(raw).get("phase") in TERMINAL_PHASES
    except Exception:
        return False


class ProgressPublisher:
    """Publishes pipeline progress for one repo to Redis pub/sub."""

    def __init__(self, repo_id, redis_client):
        self.repo_id = str(repo_id)
        self.redis = redis_client

    def _percent(self, phase: str, current: int, total: int) -> int:
        start, end = _PHASE_BANDS.get(phase, (0, 0))
        if phase in TERMINAL_PHASES:
            return start
        frac = (current / total) if total else 0.0
        frac = min(max(frac, 0.0), 1.0)
        return round(start + (end - start) * frac)

    async def publish(
        self,
        phase: str,
        message: str = "",
        current: int = 0,
        total: int = 0,
    ) -> dict:
        """Emit a progress event. Best-effort: never raises into the pipeline."""
        event = {
            "repo_id": self.repo_id,
            "phase": phase,
            "message": message,
            "current": current,
            "total": total,
            "percent": self._percent(phase, current, total),
            "ts": time.time(),
        }
        payload = json.dumps(event)
        try:
            await self.redis.publish(channel_for(self.repo_id), payload)
            await self.redis.set(snapshot_key(self.repo_id), payload, ex=SNAPSHOT_TTL)
        except Exception as e:
            print(f"[progress] publish failed for {self.repo_id}: {e}")
        return event
