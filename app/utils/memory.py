import os
import redis
import json
import uuid


def get_redis():
    """Get Redis client instance."""
    return redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))


def get_history(session_id: str) -> list:
    """Get conversation history for a session."""
    r = get_redis()
    key = f"session:{session_id}"
    data = r.get(key)
    if data is None:
        return []
    return json.loads(data)


def save_history(session_id: str, history: list):
    """Save conversation history for a session (last 6 items = 3 turns, 1 hour TTL)."""
    r = get_redis()
    key = f"session:{session_id}"
    # Keep only last 6 items (3 turns: user+assistant pairs)
    trimmed_history = history[-6:] if len(history) > 6 else history
    r.setex(key, 3600, json.dumps(trimmed_history))
