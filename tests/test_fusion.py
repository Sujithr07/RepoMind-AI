"""Unit tests for Multi-Query RAG Fusion query expansion.

These avoid hitting Groq by mocking the module-level client. They exercise the
dedupe/ordering contract and the graceful-degradation fallback.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.query.fusion import generate_fusion_queries


def _groq_response(payload: dict) -> MagicMock:
    """Build a fake Groq chat-completion response wrapping a JSON string."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(payload)
    return resp


@pytest.mark.asyncio
async def test_generate_fusion_queries_prepends_original_and_keeps_order():
    payload = {"queries": ["how does login work", "auth flow", "session handling"]}
    with patch("app.query.fusion.client") as mock_client:
        mock_client.chat.completions.create.return_value = _groq_response(payload)
        queries = await generate_fusion_queries("user authentication")

    # Original question is first, reformulations follow in order.
    assert queries == [
        "user authentication",
        "how does login work",
        "auth flow",
        "session handling",
    ]


@pytest.mark.asyncio
async def test_generate_fusion_queries_dedupes_case_insensitively():
    payload = {"queries": ["User Authentication", "auth flow", "AUTH FLOW"]}
    with patch("app.query.fusion.client") as mock_client:
        mock_client.chat.completions.create.return_value = _groq_response(payload)
        queries = await generate_fusion_queries("user authentication")

    # The reformulation that duplicates the original (modulo case) and the
    # repeated "auth flow" are both dropped.
    assert queries == ["user authentication", "auth flow"]


@pytest.mark.asyncio
async def test_generate_fusion_queries_falls_back_to_single_query_on_error():
    with patch("app.query.fusion.client") as mock_client:
        mock_client.chat.completions.create.side_effect = Exception("groq down")
        queries = await generate_fusion_queries("user authentication")

    # A flaky expansion must never break search — degrade to the raw question.
    assert queries == ["user authentication"]


@pytest.mark.asyncio
async def test_generate_fusion_queries_ignores_non_string_and_blank_entries():
    payload = {"queries": ["valid query", "", "   ", 42, None]}
    with patch("app.query.fusion.client") as mock_client:
        mock_client.chat.completions.create.return_value = _groq_response(payload)
        queries = await generate_fusion_queries("original")

    assert queries == ["original", "valid query"]
