"""Tests for the Claude LLM client."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meeting_assistant.config import Settings
from meeting_assistant.core.llm_client import ClaudeClient, _split_at_boundary
from meeting_assistant.models.schemas import MeetingContext


class TestSentenceSplitting:
    """Tests for the sentence boundary splitter."""

    def test_single_sentence(self) -> None:
        parts = _split_at_boundary("Hello world.")
        assert len(parts) >= 1
        assert "Hello world." in "".join(parts)

    def test_multiple_sentences(self) -> None:
        text = "First sentence. Second sentence! Third sentence?"
        parts = _split_at_boundary(text)
        assert len(parts) >= 2

    def test_no_sentence_end(self) -> None:
        """Text without sentence boundaries returns the full text."""
        text = "This is an incomplete"
        parts = _split_at_boundary(text)
        assert "".join(parts) == text

    def test_empty_string(self) -> None:
        parts = _split_at_boundary("")
        assert parts == [""]


class TestClaudeClient:
    """Tests for the ClaudeClient with mocked Anthropic SDK."""

    @pytest.fixture
    def client(self, test_settings: Settings) -> ClaudeClient:
        return ClaudeClient(test_settings)

    @pytest.fixture
    def mock_stream(self):
        """Mock async streaming context manager."""
        class MockStream:
            def __init__(self, text_chunks):
                self._chunks = text_chunks

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            text_stream = property(lambda self: self._text_stream())

            async def _text_stream(self):
                for chunk in self._chunks:
                    yield chunk

        return MockStream

    @pytest.mark.asyncio
    async def test_answer_streaming_yields_text(
        self, client: ClaudeClient, mock_stream
    ) -> None:
        """answer_streaming should yield text chunks from the API."""
        text_chunks = ["Hello, ", "I am ", "your assistant."]

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_context)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        async def fake_text_stream():
            for chunk in text_chunks:
                yield chunk

        mock_context.text_stream = fake_text_stream()

        with patch.object(client._client.messages, "stream", return_value=mock_context):
            chunks = []
            async for chunk in client.answer_streaming("Test question", "Test context"):
                chunks.append(chunk)

        assert "".join(chunks) == "Hello, I am your assistant."

    @pytest.mark.asyncio
    async def test_answer_returns_full_text(
        self, client: ClaudeClient
    ) -> None:
        """answer() should collect all streaming chunks and return full text."""
        full_text = "This is a complete answer."

        async def fake_stream(*args, **kwargs):
            for char in full_text:
                yield char

        with patch.object(client, "answer_streaming", return_value=fake_stream()):
            result = await client.answer("Question", "Context")

        assert result == full_text

    @pytest.mark.asyncio
    async def test_sentence_callback_invoked(
        self, client: ClaudeClient
    ) -> None:
        """sentence_callback should be called with complete sentences."""
        text_chunks = ["Hello world.", " How are you?"]
        sentences_received = []

        async def callback(sentence: str) -> None:
            sentences_received.append(sentence)

        mock_context = MagicMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_context)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        async def fake_text_stream():
            for chunk in text_chunks:
                yield chunk

        mock_context.text_stream = fake_text_stream()

        with patch.object(client._client.messages, "stream", return_value=mock_context):
            async for _ in client.answer_streaming(
                "Test", "Context", sentence_callback=callback
            ):
                pass

        # At least one sentence should have been called back
        assert len(sentences_received) >= 1

    def test_build_messages_without_history(self, client: ClaudeClient) -> None:
        """_build_messages with no history should produce single user message."""
        messages = client._build_messages("What is the answer?", None)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "What is the answer?"

    def test_build_messages_with_history(self, client: ClaudeClient) -> None:
        """_build_messages with history should prepend history."""
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        messages = client._build_messages("New question", history)
        assert len(messages) == 3
        assert messages[-1]["content"] == "New question"

    def test_meeting_context_header_formatting(
        self, client: ClaudeClient, meeting_context: MeetingContext
    ) -> None:
        """_meeting_context_header should produce a readable header string."""
        header = client._meeting_context_header(meeting_context)
        assert "zoom" in header.lower()
        assert "Weekly Standup" in header
        assert "3" in header  # participant count


class TestClaudeClientIntegration:
    """Integration tests requiring a real API key.

    These tests are skipped unless ANTHROPIC_API_KEY is set in the environment.
    """

    @pytest.fixture
    def live_client(self) -> ClaudeClient | None:
        import os
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set; skipping live API test")
        live_settings = Settings(anthropic_api_key=api_key)
        return ClaudeClient(live_settings)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_live_answer(self, live_client: ClaudeClient) -> None:
        """Verify live Claude API connection returns a non-empty response."""
        response = await live_client.answer(
            "In exactly 5 words, say 'Meeting assistant is working correctly'.",
            context_str="",
        )
        assert len(response) > 0
        assert isinstance(response, str)
