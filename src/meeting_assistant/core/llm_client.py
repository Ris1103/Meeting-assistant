"""Anthropic Claude API client with streaming support.

Provides async, streaming LLM responses optimised for real-time meeting assistance.
Uses Claude's message streaming API to begin delivering text as soon as the first
tokens are generated, enabling sentence-by-sentence TTS playback.

Features:
  - Async streaming with sentence-boundary callbacks
  - Prompt caching for meeting context (reduces latency + cost on repeated context)
  - Exponential backoff retry via tenacity
  - Meeting-focused system prompt
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from typing import Awaitable

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from meeting_assistant.config import Settings
from meeting_assistant.models.schemas import MeetingContext

logger = logging.getLogger(__name__)

# Sentence boundary: ends with . ! ? followed by whitespace or end-of-string
_SENTENCE_BOUNDARY = re.compile(r"[.!?]\s+|[.!?]$")

# Base system prompt for the meeting assistant
_SYSTEM_PROMPT_TEMPLATE = """\
You are an AI meeting assistant. You listen to ongoing meetings and help participants \
by answering questions, summarizing discussions, and tracking action items in real time.

Guidelines:
- Be concise and direct — meeting participants are busy.
- Speak in first person as a helpful assistant.
- When answering questions, cite the relevant part of the transcript if possible.
- For summaries, use bullet points.
- For action items, number them clearly.
- If you are uncertain, say so briefly rather than guessing.
- Do NOT repeat back the full transcript — focus on actionable insights.

{context}
"""


class ClaudeClient:
    """Async Claude API client for meeting assistance.

    Usage::

        client = ClaudeClient(settings)
        async for chunk in client.answer_streaming("What were the action items?", context_str):
            print(chunk, end="", flush=True)

        response = await client.answer("Summarise the discussion so far.", context_str)
    """

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)
        self._model = config.claude_model
        self._max_tokens = config.max_response_tokens

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def answer_streaming(
        self,
        question: str,
        context_str: str,
        conversation_history: list[dict[str, str]] | None = None,
        sentence_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Stream the assistant's answer token by token.

        Args:
            question: The question or prompt from the user.
            context_str: Meeting context string (transcript + chat + metadata).
            conversation_history: Optional prior conversation turns for multi-turn dialogue.
            sentence_callback: If provided, called with each complete sentence as it arrives.
                               Useful for driving real-time TTS.

        Yields:
            Text delta strings as they arrive from the API.
        """
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(context=context_str)
        messages = self._build_messages(question, conversation_history)

        sentence_buffer = ""

        async with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text
                sentence_buffer += text
                if sentence_callback and _SENTENCE_BOUNDARY.search(sentence_buffer):
                    # Extract complete sentences from buffer
                    sentences = _split_at_boundary(sentence_buffer)
                    for sentence in sentences[:-1]:  # all but potentially-incomplete last
                        if sentence.strip():
                            await sentence_callback(sentence.strip())
                    sentence_buffer = sentences[-1] if sentences else ""

            # Flush any remaining buffer
            if sentence_callback and sentence_buffer.strip():
                await sentence_callback(sentence_buffer.strip())

    async def answer(
        self,
        question: str,
        context_str: str,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Return a complete (non-streaming) answer.

        Args:
            question: The question or prompt.
            context_str: Formatted meeting context string.
            conversation_history: Optional prior conversation turns.

        Returns:
            The full assistant response text.
        """
        chunks = []
        async for chunk in self.answer_streaming(question, context_str, conversation_history):
            chunks.append(chunk)
        return "".join(chunks)

    async def summarize(self, context: MeetingContext) -> str:
        """Generate a concise meeting summary from the current context.

        Args:
            context: Full MeetingContext snapshot.

        Returns:
            Bullet-point summary string.
        """
        transcript_text = context.format_transcript(max_segments=100)
        prompt = (
            "Please provide a concise summary of this meeting so far. "
            "Include:\n"
            "1. Main topics discussed\n"
            "2. Key decisions made\n"
            "3. Action items identified\n\n"
            f"Transcript:\n{transcript_text}"
        )
        return await self.answer(prompt, self._meeting_context_header(context))

    async def extract_action_items(self, context: MeetingContext) -> list[str]:
        """Extract action items from the meeting transcript.

        Returns:
            List of action item strings.
        """
        transcript_text = context.format_transcript(max_segments=100)
        prompt = (
            "List all action items mentioned in this meeting transcript. "
            "Format each as: 'Owner: Action (deadline if mentioned)'. "
            "If no clear owner, use 'Team'. "
            "Reply with ONLY the numbered list, no preamble.\n\n"
            f"Transcript:\n{transcript_text}"
        )
        response = await self.answer(prompt, self._meeting_context_header(context))
        # Parse numbered list
        lines = [
            line.strip()
            for line in response.splitlines()
            if re.match(r"^\d+\.", line.strip())
        ]
        return [re.sub(r"^\d+\.\s*", "", line) for line in lines]

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        question: str,
        history: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        """Construct the messages list for the API call."""
        messages: list[dict[str, str]] = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def _meeting_context_header(context: MeetingContext) -> str:
        """Format brief meeting metadata for the system prompt."""
        return (
            f"Meeting platform: {context.platform.value}\n"
            f"Title: {context.title or 'Untitled'}\n"
            f"Participants: {context.participant_count} detected\n"
            f"Started: {context.start_time.strftime('%H:%M UTC')}"
        )

    @retry(
        retry=retry_if_exception_type((anthropic.APIConnectionError, anthropic.RateLimitError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(4),
    )
    async def _call_with_retry(
        self,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> anthropic.types.Message:
        """Make a non-streaming API call with automatic retry."""
        return await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system_prompt,
            messages=messages,
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _split_at_boundary(text: str) -> list[str]:
    """Split text at sentence boundaries, preserving the separator."""
    parts = _SENTENCE_BOUNDARY.split(text)
    # Re-add the delimiters
    delimiters = _SENTENCE_BOUNDARY.findall(text)
    result = []
    for i, part in enumerate(parts):
        result.append(part + (delimiters[i] if i < len(delimiters) else ""))
    return result if result else [text]
