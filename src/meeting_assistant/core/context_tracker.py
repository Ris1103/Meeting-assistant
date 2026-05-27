"""Meeting context tracker: accumulates transcript, chat, and speaker state.

The context tracker is the single source of truth for meeting state. It maintains
a rolling window of recent transcript segments and chat messages, and provides
formatted context strings for LLM consumption.

Thread-safety is guaranteed via asyncio.Lock so concurrent add_segment() and
snapshot() calls do not race.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import deque
from datetime import datetime
from typing import Any

from meeting_assistant.config import Settings
from meeting_assistant.models.schemas import (
    ChatMessage,
    MeetingContext,
    MeetingStatus,
    Platform,
    SpeakerSegment,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)

# Simple heuristic patterns for question detection
_QUESTION_PATTERNS = re.compile(
    r"(\?|"
    r"\b(what|who|when|where|why|how|can|could|would|should|is|are|was|were|do|does|did)\b"
    r".{3,})",
    re.IGNORECASE,
)

# Trigger keyword pattern (replaced at runtime with configured keyword)
_TRIGGER_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def _trigger_pattern(keyword: str) -> re.Pattern[str]:
    if keyword not in _TRIGGER_PATTERN_CACHE:
        _TRIGGER_PATTERN_CACHE[keyword] = re.compile(
            rf"\b{re.escape(keyword)}\b", re.IGNORECASE
        )
    return _TRIGGER_PATTERN_CACHE[keyword]


class MeetingContextTracker:
    """Tracks and manages meeting context state.

    Usage::

        tracker = MeetingContextTracker(settings)
        tracker.start_meeting("meeting-123", Platform.ZOOM)

        await tracker.add_segment(speaker_segment)
        await tracker.add_chat_message(chat_message)

        context = await tracker.get_context()
        prompt = tracker.build_llm_context_string(context)
    """

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._lock = asyncio.Lock()

        # Rolling deques with configurable max lengths
        self._segments: deque[SpeakerSegment] = deque(
            maxlen=config.transcript_window_size
        )
        self._chat: deque[ChatMessage] = deque(maxlen=config.chat_window_size)

        # LLM conversation history (role/content dicts for Claude)
        self._assistant_history: list[dict[str, str]] = []

        # Meeting metadata
        self._meeting_id: str = ""
        self._platform: Platform = Platform.GENERIC
        self._title: str | None = None
        self._start_time: datetime = datetime.utcnow()
        self._status: MeetingStatus = MeetingStatus.IDLE

        # Speaker name resolution: speaker_id → display name
        self._speaker_names: dict[str, str] = {}

        # Tracking
        self._last_question: str | None = None

    # -----------------------------------------------------------------------
    # Meeting lifecycle
    # -----------------------------------------------------------------------

    def start_meeting(
        self,
        meeting_id: str,
        platform: Platform,
        title: str | None = None,
    ) -> None:
        """Initialise context for a new meeting session."""
        self._meeting_id = meeting_id
        self._platform = platform
        self._title = title
        self._start_time = datetime.utcnow()
        self._status = MeetingStatus.ACTIVE
        self._segments.clear()
        self._chat.clear()
        self._assistant_history.clear()
        self._speaker_names.clear()
        self._last_question = None
        logger.info("Meeting started: id=%s platform=%s", meeting_id, platform.value)

    def end_meeting(self) -> None:
        """Mark meeting as ended."""
        self._status = MeetingStatus.ENDED
        logger.info("Meeting ended: %s", self._meeting_id)

    # -----------------------------------------------------------------------
    # Mutation methods (async for thread-safety)
    # -----------------------------------------------------------------------

    async def add_segment(self, segment: SpeakerSegment) -> None:
        """Add a transcribed speaker segment to the rolling transcript."""
        async with self._lock:
            self._segments.append(segment)
            # Check if the segment contains the trigger keyword
            if _trigger_pattern(self._config.trigger_keyword).search(segment.text):
                self._last_question = segment.text
                logger.debug("Trigger keyword detected in: %r", segment.text)

    async def add_result(self, result: TranscriptionResult) -> None:
        """Add all segments from a TranscriptionResult."""
        for segment in result.segments:
            await self.add_segment(segment)

    async def add_chat_message(self, message: ChatMessage) -> None:
        """Add an incoming chat message."""
        async with self._lock:
            self._chat.append(message)
            logger.debug("Chat message added from %s: %r", message.sender, message.content[:60])

    async def add_assistant_turn(self, user_text: str, assistant_text: str) -> None:
        """Record an LLM interaction in conversation history."""
        async with self._lock:
            self._assistant_history.append({"role": "user", "content": user_text})
            self._assistant_history.append({"role": "assistant", "content": assistant_text})
            # Trim to max_history_turns
            max_entries = self._config.max_history_turns * 2
            if len(self._assistant_history) > max_entries:
                self._assistant_history = self._assistant_history[-max_entries:]

    def register_speaker_name(self, speaker_id: str, display_name: str) -> None:
        """Map a diarization speaker ID to a human-readable name."""
        self._speaker_names[speaker_id] = display_name
        logger.debug("Speaker registered: %s → %s", speaker_id, display_name)

    # -----------------------------------------------------------------------
    # Read methods
    # -----------------------------------------------------------------------

    async def get_context(self) -> MeetingContext:
        """Return a thread-safe snapshot of the current meeting context."""
        async with self._lock:
            return MeetingContext(
                meeting_id=self._meeting_id,
                platform=self._platform,
                title=self._title,
                start_time=self._start_time,
                transcript=list(self._segments),
                chat_messages=list(self._chat),
                speaker_names=dict(self._speaker_names),
                metadata={"status": self._status.value},
            )

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Return a copy of the assistant conversation history (role/content dicts)."""
        return list(self._assistant_history)

    def detect_current_question(self) -> str | None:
        """Return the most recently detected trigger or question, if any."""
        question = self._last_question
        self._last_question = None  # consume it
        return question

    # -----------------------------------------------------------------------
    # Context formatting for LLM
    # -----------------------------------------------------------------------

    async def build_llm_context_string(self, max_transcript_segments: int = 30) -> str:
        """Format current context as a structured string for the LLM system prompt."""
        context = await self.get_context()

        sections: list[str] = []

        # Meeting metadata
        sections.append(
            f"[MEETING INFO]\n"
            f"Platform: {context.platform.value}\n"
            f"Title: {context.title or 'Unknown'}\n"
            f"Participants: {context.participant_count} speaker(s) detected\n"
        )

        # Recent transcript
        recent_segments = context.transcript[-max_transcript_segments:]
        if recent_segments:
            transcript_lines = []
            for seg in recent_segments:
                name = context.get_speaker_name(seg.speaker_id)
                transcript_lines.append(f"{name}: {seg.text}")
            sections.append("[RECENT TRANSCRIPT]\n" + "\n".join(transcript_lines))
        else:
            sections.append("[RECENT TRANSCRIPT]\n(No transcript yet)")

        # Recent chat
        if context.chat_messages:
            chat_lines = [str(m) for m in context.chat_messages[-15:]]
            sections.append("[CHAT MESSAGES]\n" + "\n".join(chat_lines))

        return "\n\n".join(sections)

    async def get_action_items(self) -> list[str]:
        """Extract potential action items from the transcript using pattern matching.

        This is a lightweight heuristic. For higher quality, use the LLM client.
        """
        action_patterns = re.compile(
            r"\b(will|shall|going to|need to|should|must|action item|todo|follow up)"
            r"\b.{5,}",
            re.IGNORECASE,
        )
        async with self._lock:
            items = []
            for seg in self._segments:
                matches = action_patterns.findall(seg.text)
                if matches:
                    items.append(seg.text.strip())
            return items
