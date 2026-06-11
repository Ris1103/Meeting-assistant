"""Tests for the meeting context tracker."""

from __future__ import annotations

import asyncio

import pytest

from meeting_assistant.config import Settings
from meeting_assistant.core.context_tracker import MeetingContextTracker
from meeting_assistant.models.schemas import (
    ChatMessage,
    MeetingContext,
    MeetingStatus,
    Platform,
    SpeakerSegment,
    TranscriptionResult,
)


class TestMeetingContextTracker:
    """Tests for MeetingContextTracker."""

    @pytest.fixture
    def tracker(self, test_settings: Settings) -> MeetingContextTracker:
        t = MeetingContextTracker(test_settings)
        t.start_meeting("meeting-001", Platform.ZOOM, title="Test Meeting")
        return t

    def test_start_meeting_initialises_state(self, tracker: MeetingContextTracker) -> None:
        """start_meeting sets correct initial state."""
        assert tracker._meeting_id == "meeting-001"
        assert tracker._platform == Platform.ZOOM
        assert tracker._title == "Test Meeting"
        assert tracker._status == MeetingStatus.ACTIVE
        assert len(tracker._segments) == 0
        assert len(tracker._chat) == 0

    def test_end_meeting_sets_status(self, tracker: MeetingContextTracker) -> None:
        tracker.end_meeting()
        assert tracker._status == MeetingStatus.ENDED

    @pytest.mark.asyncio
    async def test_add_segment_appends_to_deque(
        self, tracker: MeetingContextTracker, speaker_segment: SpeakerSegment
    ) -> None:
        await tracker.add_segment(speaker_segment)
        assert len(tracker._segments) == 1
        assert tracker._segments[0] == speaker_segment

    @pytest.mark.asyncio
    async def test_add_segment_respects_maxlen(
        self, test_settings: Settings, speaker_segment: SpeakerSegment
    ) -> None:
        """Deque should not exceed transcript_window_size."""
        test_settings = Settings(
            anthropic_api_key="test",
            transcript_window_size=3,
        )
        tracker = MeetingContextTracker(test_settings)
        tracker.start_meeting("m", Platform.GENERIC)

        for i in range(5):
            seg = SpeakerSegment(
                speaker_id=f"SPEAKER_{i:02d}",
                start=float(i),
                end=float(i + 1),
                text=f"Segment {i}",
                confidence=0.9,
            )
            await tracker.add_segment(seg)

        # Only the last 3 should remain
        assert len(tracker._segments) == 3
        assert tracker._segments[-1].text == "Segment 4"

    @pytest.mark.asyncio
    async def test_add_result_adds_all_segments(
        self,
        tracker: MeetingContextTracker,
        transcription_result: TranscriptionResult,
    ) -> None:
        await tracker.add_result(transcription_result)
        assert len(tracker._segments) == len(transcription_result.segments)

    @pytest.mark.asyncio
    async def test_add_chat_message(
        self, tracker: MeetingContextTracker, chat_message: ChatMessage
    ) -> None:
        await tracker.add_chat_message(chat_message)
        assert len(tracker._chat) == 1
        assert tracker._chat[0] == chat_message

    @pytest.mark.asyncio
    async def test_trigger_keyword_detection(
        self,
        tracker: MeetingContextTracker,
        speaker_segment_trigger: SpeakerSegment,
    ) -> None:
        """Trigger keyword should be detected and available via detect_current_question."""
        await tracker.add_segment(speaker_segment_trigger)
        question = tracker.detect_current_question()
        assert question is not None
        assert "test" in question.lower()

    @pytest.mark.asyncio
    async def test_detect_current_question_consumes_trigger(
        self,
        tracker: MeetingContextTracker,
        speaker_segment_trigger: SpeakerSegment,
    ) -> None:
        """detect_current_question() should consume the trigger (idempotent)."""
        await tracker.add_segment(speaker_segment_trigger)
        first_call = tracker.detect_current_question()
        second_call = tracker.detect_current_question()
        assert first_call is not None
        assert second_call is None

    @pytest.mark.asyncio
    async def test_no_trigger_in_normal_segment(
        self, tracker: MeetingContextTracker, speaker_segment: SpeakerSegment
    ) -> None:
        """Segment without trigger keyword should not set last_question."""
        await tracker.add_segment(speaker_segment)
        assert tracker.detect_current_question() is None

    @pytest.mark.asyncio
    async def test_add_assistant_turn_stores_history(
        self, tracker: MeetingContextTracker
    ) -> None:
        await tracker.add_assistant_turn("What are the action items?", "There are 3 action items.")
        history = tracker.get_conversation_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_assistant_history_trimmed(self, test_settings: Settings) -> None:
        """History should be trimmed to max_history_turns."""
        test_settings = Settings(anthropic_api_key="test", max_history_turns=2)
        tracker = MeetingContextTracker(test_settings)
        tracker.start_meeting("m", Platform.GENERIC)

        for i in range(5):
            await tracker.add_assistant_turn(f"Q{i}", f"A{i}")

        history = tracker.get_conversation_history()
        assert len(history) == 4  # 2 turns × 2 entries each

    @pytest.mark.asyncio
    async def test_register_speaker_name(self, tracker: MeetingContextTracker) -> None:
        tracker.register_speaker_name("SPEAKER_00", "Alice")
        context = await tracker.get_context()
        assert context.get_speaker_name("SPEAKER_00") == "Alice"
        assert context.get_speaker_name("UNKNOWN") == "UNKNOWN"

    @pytest.mark.asyncio
    async def test_get_context_returns_snapshot(
        self,
        tracker: MeetingContextTracker,
        speaker_segment: SpeakerSegment,
        chat_message: ChatMessage,
    ) -> None:
        await tracker.add_segment(speaker_segment)
        await tracker.add_chat_message(chat_message)
        context = await tracker.get_context()

        assert isinstance(context, MeetingContext)
        assert context.meeting_id == "meeting-001"
        assert len(context.transcript) == 1
        assert len(context.chat_messages) == 1

    @pytest.mark.asyncio
    async def test_concurrent_add_and_snapshot(
        self, tracker: MeetingContextTracker, multi_speaker_segments: list[SpeakerSegment]
    ) -> None:
        """Concurrent add_segment and get_context should not race."""
        async def _add_all():
            for seg in multi_speaker_segments:
                await tracker.add_segment(seg)

        async def _snapshot():
            for _ in range(10):
                ctx = await tracker.get_context()
                assert isinstance(ctx, MeetingContext)
                await asyncio.sleep(0)

        await asyncio.gather(_add_all(), _snapshot())
        context = await tracker.get_context()
        assert len(context.transcript) == len(multi_speaker_segments)

    @pytest.mark.asyncio
    async def test_build_llm_context_string(
        self,
        tracker: MeetingContextTracker,
        multi_speaker_segments: list[SpeakerSegment],
    ) -> None:
        for seg in multi_speaker_segments:
            await tracker.add_segment(seg)

        ctx_str = await tracker.build_llm_context_string()

        assert "[MEETING INFO]" in ctx_str
        assert "[RECENT TRANSCRIPT]" in ctx_str
        assert "SPEAKER_00" in ctx_str or "SPEAKER" in ctx_str

    @pytest.mark.asyncio
    async def test_get_action_items_detects_patterns(
        self, tracker: MeetingContextTracker
    ) -> None:
        """Action item patterns should be detected in transcript."""
        seg = SpeakerSegment(
            speaker_id="SPEAKER_00",
            start=0.0,
            end=5.0,
            text="I will finalize the design document by Friday.",
            confidence=0.9,
        )
        await tracker.add_segment(seg)
        items = await tracker.get_action_items()
        assert len(items) >= 1
