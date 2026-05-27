"""Shared pytest fixtures for the meeting assistant test suite."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from meeting_assistant.config import Settings
from meeting_assistant.models.schemas import (
    AudioChunk,
    ChatMessage,
    MeetingContext,
    Platform,
    SpeakerSegment,
    TranscriptionResult,
)


# ---------------------------------------------------------------------------
# Settings fixture — override with test values
# ---------------------------------------------------------------------------


@pytest.fixture
def test_settings() -> Settings:
    """Settings instance with safe test defaults (no real API calls)."""
    return Settings(
        anthropic_api_key="test-key-not-real",
        whisper_model_size="tiny",
        whisper_device="cpu",
        whisper_compute_type="int8",
        huggingface_token="",
        audio_sample_rate=16_000,
        vad_frame_duration_ms=30,
        transcription_buffer_seconds=2.0,
        trigger_keyword="test",
        language="en",
        max_speakers=4,
        response_mode="local_playback",
        tts_voice="en-US-JennyNeural",
        transcript_window_size=20,
        chat_window_size=10,
        max_history_turns=5,
        no_store=True,
    )


# ---------------------------------------------------------------------------
# Audio fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_rate() -> int:
    return 16_000


@pytest.fixture
def silence_audio(sample_rate: int) -> np.ndarray:
    """2 seconds of silence (near-zero amplitude)."""
    return np.zeros(sample_rate * 2, dtype=np.float32)


@pytest.fixture
def speech_audio(sample_rate: int) -> np.ndarray:
    """2 seconds of synthetic 'speech-like' audio (sine wave burst)."""
    t = np.linspace(0, 2.0, sample_rate * 2, dtype=np.float32)
    # 440Hz sine wave with amplitude 0.3 (audible but not clipping)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio


@pytest.fixture
def audio_chunk(sample_rate: int, speech_audio: np.ndarray) -> AudioChunk:
    """A single AudioChunk containing speech audio."""
    return AudioChunk(
        data=speech_audio.tobytes(),
        sample_rate=sample_rate,
        timestamp=time.time(),
        source="mic",
    )


# ---------------------------------------------------------------------------
# Transcript fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def speaker_segment() -> SpeakerSegment:
    """A single transcribed speaker segment."""
    return SpeakerSegment(
        speaker_id="SPEAKER_00",
        start=0.0,
        end=3.5,
        text="Hello, can you summarize the action items from the last meeting?",
        confidence=0.92,
        language="en",
    )


@pytest.fixture
def speaker_segment_trigger(test_settings: Settings) -> SpeakerSegment:
    """A segment containing the trigger keyword."""
    return SpeakerSegment(
        speaker_id="SPEAKER_01",
        start=5.0,
        end=8.0,
        text=f"Hey {test_settings.trigger_keyword}, what were the key decisions?",
        confidence=0.88,
        language="en",
    )


@pytest.fixture
def multi_speaker_segments() -> list[SpeakerSegment]:
    """Multiple segments from different speakers."""
    return [
        SpeakerSegment(
            speaker_id="SPEAKER_00",
            start=0.0,
            end=5.0,
            text="Welcome everyone. Let's start with the status update.",
            confidence=0.95,
        ),
        SpeakerSegment(
            speaker_id="SPEAKER_01",
            start=5.5,
            end=12.0,
            text="Thanks. The project is on track. We completed the API integration last week.",
            confidence=0.91,
        ),
        SpeakerSegment(
            speaker_id="SPEAKER_00",
            start=12.5,
            end=18.0,
            text="Great. What are the blockers we need to resolve this week?",
            confidence=0.93,
        ),
        SpeakerSegment(
            speaker_id="SPEAKER_02",
            start=18.5,
            end=25.0,
            text="We still need to finalize the database schema. I'll have it done by Thursday.",
            confidence=0.89,
        ),
    ]


@pytest.fixture
def transcription_result(multi_speaker_segments: list[SpeakerSegment]) -> TranscriptionResult:
    """A TranscriptionResult with multiple speaker segments."""
    return TranscriptionResult(
        segments=multi_speaker_segments,
        language="en",
        duration=25.0,
        model_name="whisper-tiny",
    )


# ---------------------------------------------------------------------------
# Chat fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def chat_message() -> ChatMessage:
    """A sample chat message from Zoom."""
    return ChatMessage(
        sender="Alice Smith",
        content="Can we add the budget discussion to the agenda?",
        timestamp=datetime.utcnow(),
        platform=Platform.ZOOM,
        message_id="msg-001",
    )


# ---------------------------------------------------------------------------
# Meeting context fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def meeting_context(
    multi_speaker_segments: list[SpeakerSegment],
    chat_message: ChatMessage,
) -> MeetingContext:
    """A populated MeetingContext snapshot."""
    return MeetingContext(
        meeting_id="test-meeting-123",
        platform=Platform.ZOOM,
        title="Weekly Standup",
        start_time=datetime.utcnow(),
        transcript=multi_speaker_segments,
        chat_messages=[chat_message],
        speaker_names={
            "SPEAKER_00": "Alice",
            "SPEAKER_01": "Bob",
            "SPEAKER_02": "Carol",
        },
    )


# ---------------------------------------------------------------------------
# Fixture: sample WAV file path
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_wav_path(tmp_path: Path, speech_audio: np.ndarray, sample_rate: int) -> Path:
    """Create a temporary WAV file from the speech_audio fixture."""
    wav_path = tmp_path / "test_audio.wav"
    try:
        import soundfile as sf  # type: ignore[import]
        sf.write(str(wav_path), speech_audio, sample_rate)
    except ImportError:
        # Fallback: raw PCM (not proper WAV, but usable for byte-level tests)
        wav_path.write_bytes(speech_audio.tobytes())
    return wav_path
