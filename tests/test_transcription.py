"""Tests for the speech recognition / transcription engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from meeting_assistant.config import Settings
from meeting_assistant.core.transcription import TranscriptionEngine
from meeting_assistant.models.schemas import TranscriptionResult

# ---------------------------------------------------------------------------
# Mock Whisper segment helper
# ---------------------------------------------------------------------------


def _make_whisper_segment(text: str, start: float, end: float, words: list | None = None):
    """Create a mock faster-whisper segment."""
    seg = MagicMock()
    seg.text = text
    seg.start = start
    seg.end = end
    seg.language = "en"
    seg.words = words or []
    return seg


def _make_whisper_word(word: str, start: float, end: float, probability: float = 0.9):
    w = MagicMock()
    w.word = word
    w.start = start
    w.end = end
    w.probability = probability
    return w


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTranscriptionEngine:
    """Unit tests for TranscriptionEngine using mocked faster-whisper."""

    @pytest.fixture
    def engine(self, test_settings: Settings) -> TranscriptionEngine:
        return TranscriptionEngine(test_settings)

    @pytest.fixture
    def mock_whisper_model(self):
        """Provide a mocked faster-whisper WhisperModel."""
        segments = [
            _make_whisper_segment(
                "Hello, this is a test.",
                start=0.0,
                end=2.5,
                words=[
                    _make_whisper_word("Hello,", 0.0, 0.5),
                    _make_whisper_word("this", 0.5, 0.7),
                    _make_whisper_word("is", 0.7, 0.9),
                    _make_whisper_word("a", 0.9, 1.0),
                    _make_whisper_word("test.", 1.0, 2.5),
                ],
            )
        ]
        info = MagicMock()
        info.language = "en"
        info.duration = 2.5

        model = MagicMock()
        model.transcribe.return_value = (iter(segments), info)
        return model

    @pytest.mark.asyncio
    async def test_initialize_loads_model(self, engine: TranscriptionEngine) -> None:
        """initialize() should load the Whisper model without error."""
        mock_model = MagicMock()
        with patch(
            "meeting_assistant.core.transcription.TranscriptionEngine._load_whisper",
            side_effect=lambda: setattr(engine, "_model", mock_model),
        ):
            with patch(
                "meeting_assistant.core.transcription.TranscriptionEngine._load_diarization"
            ):
                await engine.initialize()

        assert engine._initialized

    @pytest.mark.asyncio
    async def test_transcribe_returns_result(
        self, engine: TranscriptionEngine, mock_whisper_model, speech_audio: np.ndarray
    ) -> None:
        """transcribe() should return a TranscriptionResult with segments."""
        engine._model = mock_whisper_model
        engine._diarize_pipeline = None
        engine._initialized = True

        result = await engine.transcribe(speech_audio)

        assert isinstance(result, TranscriptionResult)
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello, this is a test."
        assert result.segments[0].speaker_id == "SPEAKER_00"

    @pytest.mark.asyncio
    async def test_transcribe_empty_audio_returns_empty(
        self, engine: TranscriptionEngine, mock_whisper_model
    ) -> None:
        """Empty audio array should return empty TranscriptionResult."""
        engine._model = mock_whisper_model
        engine._initialized = True

        result = await engine.transcribe(np.array([], dtype=np.float32))

        assert isinstance(result, TranscriptionResult)
        assert len(result.segments) == 0

    @pytest.mark.asyncio
    async def test_transcribe_not_initialized_raises(
        self, engine: TranscriptionEngine, speech_audio: np.ndarray
    ) -> None:
        """transcribe() before initialize() should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await engine.transcribe(speech_audio)

    @pytest.mark.asyncio
    async def test_transcribe_no_segments_returns_empty(
        self, engine: TranscriptionEngine, speech_audio: np.ndarray
    ) -> None:
        """When Whisper finds no segments, return empty result."""
        info = MagicMock()
        info.language = "en"
        info.duration = 2.0

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (iter([]), info)

        engine._model = mock_model
        engine._diarize_pipeline = None
        engine._initialized = True

        result = await engine.transcribe(speech_audio)

        assert len(result.segments) == 0

    def test_segments_without_diarization_single_speaker(
        self, engine: TranscriptionEngine
    ) -> None:
        """Without diarization, all segments attributed to SPEAKER_00."""
        raw = [
            _make_whisper_segment("First sentence.", 0.0, 2.0),
            _make_whisper_segment("Second sentence.", 2.5, 5.0),
        ]
        result = engine._segments_without_diarization(raw, None)

        assert len(result) == 2
        assert all(s.speaker_id == "SPEAKER_00" for s in result)
        assert result[0].text == "First sentence."
        assert result[1].text == "Second sentence."

    def test_assign_speaker_max_overlap(self, engine: TranscriptionEngine) -> None:
        """_assign_speaker returns speaker with maximum overlap."""
        timeline = [
            (0.0, 3.0, "SPEAKER_00"),
            (3.0, 6.0, "SPEAKER_01"),
        ]
        # Segment mostly in SPEAKER_00 range
        speaker = engine._assign_speaker(0.5, 2.5, timeline)
        assert speaker == "SPEAKER_00"

        # Segment mostly in SPEAKER_01 range
        speaker = engine._assign_speaker(3.5, 5.5, timeline)
        assert speaker == "SPEAKER_01"

    def test_assign_speaker_no_overlap_default(self, engine: TranscriptionEngine) -> None:
        """When no timeline overlap, default to SPEAKER_00."""
        speaker = engine._assign_speaker(10.0, 12.0, [])
        assert speaker == "SPEAKER_00"

    @pytest.mark.asyncio
    async def test_initialize_without_hf_token_skips_diarization(
        self, engine: TranscriptionEngine
    ) -> None:
        """Without HF token, diarization pipeline should remain None."""
        assert not engine._config.diarization_enabled

        mock_model = MagicMock()
        with patch(
            "meeting_assistant.core.transcription.TranscriptionEngine._load_whisper",
            side_effect=lambda: setattr(engine, "_model", mock_model),
        ):
            await engine.initialize()

        assert engine._diarize_pipeline is None
        assert engine._initialized

    @pytest.mark.asyncio
    async def test_shutdown_resets_state(self, engine: TranscriptionEngine) -> None:
        """shutdown() clears model references."""
        engine._model = MagicMock()
        engine._initialized = True
        await engine.shutdown()

        assert engine._model is None
        assert not engine._initialized

    def test_words_confidence_averaged(self, engine: TranscriptionEngine) -> None:
        """Confidence score should be the average word probability."""
        raw = [
            _make_whisper_segment(
                "Test.",
                0.0,
                1.0,
                words=[
                    _make_whisper_word("Test.", 0.0, 1.0, probability=0.8),
                ],
            )
        ]
        result = engine._segments_without_diarization(raw, None)
        assert abs(result[0].confidence - 0.8) < 0.01
