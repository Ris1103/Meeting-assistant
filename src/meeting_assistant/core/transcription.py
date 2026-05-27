"""Speech-to-text transcription with optional speaker diarization.

Uses faster-whisper (a fast, memory-efficient Whisper implementation based on
CTranslate2) for transcription, with optional pyannote.audio for speaker
diarization when a HuggingFace token is provided.

The transcription engine runs all model inference in a thread pool executor
so the asyncio event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from meeting_assistant.config import Settings
from meeting_assistant.models.schemas import SpeakerSegment, TranscriptionResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TranscriptionEngine:
    """Async STT engine using faster-whisper with optional pyannote diarization.

    Usage::

        engine = TranscriptionEngine(settings)
        await engine.initialize()
        result = await engine.transcribe(audio_array)
        for segment in result.segments:
            print(segment)
        await engine.shutdown()
    """

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._model = None
        self._diarize_pipeline = None
        self._align_model = None
        self._align_metadata = None
        self._initialized = False
        self._executor = None  # uses default thread pool

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load Whisper model (and optionally pyannote) in a thread pool."""
        loop = asyncio.get_running_loop()

        logger.info(
            "Loading Whisper model '%s' (device=%s, compute_type=%s)…",
            self._config.whisper_model_size,
            self._config.whisper_device,
            self._config.whisper_compute_type,
        )

        await loop.run_in_executor(None, self._load_whisper)

        if self._config.diarization_enabled:
            logger.info("Loading pyannote diarization pipeline…")
            await loop.run_in_executor(None, self._load_diarization)
        else:
            logger.info(
                "Diarization disabled (HUGGINGFACE_TOKEN not set). "
                "All speech will be attributed to a single speaker."
            )

        self._initialized = True
        logger.info("TranscriptionEngine ready")

    async def shutdown(self) -> None:
        """Release model resources."""
        self._model = None
        self._diarize_pipeline = None
        self._initialized = False
        logger.info("TranscriptionEngine shut down")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def transcribe(
        self,
        audio: np.ndarray,
        meeting_start_time: float | None = None,
    ) -> TranscriptionResult:
        """Transcribe a numpy audio array (16kHz, mono, float32).

        Args:
            audio: PCM audio array, shape (N,), dtype float32, range [-1.0, 1.0].
            meeting_start_time: Unix timestamp of meeting start (for relative timestamps).

        Returns:
            TranscriptionResult with SpeakerSegments.
        """
        if not self._initialized:
            raise RuntimeError("TranscriptionEngine not initialized. Call initialize() first.")
        if len(audio) == 0:
            return TranscriptionResult(duration=0.0, model_name=self._config.whisper_model_size)

        loop = asyncio.get_running_loop()
        t0 = time.monotonic()

        result = await loop.run_in_executor(
            None,
            self._run_transcription,
            audio,
            meeting_start_time,
        )

        elapsed = time.monotonic() - t0
        logger.debug(
            "Transcribed %.1fs audio in %.2fs (RTF=%.2f)",
            len(audio) / self._config.audio_sample_rate,
            elapsed,
            elapsed / (len(audio) / self._config.audio_sample_rate),
        )
        return result

    # -----------------------------------------------------------------------
    # Synchronous inference (runs in thread pool)
    # -----------------------------------------------------------------------

    def _load_whisper(self) -> None:
        """Load faster-whisper model. Called from thread pool."""
        try:
            from faster_whisper import WhisperModel  # type: ignore[import]

            self._model = WhisperModel(
                self._config.whisper_model_size,
                device=self._config.whisper_device,
                compute_type=self._config.whisper_compute_type,
            )
        except ImportError:
            raise ImportError(
                "faster-whisper is not installed. "
                "Install with: pip install faster-whisper"
            )

    def _load_diarization(self) -> None:
        """Load pyannote diarization pipeline. Called from thread pool."""
        try:
            from pyannote.audio import Pipeline  # type: ignore[import]
            import torch  # type: ignore[import]

            self._diarize_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self._config.huggingface_token,
            )
            # Move to GPU if available
            if torch.cuda.is_available() and self._config.whisper_device != "cpu":
                self._diarize_pipeline = self._diarize_pipeline.to(torch.device("cuda"))
                logger.info("Diarization pipeline moved to CUDA")
        except ImportError:
            logger.warning(
                "pyannote.audio not installed. Install optional diarization extras: "
                "pip install 'meeting-assistant[diarization]'"
            )
            self._diarize_pipeline = None
        except Exception as exc:
            logger.warning("Failed to load diarization pipeline: %s", exc)
            logger.warning(
                "Ensure you have accepted the pyannote model terms at: "
                "https://huggingface.co/pyannote/speaker-diarization-3.1"
            )
            self._diarize_pipeline = None

    def _run_transcription(
        self,
        audio: np.ndarray,
        meeting_start_time: float | None,
    ) -> TranscriptionResult:
        """Run full STT (+ optional diarization) pipeline synchronously."""
        segments_raw, info = self._model.transcribe(
            audio,
            language=self._config.language if self._config.language != "auto" else None,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        # Materialise the lazy generator
        segments_list = list(segments_raw)
        duration = info.duration

        if not segments_list:
            return TranscriptionResult(
                duration=duration,
                language=info.language,
                model_name=self._config.whisper_model_size,
            )

        # Build speaker segments (no diarization = single speaker)
        if self._diarize_pipeline is not None:
            speaker_segments = self._diarize(audio, segments_list, meeting_start_time)
        else:
            speaker_segments = self._segments_without_diarization(segments_list, meeting_start_time)

        return TranscriptionResult(
            segments=speaker_segments,
            language=info.language,
            duration=duration,
            model_name=self._config.whisper_model_size,
        )

    def _segments_without_diarization(
        self,
        raw_segments: list,
        meeting_start_time: float | None,
    ) -> list[SpeakerSegment]:
        """Convert faster-whisper segments to SpeakerSegments with a single speaker."""
        result = []
        for seg in raw_segments:
            words = [
                {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                for w in (seg.words or [])
            ]
            avg_confidence = (
                sum(w["probability"] for w in words) / len(words) if words else 0.8
            )
            result.append(
                SpeakerSegment(
                    speaker_id="SPEAKER_00",
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    confidence=avg_confidence,
                    language=getattr(seg, "language", self._config.language) or self._config.language,
                )
            )
        return result

    def _diarize(
        self,
        audio: np.ndarray,
        raw_segments: list,
        meeting_start_time: float | None,
    ) -> list[SpeakerSegment]:
        """Perform speaker diarization and merge with Whisper segments."""
        import torch  # type: ignore[import]

        try:
            # pyannote expects a dict with waveform tensor
            waveform = torch.from_numpy(audio).unsqueeze(0)  # (1, T)
            diarization = self._diarize_pipeline(
                {"waveform": waveform, "sample_rate": self._config.audio_sample_rate},
                min_speakers=1,
                max_speakers=self._config.max_speakers,
            )

            # Build a speaker timeline: list of (start, end, speaker_label)
            speaker_timeline = [
                (turn.start, turn.end, speaker)
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ]

            # Assign each Whisper segment to the speaker with max overlap
            result = []
            for seg in raw_segments:
                speaker_id = self._assign_speaker(seg.start, seg.end, speaker_timeline)
                words = [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in (seg.words or [])
                ]
                avg_confidence = (
                    sum(w["probability"] for w in words) / len(words) if words else 0.8
                )
                result.append(
                    SpeakerSegment(
                        speaker_id=speaker_id,
                        start=seg.start,
                        end=seg.end,
                        text=seg.text.strip(),
                        confidence=avg_confidence,
                        language=getattr(seg, "language", self._config.language) or self._config.language,
                    )
                )
            return result

        except Exception as exc:
            logger.warning("Diarization failed (%s); falling back to single speaker.", exc)
            return self._segments_without_diarization(raw_segments, meeting_start_time)

    @staticmethod
    def _assign_speaker(
        seg_start: float,
        seg_end: float,
        timeline: list[tuple[float, float, str]],
    ) -> str:
        """Return the speaker label with the most overlap with [seg_start, seg_end]."""
        best_speaker = "SPEAKER_00"
        best_overlap = 0.0
        for t_start, t_end, speaker in timeline:
            overlap = max(0.0, min(seg_end, t_end) - max(seg_start, t_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        return best_speaker
