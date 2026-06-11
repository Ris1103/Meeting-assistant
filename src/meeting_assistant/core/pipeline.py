"""Main meeting assistant pipeline orchestrator.

Wires together all core components into a single async pipeline:

  Audio Capture → VAD Buffer → WhisperX STT → Context Tracker
       ↑                                             ↓
  Platform Adapter (chat) ──────────────────────────→┘
                                                     ↓
                                           (trigger detected?)
                                                     ↓
                                           Claude LLM → TTS Output

The pipeline runs as a set of concurrent asyncio tasks managed by
asyncio.TaskGroup (Python 3.11+). Each component communicates via
asyncio.Queue or direct method calls.

Privacy note:
  - A consent prompt is shown before any audio capture begins.
  - Raw audio never leaves the local machine (Whisper runs locally).
  - Only transcript text is sent to the Claude API.
  - No data is persisted unless the user opts in via TRANSCRIPT_DIR.
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from meeting_assistant.config import Settings
from meeting_assistant.core.audio_capture import AudioCaptureManager
from meeting_assistant.core.context_tracker import MeetingContextTracker
from meeting_assistant.core.llm_client import ClaudeClient
from meeting_assistant.core.transcription import TranscriptionEngine
from meeting_assistant.core.tts import TTSEngine
from meeting_assistant.models.schemas import MeetingStatus, Platform
from meeting_assistant.platforms.base import PlatformAdapter

logger = logging.getLogger(__name__)


class MeetingAssistantPipeline:
    """Top-level pipeline that orchestrates all meeting assistant components.

    Usage::

        pipeline = MeetingAssistantPipeline(settings, platform_adapter)
        await pipeline.start()          # begins capturing and processing
        # ... meeting proceeds ...
        await pipeline.stop()           # graceful shutdown
        summary = await pipeline.get_summary()
    """

    def __init__(
        self,
        config: Settings,
        platform_adapter: PlatformAdapter | None = None,
    ) -> None:
        self._config = config
        self._platform = platform_adapter

        # Core components
        self._audio = AudioCaptureManager(config)
        self._stt = TranscriptionEngine(config)
        self._context = MeetingContextTracker(config)
        self._llm = ClaudeClient(config)
        self._tts = TTSEngine(config)

        # State
        self._running = False
        self._status = MeetingStatus.IDLE
        self._tasks: list[asyncio.Task] = []

        # VAD: accumulate audio until silence or buffer limit
        self._speech_buffer: list[np.ndarray] = []
        self._last_speech_time: float = 0.0
        self._silence_threshold_s: float = 1.5  # flush after 1.5s silence

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def start(
        self,
        meeting_id: str = "unknown",
        platform: Platform = Platform.GENERIC,
        title: str | None = None,
    ) -> None:
        """Initialise all components and begin the pipeline.

        Args:
            meeting_id: Platform-specific meeting identifier.
            platform: Which meeting platform this session is for.
            title: Optional meeting title from calendar/API.
        """
        logger.info("=== Meeting Assistant Pipeline Starting ===")

        if not await self._check_consent():
            logger.warning("User declined consent — aborting pipeline start.")
            return

        # Initialise components
        logger.info("Initialising components…")
        await self._stt.initialize()
        await self._tts.initialize()

        # Start meeting context
        self._context.start_meeting(meeting_id, platform, title)

        # Register speaker names from platform if available
        if self._platform:
            try:
                metadata = await self._platform.get_meeting_metadata()
                participants = metadata.get("participants", [])
                for i, name in enumerate(participants):
                    self._context.register_speaker_name(f"SPEAKER_{i:02d}", name)
            except Exception as exc:
                logger.debug("Failed to fetch participant list: %s", exc)

        # Start audio capture
        await self._audio.start()

        self._running = True
        self._status = MeetingStatus.ACTIVE

        logger.info(
            "Pipeline active — say '%s' to trigger the assistant.",
            self._config.trigger_keyword,
        )

        # Launch concurrent pipeline tasks
        self._tasks = [
            asyncio.create_task(self._audio_pipeline_task(), name="audio-pipeline"),
            asyncio.create_task(self._response_trigger_task(), name="response-trigger"),
        ]

        if self._platform:
            self._tasks.append(
                asyncio.create_task(self._chat_ingestion_task(), name="chat-ingestion")
            )

    async def stop(self) -> None:
        """Gracefully stop all pipeline tasks and release resources."""
        logger.info("Stopping pipeline…")
        self._running = False
        self._status = MeetingStatus.ENDED

        # Cancel all running tasks
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Shutdown components
        await self._audio.stop()
        await self._stt.shutdown()
        await self._tts.shutdown()
        self._context.end_meeting()

        logger.info("Pipeline stopped.")

    async def run_until_stopped(self) -> None:
        """Block until the pipeline is stopped (e.g. via KeyboardInterrupt).

        Call after start() to keep the pipeline running in a background async loop.
        """
        try:
            while self._running:
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    # -----------------------------------------------------------------------
    # Pipeline task: audio processing
    # -----------------------------------------------------------------------

    async def _audio_pipeline_task(self) -> None:
        """Main audio pipeline: capture → buffer → STT → context."""
        logger.debug("Audio pipeline task started")

        buffer: list[np.ndarray] = []
        last_speech_t = time.monotonic()

        try:
            async for chunk in self._audio.stream():
                if not self._running:
                    break

                arr = np.frombuffer(chunk.data, dtype=np.float32)
                rms = float(np.sqrt(np.mean(arr**2)))
                is_speech = rms > 0.005  # Simple energy-based VAD

                if is_speech:
                    buffer.append(arr)
                    last_speech_t = time.monotonic()
                elif buffer:
                    # Flush buffer after silence
                    silence_duration = time.monotonic() - last_speech_t
                    if silence_duration >= self._silence_threshold_s:
                        await self._flush_audio_buffer(buffer)
                        buffer = []

                # Also flush if buffer exceeds max duration
                total_samples = sum(a.shape[0] for a in buffer)
                max_samples = int(
                    self._config.transcription_buffer_seconds * self._config.audio_sample_rate
                )
                if total_samples >= max_samples:
                    await self._flush_audio_buffer(buffer)
                    buffer = []
                    last_speech_t = time.monotonic()

        except asyncio.CancelledError:
            logger.debug("Audio pipeline task cancelled")
        except Exception as exc:
            logger.error("Audio pipeline task error: %s", exc, exc_info=True)

    async def _flush_audio_buffer(self, buffer: list[np.ndarray]) -> None:
        """Transcribe accumulated audio buffer and update context."""
        if not buffer:
            return

        audio = np.concatenate(buffer)
        duration = len(audio) / self._config.audio_sample_rate

        if duration < 0.3:  # too short to transcribe
            return

        logger.debug("Transcribing %.1fs audio buffer…", duration)
        try:
            result = await self._stt.transcribe(audio)
            if result.segments:
                await self._context.add_result(result)
                for seg in result.segments:
                    logger.info("[%s] %s", seg.speaker_id, seg.text)
        except Exception as exc:
            logger.warning("Transcription failed: %s", exc)

    # -----------------------------------------------------------------------
    # Pipeline task: trigger detection and response generation
    # -----------------------------------------------------------------------

    async def _response_trigger_task(self) -> None:
        """Poll for trigger keywords and generate LLM responses."""
        logger.debug("Response trigger task started")
        poll_interval = 0.5  # seconds

        try:
            while self._running:
                await asyncio.sleep(poll_interval)

                question = self._context.detect_current_question()
                if not question:
                    continue

                logger.info("Trigger detected: %r", question[:80])

                # Build context string for LLM
                context_str = await self._context.build_llm_context_string()
                history = self._context.get_conversation_history()

                # Announce we're processing (non-blocking)
                await self._tts.speak("Let me think about that…")

                # Stream response with sentence-by-sentence TTS
                response_chunks: list[str] = []

                async def _on_sentence(sentence: str) -> None:
                    await self._tts.speak(sentence)

                async for chunk in self._llm.answer_streaming(
                    question=question,
                    context_str=context_str,
                    conversation_history=history,
                    sentence_callback=_on_sentence,
                ):
                    response_chunks.append(chunk)

                full_response = "".join(response_chunks)
                await self._context.add_assistant_turn(question, full_response)

                logger.info("Assistant responded: %r…", full_response[:100])

        except asyncio.CancelledError:
            logger.debug("Response trigger task cancelled")
        except Exception as exc:
            logger.error("Response trigger task error: %s", exc, exc_info=True)

    # -----------------------------------------------------------------------
    # Pipeline task: chat ingestion from platform
    # -----------------------------------------------------------------------

    async def _chat_ingestion_task(self) -> None:
        """Stream chat messages from the platform adapter into context."""
        if not self._platform:
            return

        logger.debug("Chat ingestion task started (platform: %s)", self._platform.PLATFORM_NAME)
        try:
            async for message in self._platform.stream_chat_messages():
                await self._context.add_chat_message(message)
                logger.debug("Chat: %s", message)
        except asyncio.CancelledError:
            logger.debug("Chat ingestion task cancelled")
        except Exception as exc:
            logger.warning("Chat ingestion error: %s", exc)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    async def get_summary(self) -> str:
        """Generate a meeting summary via LLM. Call after stop()."""
        context = await self._context.get_context()
        if not context.transcript:
            return "No transcript available for summary."
        return await self._llm.summarize(context)

    async def get_action_items(self) -> list[str]:
        """Extract action items from the meeting via LLM."""
        context = await self._context.get_context()
        return await self._llm.extract_action_items(context)

    @property
    def status(self) -> MeetingStatus:
        return self._status

    async def _check_consent(self) -> bool:
        """Prompt the user for recording consent.

        Returns True if consent is granted, False otherwise.
        """
        print(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║              MEETING ASSISTANT — CONSENT NOTICE             ║\n"
            "╠══════════════════════════════════════════════════════════════╣\n"
            "║  This assistant will RECORD and TRANSCRIBE audio from this  ║\n"
            "║  meeting session for real-time AI assistance.               ║\n"
            "║                                                              ║\n"
            "║  • Audio is processed locally (Whisper STT on-device).      ║\n"
            "║  • Only transcript TEXT is sent to the Anthropic Claude API. ║\n"
            "║  • No audio is stored beyond this session.                   ║\n"
            "║  • Ensure you have consent from all meeting participants.    ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n"
        )
        if self._config.no_store:
            print("  ℹ  Storage disabled (--no-store / NO_STORE=true)\n")

        try:
            response = input("Do you have participant consent to record? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False

        if response not in ("y", "yes"):
            print("Recording cancelled. Exiting.")
            return False

        print("✓ Consent confirmed. Starting meeting assistant…\n")
        return True
