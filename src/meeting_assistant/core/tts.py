"""Text-to-speech engine with primary edge-tts and offline pyttsx3 fallback.

edge-tts uses Microsoft's Edge browser TTS service (free, no API key required,
async-native, high-quality voices). pyttsx3 is a fully offline fallback using
OS-bundled TTS engines.

Audio is played locally to the user's speakers/headphones, NOT injected into
the meeting channel — only the user hears the assistant's voice.
"""

from __future__ import annotations

import asyncio
import io
import logging
import platform
import tempfile
from pathlib import Path

from meeting_assistant.config import Settings

logger = logging.getLogger(__name__)


class TTSEngine:
    """Text-to-speech with edge-tts primary and pyttsx3 fallback.

    Usage::

        engine = TTSEngine(settings)
        await engine.initialize()
        await engine.speak("Hello, I'm your meeting assistant.")
        await engine.shutdown()
    """

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._voice = config.tts_voice
        self._use_edge = True  # set False if edge-tts unavailable
        self._pyttsx3_engine = None
        self._lock = asyncio.Lock()  # prevent overlapping speech

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def initialize(self) -> None:
        """Check available TTS backends and configure."""
        # Try edge-tts first
        try:
            import edge_tts  # type: ignore[import]  # noqa: F401
            self._use_edge = True
            logger.info("TTS engine: edge-tts (voice: %s)", self._voice)
        except ImportError:
            logger.warning(
                "edge-tts not installed. Falling back to pyttsx3 (lower quality). "
                "Install with: pip install edge-tts"
            )
            self._use_edge = False
            await self._init_pyttsx3()

    async def shutdown(self) -> None:
        """Release TTS resources."""
        if self._pyttsx3_engine:
            try:
                self._pyttsx3_engine.stop()
            except Exception:
                pass
            self._pyttsx3_engine = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def speak(self, text: str) -> None:
        """Speak the given text through local audio output.

        This method is safe to call from concurrent contexts — it uses a lock
        to prevent overlapping speech.

        Args:
            text: Text to synthesize and play.
        """
        if not text.strip():
            return

        async with self._lock:
            if self._use_edge:
                await self._speak_edge_tts(text)
            else:
                await self._speak_pyttsx3(text)

    async def synthesize_to_bytes(self, text: str) -> bytes:
        """Synthesize text to MP3 audio bytes without playing.

        Useful for tests or injecting audio into a virtual microphone.

        Args:
            text: Text to synthesize.

        Returns:
            MP3 audio bytes (edge-tts) or WAV bytes (pyttsx3 fallback).
        """
        if self._use_edge:
            return await self._edge_tts_to_bytes(text)
        return b""  # pyttsx3 doesn't support easy byte output

    # -----------------------------------------------------------------------
    # edge-tts implementation
    # -----------------------------------------------------------------------

    async def _speak_edge_tts(self, text: str) -> None:
        """Synthesize and play via edge-tts."""
        try:
            audio_bytes = await self._edge_tts_to_bytes(text)
            await self._play_audio_bytes(audio_bytes)
        except Exception as exc:
            logger.warning("edge-tts failed: %s. Attempting pyttsx3 fallback.", exc)
            await self._speak_pyttsx3(text)

    async def _edge_tts_to_bytes(self, text: str) -> bytes:
        """Use edge-tts to generate MP3 audio bytes."""
        import edge_tts  # type: ignore[import]

        communicate = edge_tts.Communicate(text, self._voice)
        chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)

    async def _play_audio_bytes(self, audio_bytes: bytes) -> None:
        """Decode and play MP3 audio bytes through the default output device."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._play_mp3_sync, audio_bytes)

    @staticmethod
    def _play_mp3_sync(audio_bytes: bytes) -> None:
        """Synchronous MP3 playback (runs in thread pool)."""
        try:
            # Try pydub + sounddevice for clean cross-platform playback
            import sounddevice as sd  # type: ignore[import]
            from pydub import AudioSegment  # type: ignore[import]

            segment = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
            samples = segment.get_array_of_samples()
            import numpy as np

            arr = np.array(samples, dtype=np.float32) / 32768.0
            if segment.channels == 2:
                arr = arr.reshape(-1, 2).mean(axis=1)  # stereo → mono
            sd.play(arr, samplerate=segment.frame_rate, blocking=True)

        except ImportError:
            # Fallback: write to temp file and use OS player
            TTSEngine._play_with_os_player(audio_bytes)
        except Exception as exc:
            logger.warning("Audio playback failed: %s", exc)

    @staticmethod
    def _play_with_os_player(audio_bytes: bytes) -> None:
        """Write MP3 to a temp file and play with the OS default player."""
        import subprocess
        import sys

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name

        try:
            system = platform.system()
            if system == "Darwin":
                subprocess.run(["afplay", tmp_path], check=True, capture_output=True)
            elif system == "Linux":
                # Try common Linux audio players in order
                for player in ["mpg123", "mpg321", "ffplay", "aplay"]:
                    try:
                        subprocess.run(
                            [player, "-q", tmp_path],
                            check=True,
                            capture_output=True,
                        )
                        break
                    except (FileNotFoundError, subprocess.CalledProcessError):
                        continue
            elif system == "Windows":
                import winsound  # type: ignore[import]

                # winsound doesn't support MP3; fall through silently
                logger.warning("Install pydub for Windows audio playback.")
        except Exception as exc:
            logger.warning("OS audio player failed: %s", exc)
        finally:
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

    # -----------------------------------------------------------------------
    # pyttsx3 implementation (offline fallback)
    # -----------------------------------------------------------------------

    async def _init_pyttsx3(self) -> None:
        """Initialise pyttsx3 in a thread pool (it's not async-native)."""
        loop = asyncio.get_running_loop()
        try:
            self._pyttsx3_engine = await loop.run_in_executor(None, self._create_pyttsx3)
            logger.info("TTS engine: pyttsx3 (offline fallback)")
        except Exception as exc:
            logger.error("pyttsx3 initialisation failed: %s", exc)
            logger.error("No TTS engine available. Responses will be text-only.")

    @staticmethod
    def _create_pyttsx3():
        """Create and configure a pyttsx3 engine (sync)."""
        import pyttsx3  # type: ignore[import]

        engine = pyttsx3.init()
        engine.setProperty("rate", 175)  # words per minute
        engine.setProperty("volume", 0.9)
        return engine

    async def _speak_pyttsx3(self, text: str) -> None:
        """Speak via pyttsx3 (blocking, runs in thread pool)."""
        if not self._pyttsx3_engine:
            logger.warning("No TTS engine available; cannot speak: %r", text[:60])
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._pyttsx3_say_sync, text)

    def _pyttsx3_say_sync(self, text: str) -> None:
        """Blocking TTS playback via pyttsx3."""
        try:
            self._pyttsx3_engine.say(text)
            self._pyttsx3_engine.runAndWait()
        except Exception as exc:
            logger.warning("pyttsx3 speech failed: %s", exc)

    # -----------------------------------------------------------------------
    # Voice management
    # -----------------------------------------------------------------------

    async def list_edge_voices(self) -> list[dict]:
        """Return available edge-tts voices (requires internet)."""
        try:
            import edge_tts  # type: ignore[import]

            voices = await edge_tts.list_voices()
            return [
                {
                    "name": v["ShortName"],
                    "locale": v["Locale"],
                    "gender": v["Gender"],
                }
                for v in voices
            ]
        except Exception as exc:
            logger.warning("Failed to list edge-tts voices: %s", exc)
            return []
