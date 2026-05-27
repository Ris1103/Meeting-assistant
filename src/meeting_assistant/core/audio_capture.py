"""Cross-platform audio capture: system loopback + microphone.

Captures two audio streams simultaneously:
  - **Loopback**: System speaker output (what meeting participants say).
  - **Microphone**: Local user's voice.

Both streams are 16kHz, mono, float32 — optimal for Whisper STT.

Platform-specific loopback strategies:
  - Linux:   PulseAudio/PipeWire monitor source (auto-detected via pactl)
  - Windows: WASAPI loopback via pyaudiowpatch
  - macOS:   BlackHole 2ch virtual device (user must install BlackHole)
"""

from __future__ import annotations

import asyncio
import logging
import platform
import queue
import struct
import subprocess
import sys
import time
from typing import AsyncGenerator

import numpy as np

from meeting_assistant.config import Settings
from meeting_assistant.models.schemas import AudioChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AudioDeviceError(Exception):
    """Raised when the required audio device cannot be found or opened."""


# ---------------------------------------------------------------------------
# Helper: device discovery
# ---------------------------------------------------------------------------


def _get_default_loopback_device(os_name: str, preferred_name: str = "") -> str | None:
    """Return the best loopback device name for the current OS.

    Returns None if no loopback device is found (user needs to set up manually).
    """
    if preferred_name:
        return preferred_name

    if os_name == "linux":
        return _find_pulseaudio_monitor()
    elif os_name == "macos":
        return _find_blackhole_device()
    elif os_name == "windows":
        return None  # pyaudiowpatch handles device selection internally
    return None


def _find_pulseaudio_monitor() -> str | None:
    """Find a PulseAudio/PipeWire monitor source (Linux)."""
    try:
        result = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            # PulseAudio monitor sources end with ".monitor"
            if ".monitor" in line:
                parts = line.split()
                if len(parts) >= 2:
                    source_name = parts[1]
                    logger.info("Found PulseAudio monitor source: %s", source_name)
                    return source_name
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug("pactl not available or failed: %s", e)
    return None


def _find_blackhole_device() -> str | None:
    """Find BlackHole virtual audio device (macOS)."""
    try:
        import sounddevice as sd  # lazy import

        devices = sd.query_devices()
        for dev in devices:
            if isinstance(dev, dict) and "BlackHole" in dev.get("name", ""):
                logger.info("Found BlackHole device: %s", dev["name"])
                return dev["name"]
    except Exception as e:
        logger.debug("sounddevice query failed: %s", e)
    return None


def list_audio_devices() -> list[dict]:
    """Return all available audio devices as a list of dicts."""
    try:
        import sounddevice as sd  # lazy import

        devices = sd.query_devices()
        return [
            {
                "index": i,
                "name": dev["name"],
                "max_input_channels": dev["max_input_channels"],
                "max_output_channels": dev["max_output_channels"],
                "default_samplerate": dev["default_samplerate"],
            }
            for i, dev in enumerate(devices)
        ]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Main audio capture class
# ---------------------------------------------------------------------------


class AudioCaptureManager:
    """Captures microphone and system loopback audio streams concurrently.

    Usage::

        manager = AudioCaptureManager(settings)
        await manager.start()
        async for chunk in manager.stream():
            process(chunk)
        await manager.stop()
    """

    CHUNK_DTYPE = "float32"

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._sample_rate = config.audio_sample_rate
        self._samples_per_frame = config.samples_per_frame
        self._os_name = config.os_name

        # Thread-safe queues for sounddevice callbacks → async world
        self._mic_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)
        self._loopback_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=100)

        # Unified async queue: mixes both sources
        self._output_q: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=200)

        self._mic_stream: sd.InputStream | None = None
        self._loopback_stream: sd.InputStream | None = None
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._tasks: list[asyncio.Task] = []

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def start(self) -> None:
        """Open audio device streams and begin capturing."""
        self._loop = asyncio.get_running_loop()
        self._running = True

        logger.info("Starting audio capture (OS: %s, sample_rate: %d Hz)", self._os_name, self._sample_rate)

        # Start microphone capture
        import sounddevice as sd  # lazy import

        mic_device = self._config.mic_device_index  # None = default
        self._mic_stream = sd.InputStream(
            device=mic_device,
            samplerate=self._sample_rate,
            channels=1,
            dtype=self.CHUNK_DTYPE,
            blocksize=self._samples_per_frame,
            callback=self._mic_callback,
        )
        self._mic_stream.start()
        logger.info("Microphone stream opened (device index: %s)", mic_device)

        # Start loopback capture (platform-specific)
        await self._start_loopback()

        # Background tasks to drain the raw queues into the async output queue
        self._tasks = [
            asyncio.create_task(self._drain_queue(self._mic_q, "mic"), name="drain-mic"),
            asyncio.create_task(self._drain_queue(self._loopback_q, "loopback"), name="drain-loopback"),
        ]

    async def stop(self) -> None:
        """Gracefully stop all audio streams."""
        self._running = False

        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        if self._mic_stream:
            self._mic_stream.stop()
            self._mic_stream.close()
            self._mic_stream = None

        if self._loopback_stream:
            self._loopback_stream.stop()
            self._loopback_stream.close()
            self._loopback_stream = None

        logger.info("Audio capture stopped")

    async def stream(self) -> AsyncGenerator[AudioChunk, None]:
        """Async generator that yields audio chunks from both mic and loopback."""
        while self._running:
            try:
                chunk = await asyncio.wait_for(self._output_q.get(), timeout=1.0)
                yield chunk
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    def get_queue(self) -> asyncio.Queue[AudioChunk]:
        """Return the output queue directly (for polling-based consumers)."""
        return self._output_q

    # -----------------------------------------------------------------------
    # Loopback setup
    # -----------------------------------------------------------------------

    async def _start_loopback(self) -> None:
        """Configure and start the loopback audio stream (platform-specific)."""
        if self._os_name == "windows":
            await self._start_windows_loopback()
        else:
            await self._start_sounddevice_loopback()

    async def _start_sounddevice_loopback(self) -> None:
        """Start loopback via sounddevice (Linux/macOS)."""
        import sounddevice as sd  # lazy import

        device_name = _get_default_loopback_device(self._os_name, self._config.loopback_device_name)

        if device_name is None:
            if self._os_name == "linux":
                instructions = (
                    "No PulseAudio monitor source found. "
                    "Ensure PulseAudio/PipeWire is running, or set LOOPBACK_DEVICE_NAME "
                    "to a valid monitor source (run: pactl list short sources)."
                )
            elif self._os_name == "macos":
                instructions = (
                    "No BlackHole device found. Install BlackHole from "
                    "https://github.com/ExistentialAudio/BlackHole and set your "
                    "system output to BlackHole, then restart the assistant. "
                    "Or set LOOPBACK_DEVICE_NAME to your loopback device name."
                )
            else:
                instructions = "Set LOOPBACK_DEVICE_NAME to your loopback audio device."

            logger.warning(
                "Loopback device not found — running with microphone only. %s",
                instructions,
            )
            return

        try:
            self._loopback_stream = sd.InputStream(
                device=device_name,
                samplerate=self._sample_rate,
                channels=1,
                dtype=self.CHUNK_DTYPE,
                blocksize=self._samples_per_frame,
                callback=self._loopback_callback,
            )
            self._loopback_stream.start()
            logger.info("Loopback stream opened (device: %s)", device_name)
        except sd.PortAudioError as exc:
            logger.warning("Failed to open loopback device '%s': %s", device_name, exc)

    async def _start_windows_loopback(self) -> None:
        """Start WASAPI loopback capture on Windows (via pyaudiowpatch)."""
        try:
            import pyaudiowpatch as pyaudio  # type: ignore[import]

            p = pyaudio.PyAudio()
            # Find default output device's loopback
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_output = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            loopback_device = None
            for idx in range(p.get_device_count()):
                dev = p.get_device_info_by_index(idx)
                if (
                    dev.get("isLoopbackDevice", False)
                    and dev["name"] == default_output["name"] + " [Loopback]"
                ):
                    loopback_device = dev
                    break

            if loopback_device is None:
                logger.warning("No WASAPI loopback device found; audio capture limited to mic.")
                p.terminate()
                return

            logger.info("Windows WASAPI loopback: %s", loopback_device["name"])

            def _wasapi_callback(in_data, frame_count, time_info, status):
                arr = np.frombuffer(in_data, dtype=np.float32).copy()
                self._loopback_q.put_nowait(arr)
                return (None, pyaudio.paContinue)

            stream = p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self._sample_rate,
                frames_per_buffer=self._samples_per_frame,
                input=True,
                input_device_index=loopback_device["index"],
                stream_callback=_wasapi_callback,
            )
            stream.start_stream()
            # Store for cleanup (pyaudio stream doesn't subclass sd.InputStream)
            self._loopback_stream = stream  # type: ignore[assignment]
            logger.info("Windows WASAPI loopback stream started")
        except ImportError:
            logger.warning(
                "pyaudiowpatch not installed. Install with: pip install pyaudiowpatch. "
                "Running with microphone only."
            )
        except Exception as exc:
            logger.warning("WASAPI loopback setup failed: %s", exc)

    # -----------------------------------------------------------------------
    # sounddevice callbacks (called from C thread)
    # -----------------------------------------------------------------------

    def _mic_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        if status:
            logger.debug("Mic callback status: %s", status)
        if self._running and self._loop:
            arr = indata[:, 0].copy()  # mono
            try:
                self._mic_q.put_nowait(arr)
            except queue.Full:
                pass  # drop frame if queue is full

    def _loopback_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        if status:
            logger.debug("Loopback callback status: %s", status)
        if self._running and self._loop:
            arr = indata[:, 0].copy()  # mono
            try:
                self._loopback_q.put_nowait(arr)
            except queue.Full:
                pass

    # -----------------------------------------------------------------------
    # Async queue drainer
    # -----------------------------------------------------------------------

    async def _drain_queue(self, raw_q: queue.Queue[np.ndarray], source: str) -> None:
        """Continuously drain a thread-safe Queue into the async output queue."""
        while self._running:
            try:
                arr = raw_q.get_nowait()
                chunk = AudioChunk(
                    data=arr.tobytes(),
                    sample_rate=self._sample_rate,
                    timestamp=time.time(),
                    source=source,
                )
                await self._output_q.put(chunk)
            except queue.Empty:
                await asyncio.sleep(0.005)  # 5ms poll interval
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug("Error in _drain_queue(%s): %s", source, exc)
