"""Tests for the audio capture module."""

from __future__ import annotations

import asyncio
import queue
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# Patch sounddevice before it's imported so tests work without PortAudio
_mock_sd = MagicMock()
_mock_sd.PortAudioError = OSError
sys.modules.setdefault("sounddevice", _mock_sd)

from meeting_assistant.config import Settings
from meeting_assistant.core.audio_capture import (
    AudioCaptureManager,
    AudioDeviceError,
    _find_blackhole_device,
    _find_pulseaudio_monitor,
    _get_default_loopback_device,
    list_audio_devices,
)
from meeting_assistant.models.schemas import AudioChunk


class TestAudioDeviceDiscovery:
    """Tests for device discovery helpers."""

    def test_list_audio_devices_returns_list(self) -> None:
        """list_audio_devices should always return a list."""
        with patch("sounddevice.query_devices", return_value=[
            {
                "name": "Test Device",
                "max_input_channels": 2,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
            }
        ]):
            devices = list_audio_devices()
        assert isinstance(devices, list)
        assert len(devices) == 1
        assert devices[0]["name"] == "Test Device"

    def test_list_audio_devices_handles_error(self) -> None:
        """list_audio_devices returns empty list on sounddevice error."""
        with patch("sounddevice.query_devices", side_effect=Exception("no devices")):
            devices = list_audio_devices()
        assert devices == []

    def test_find_pulseaudio_monitor_found(self) -> None:
        """Should return monitor source name when pactl output contains one."""
        mock_output = (
            "0\talsa_input.pci-0000_00_1f.3-platform-skl_hda_dsp_generic.HiFi__hw_1__source\tsink\n"
            "1\talsa_output.pci-0000_00_1f.3.analog-stereo.monitor\tsource\n"
        )
        with patch(
            "subprocess.run",
            return_value=MagicMock(stdout=mock_output, returncode=0),
        ):
            result = _find_pulseaudio_monitor()
        assert result == "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor"

    def test_find_pulseaudio_monitor_not_found(self) -> None:
        """Returns None when no monitor source exists."""
        with patch(
            "subprocess.run",
            return_value=MagicMock(stdout="0\tmic_source\tsource\n", returncode=0),
        ):
            result = _find_pulseaudio_monitor()
        assert result is None

    def test_find_pulseaudio_monitor_pactl_missing(self) -> None:
        """Returns None when pactl is not installed."""
        with patch("subprocess.run", side_effect=FileNotFoundError("pactl not found")):
            result = _find_pulseaudio_monitor()
        assert result is None

    def test_find_blackhole_device_found(self) -> None:
        """Should return BlackHole device name when present."""
        with patch(
            "sounddevice.query_devices",
            return_value=[
                {"name": "BlackHole 2ch", "max_input_channels": 2, "max_output_channels": 2},
                {"name": "Built-in Microphone", "max_input_channels": 1, "max_output_channels": 0},
            ],
        ):
            result = _find_blackhole_device()
        assert result == "BlackHole 2ch"

    def test_find_blackhole_device_not_found(self) -> None:
        """Returns None when BlackHole is not installed."""
        with patch(
            "sounddevice.query_devices",
            return_value=[{"name": "Built-in Microphone", "max_input_channels": 1, "max_output_channels": 0}],
        ):
            result = _find_blackhole_device()
        assert result is None

    def test_get_default_loopback_device_preferred_name(self) -> None:
        """Preferred name always takes precedence over auto-detection."""
        result = _get_default_loopback_device("linux", preferred_name="my-device")
        assert result == "my-device"

    def test_get_default_loopback_device_linux_auto(self) -> None:
        """Linux falls back to PulseAudio monitor auto-detection."""
        with patch(
            "meeting_assistant.core.audio_capture._find_pulseaudio_monitor",
            return_value="default.monitor",
        ):
            result = _get_default_loopback_device("linux", preferred_name="")
        assert result == "default.monitor"

    def test_get_default_loopback_device_windows(self) -> None:
        """Windows returns None (pyaudiowpatch handles internally)."""
        result = _get_default_loopback_device("windows", preferred_name="")
        assert result is None


class TestAudioCaptureManager:
    """Tests for AudioCaptureManager."""

    @pytest.fixture
    def manager(self, test_settings: Settings) -> AudioCaptureManager:
        return AudioCaptureManager(test_settings)

    def test_init_creates_queues(self, manager: AudioCaptureManager) -> None:
        """Manager initialises with correct queue types."""
        import asyncio
        assert hasattr(manager, "_output_q")
        assert isinstance(manager._output_q, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_drain_queue_yields_chunks(self, manager: AudioCaptureManager) -> None:
        """_drain_queue converts thread queue entries to AudioChunks in output queue."""
        import asyncio

        manager._loop = asyncio.get_running_loop()
        manager._running = True

        # Pre-populate the mic raw queue
        audio_data = np.zeros(480, dtype=np.float32)
        manager._mic_q.put_nowait(audio_data)

        # Drain one item
        task = asyncio.create_task(manager._drain_queue(manager._mic_q, "mic"))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Check output queue has the chunk
        assert not manager._output_q.empty()
        chunk = manager._output_q.get_nowait()
        assert isinstance(chunk, AudioChunk)
        assert chunk.source == "mic"

    @pytest.mark.asyncio
    async def test_stream_stops_when_not_running(self, manager: AudioCaptureManager) -> None:
        """stream() exits cleanly when _running is set to False."""
        manager._running = False

        chunks = []
        async for chunk in manager.stream():
            chunks.append(chunk)
            break  # should not be reached

        assert chunks == []

    @pytest.mark.asyncio
    async def test_stop_cancels_tasks_and_closes_streams(
        self, manager: AudioCaptureManager
    ) -> None:
        """stop() cancels tasks and marks running=False."""
        import asyncio

        manager._running = True
        mock_task = asyncio.create_task(asyncio.sleep(100))
        manager._tasks = [mock_task]

        # Mock stream objects
        mock_stream = MagicMock()
        mock_stream.stop = MagicMock()
        mock_stream.close = MagicMock()
        manager._mic_stream = mock_stream

        await manager.stop()

        assert manager._running is False
        assert mock_task.cancelled()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()
