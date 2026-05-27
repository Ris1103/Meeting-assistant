"""Core processing components for the meeting assistant pipeline.

Imports are lazy to avoid errors when optional system libraries
(PortAudio/sounddevice) are not installed in the current environment.
"""

from __future__ import annotations

__all__ = [
    "AudioCaptureManager",
    "ClaudeClient",
    "MeetingAssistantPipeline",
    "MeetingContextTracker",
    "TranscriptionEngine",
    "TTSEngine",
]


def __getattr__(name: str):
    """Lazy-import core components to avoid hard PortAudio/CUDA dependency at import time."""
    if name == "AudioCaptureManager":
        from meeting_assistant.core.audio_capture import AudioCaptureManager
        return AudioCaptureManager
    if name == "MeetingContextTracker":
        from meeting_assistant.core.context_tracker import MeetingContextTracker
        return MeetingContextTracker
    if name == "ClaudeClient":
        from meeting_assistant.core.llm_client import ClaudeClient
        return ClaudeClient
    if name == "MeetingAssistantPipeline":
        from meeting_assistant.core.pipeline import MeetingAssistantPipeline
        return MeetingAssistantPipeline
    if name == "TranscriptionEngine":
        from meeting_assistant.core.transcription import TranscriptionEngine
        return TranscriptionEngine
    if name == "TTSEngine":
        from meeting_assistant.core.tts import TTSEngine
        return TTSEngine
    raise AttributeError(f"module 'meeting_assistant.core' has no attribute {name!r}")
