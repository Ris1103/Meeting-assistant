"""Application configuration via pydantic-settings.

All values are loaded from environment variables (with .env file support).
See .env.example for the full list of available settings.
"""

from __future__ import annotations

import platform

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated application settings.

    Values are loaded in this precedence order:
    1. Environment variables (highest priority)
    2. .env file
    3. Field defaults (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
    )

    # -----------------------------------------------------------------------
    # LLM (Anthropic Claude)
    # -----------------------------------------------------------------------
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-4-5"
    max_response_tokens: int = 1024

    # -----------------------------------------------------------------------
    # Speech Recognition (Whisper via whisperX / faster-whisper)
    # -----------------------------------------------------------------------
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"  # cpu | cuda | mps
    whisper_compute_type: str = "int8"  # int8 | float16 | float32

    # HuggingFace token — required for pyannote diarization models
    huggingface_token: str = ""

    # -----------------------------------------------------------------------
    # Platform integrations
    # -----------------------------------------------------------------------
    # Zoom
    zoom_client_id: str = ""
    zoom_client_secret: str = ""
    zoom_redirect_uri: str = "http://localhost:8080/callback"

    # Microsoft Teams (Graph API)
    teams_client_id: str = ""
    teams_tenant_id: str = "common"
    teams_client_secret: str = ""

    # Google Meet / Calendar API
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8080/callback"

    # -----------------------------------------------------------------------
    # Audio capture
    # -----------------------------------------------------------------------
    loopback_device_name: str = ""  # auto-detect when empty
    mic_device_index: int | None = None
    audio_sample_rate: int = 16_000  # Hz — Whisper optimal
    vad_frame_duration_ms: int = 30  # must be 10, 20, or 30
    vad_aggressiveness: int = 2  # 0–3
    transcription_buffer_seconds: float = 5.0

    @field_validator("vad_frame_duration_ms")
    @classmethod
    def valid_frame_duration(cls, v: int) -> int:
        if v not in (10, 20, 30):
            raise ValueError("vad_frame_duration_ms must be 10, 20, or 30")
        return v

    @field_validator("vad_aggressiveness")
    @classmethod
    def valid_aggressiveness(cls, v: int) -> int:
        if not (0 <= v <= 3):
            raise ValueError("vad_aggressiveness must be 0–3")
        return v

    # -----------------------------------------------------------------------
    # Pipeline behaviour
    # -----------------------------------------------------------------------
    trigger_keyword: str = "assistant"
    language: str = "en"
    max_speakers: int = 8
    response_mode: str = "local_playback"  # local_playback | chat_message
    tts_voice: str = "en-US-JennyNeural"

    # Context window sizes
    transcript_window_size: int = 50  # max SpeakerSegments kept in memory
    chat_window_size: int = 30
    max_history_turns: int = 10  # LLM conversation history turns

    # -----------------------------------------------------------------------
    # Privacy & storage
    # -----------------------------------------------------------------------
    no_store: bool = False  # if True, disable all transcript persistence
    transcript_dir: str = "./transcripts"
    enable_pii_redaction: bool = True

    # -----------------------------------------------------------------------
    # Derived / computed properties
    # -----------------------------------------------------------------------
    @property
    def os_name(self) -> str:
        """Normalised OS name: 'windows', 'macos', or 'linux'."""
        system = platform.system().lower()
        if system == "darwin":
            return "macos"
        return system  # 'windows' or 'linux'

    @property
    def has_anthropic_key(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_hf_token(self) -> bool:
        return bool(self.huggingface_token)

    @property
    def diarization_enabled(self) -> bool:
        """Diarization requires a HuggingFace token for pyannote models."""
        return self.has_hf_token

    @property
    def samples_per_frame(self) -> int:
        """Number of PCM samples per VAD frame."""
        return int(self.audio_sample_rate * self.vad_frame_duration_ms / 1000)


# Singleton — import and use `settings` everywhere
settings = Settings()
