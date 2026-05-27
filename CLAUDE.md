# Meeting Assistant — CLAUDE.md

This file documents the codebase for AI coding assistants.

## Project Overview

Cross-platform AI meeting assistant for Zoom, Microsoft Teams, and Google Meet.
Captures audio via system loopback, transcribes with Whisper, and generates responses using Claude API.

**Language**: Python 3.11+
**Entry point**: `src/meeting_assistant/ui/cli.py` → `meeting-assistant` CLI command

## Architecture

```
Audio Capture → Transcription → Context Tracker → LLM → TTS
     ↑                                   ↑
Platform Adapters (chat/metadata) ────────┘
```

### Key Files

| File | Purpose |
|---|---|
| `src/meeting_assistant/core/pipeline.py` | Main orchestrator — wires all components together |
| `src/meeting_assistant/core/audio_capture.py` | Cross-platform loopback + mic capture |
| `src/meeting_assistant/core/transcription.py` | Whisper STT + pyannote diarization |
| `src/meeting_assistant/core/context_tracker.py` | Rolling meeting context state |
| `src/meeting_assistant/core/llm_client.py` | Anthropic Claude API with streaming |
| `src/meeting_assistant/core/tts.py` | edge-tts + pyttsx3 TTS |
| `src/meeting_assistant/platforms/base.py` | Abstract `PlatformAdapter` interface |
| `src/meeting_assistant/platforms/zoom.py` | Zoom integration |
| `src/meeting_assistant/platforms/teams.py` | Teams (Microsoft Graph API) |
| `src/meeting_assistant/platforms/google_meet.py` | Google Meet (Calendar API) |
| `src/meeting_assistant/models/schemas.py` | Pydantic data models |
| `src/meeting_assistant/config.py` | pydantic-settings config (loads from .env) |

## Data Models

```
AudioChunk       → raw PCM bytes from capture
SpeakerSegment   → transcribed text attributed to a speaker
TranscriptionResult → list of SpeakerSegments from one STT pass
ChatMessage      → chat message from a platform
MeetingContext   → full snapshot of meeting state (transcript + chat)
```

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run without slow model tests  
pytest tests/ -m "not slow" -v

# Lint
ruff check src/ tests/

# Start assistant
meeting-assistant start --auto-detect
meeting-assistant start --platform zoom
meeting-assistant devices
meeting-assistant test-stt
```

## Adding a New Platform

1. Create `src/meeting_assistant/platforms/<name>.py`
2. Subclass `PlatformAdapter` from `platforms/base.py`
3. Implement `detect_active_meeting()`, `get_meeting_metadata()`, `stream_chat_messages()`
4. Set `PLATFORM_NAME` and `PLATFORM_ENUM` class attributes
5. Register in `platforms/__init__.py`
6. Add to the platform mapping in `ui/cli.py::_get_adapter()`
7. Write tests in `tests/test_platforms/test_<name>.py`

## Environment Variables

Required:
- `ANTHROPIC_API_KEY` — Anthropic Claude API key

Optional:
- `HUGGINGFACE_TOKEN` — enables speaker diarization
- `WHISPER_MODEL_SIZE` — tiny|base|small|medium|large-v3 (default: base)
- `TRIGGER_KEYWORD` — hotword to trigger responses (default: assistant)
- `LOOPBACK_DEVICE_NAME` — audio device name for meeting capture

See `.env.example` for the full list.

## Privacy Constraints

- Audio MUST NOT leave the local machine — Whisper runs locally
- Only transcript TEXT is sent to the Claude API
- Consent must be confirmed before audio capture starts (`_check_consent()` in `pipeline.py`)
- No audio files are persisted by default
- When adding features: follow the same data minimisation principles

## Testing Patterns

- Use `test_settings` fixture from `conftest.py` — provides safe test settings
- Mock `faster_whisper.WhisperModel` and `psutil.process_iter` in unit tests
- Platform adapter tests use `patch("psutil.process_iter", ...)` for process detection
- Async tests use `pytest-asyncio` with `asyncio_mode = "auto"` (configured in `pyproject.toml`)
- Integration tests marked with `@pytest.mark.integration` are skipped without real API keys
