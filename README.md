# Meeting Assistant

A cross-platform AI meeting assistant that listens to Zoom, Microsoft Teams, and Google Meet sessions, transcribes speech in real time, and answers questions using Anthropic Claude — all while keeping the assistant UI invisible to other participants.

```
Audio (loopback + mic) → Whisper STT → Context Tracker → Claude LLM → TTS
                                 ↑
           Platform APIs (Teams chat, Zoom, Meet Calendar) ──────────────┘
```

## Features

- **Real-time transcription** using [faster-whisper](https://github.com/guillaumekynast/faster-whisper) (runs locally, free, no data leaves your machine)
- **Speaker diarization** via [pyannote.audio](https://github.com/pyannote/pyannote-audio) (optional, requires HuggingFace token)
- **Cross-platform audio capture** — works with Zoom, Teams, and Google Meet via system loopback (no meeting SDK required)
- **AI responses** powered by [Anthropic Claude](https://www.anthropic.com) with streaming text-to-speech
- **Platform integrations**: Microsoft Graph API (Teams chat), Google Calendar API (Meet metadata)
- **Privacy by design**: audio processed locally, only text sent to LLM, no persistent storage by default

## Supported Platforms

| Platform | Audio | Chat | Metadata |
|---|---|---|---|
| Zoom | ✅ Loopback | 🔜 (SDK required) | ✅ Process detection |
| Microsoft Teams | ✅ Loopback | ✅ Graph API | ✅ Calendar API |
| Google Meet | ✅ Loopback | 🔜 (API preview) | ✅ Calendar API |

## Requirements

- Python 3.11+
- Anthropic API key ([get one here](https://console.anthropic.com/))
- For diarization: HuggingFace token + accept [pyannote model terms](https://huggingface.co/pyannote/speaker-diarization-3.1)
- For loopback audio capture: see platform-specific setup below

## Quick Start

```bash
# Clone and install
git clone https://github.com/ris1103/meeting-assistant
cd meeting-assistant
pip install -e .

# Configure
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Check audio devices
meeting-assistant devices

# Start (auto-detects running meeting platform)
meeting-assistant start --auto-detect
```

## Audio Loopback Setup

The assistant captures meeting audio via the system speaker output (loopback). Setup varies by OS:

### Linux (PulseAudio / PipeWire)

Loopback is auto-detected. No setup needed if PulseAudio or PipeWire is running:

```bash
# Verify monitor source exists
pactl list short sources | grep monitor
```

### macOS

Install [BlackHole](https://github.com/ExistentialAudio/BlackHole) virtual audio device:

1. Install BlackHole 2ch: `brew install --cask blackhole-2ch`
2. Open **Audio MIDI Setup** → create a **Multi-Output Device** with your speakers + BlackHole
3. Set this Multi-Output Device as your system output
4. Add to `.env`: `LOOPBACK_DEVICE_NAME=BlackHole 2ch`

### Windows

WASAPI loopback is auto-detected via [pyaudiowpatch](https://github.com/s0d3s/PyAudioWPatch):

```bash
pip install pyaudiowpatch
```

No additional setup needed — your system's default speaker output is automatically captured.

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# Required
ANTHROPIC_API_KEY=sk-ant-...

# Optional: speaker diarization (identifies who said what)
HUGGINGFACE_TOKEN=hf_...
WHISPER_MODEL_SIZE=base    # tiny | base | small | medium | large-v3

# Platform integrations (optional — enables chat + metadata)
TEAMS_CLIENT_ID=...
TEAMS_TENANT_ID=...
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
ZOOM_CLIENT_ID=...
ZOOM_CLIENT_SECRET=...
```

See [`.env.example`](.env.example) for the complete reference.

## Commands

```bash
# Start assistant (auto-detect platform)
meeting-assistant start --auto-detect

# Start for a specific platform
meeting-assistant start --platform zoom
meeting-assistant start --platform teams
meeting-assistant start --platform google_meet

# Diagnostics
meeting-assistant devices          # list audio devices
meeting-assistant test-audio       # test loopback capture (10s)
meeting-assistant test-stt         # test Whisper with mic recording
meeting-assistant test-llm         # test Claude API connection

# OAuth setup wizard
meeting-assistant setup

# Offline file transcription
meeting-assistant transcribe meeting.wav
meeting-assistant transcribe meeting.wav --output transcript.txt
```

## Usage During a Meeting

1. Join your meeting in Zoom/Teams/Meet as normal
2. In a terminal: `meeting-assistant start --auto-detect`
3. Grant the consent prompt (confirms you have participant consent to record)
4. The assistant listens in the background
5. Say **"assistant"** (or your configured `TRIGGER_KEYWORD`) to trigger a response
6. The response is spoken through your local speakers (only you hear it)
7. Press `Ctrl+C` to stop — a meeting summary is generated automatically

## Privacy

- **Audio never leaves your machine** — Whisper STT runs locally
- **Only transcript text is sent to Claude** via HTTPS
- **No persistent storage** by default (`NO_STORE=true`)
- Consent is required at startup
- OAuth tokens stored in OS keyring, not in `.env`

## Platform Integration Setup

### Microsoft Teams

1. Register an app at [Azure Portal](https://portal.azure.com/#view/Microsoft_AAD_RegisteredApps)
2. Add delegated permissions: `Chat.Read`, `OnlineMeetings.Read`, `Calendars.Read`
3. Set redirect URI: `http://localhost:8080/callback`
4. Add `TEAMS_CLIENT_ID` and `TEAMS_TENANT_ID` to `.env`

### Google Meet / Calendar

1. Create OAuth 2.0 credentials at [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Enable the Google Calendar API and Meet API
3. Add `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` to `.env`
4. First run will open a browser for OAuth authorization

### Zoom

1. Create an OAuth app at [Zoom Marketplace](https://marketplace.zoom.us/develop/create)
2. Add scopes: `meeting:read`, `chat_message:read`
3. Add `ZOOM_CLIENT_ID` and `ZOOM_CLIENT_SECRET` to `.env`

## Architecture

```
src/meeting_assistant/
├── core/
│   ├── audio_capture.py    # Loopback + mic capture (sounddevice)
│   ├── transcription.py    # Whisper STT + pyannote diarization
│   ├── context_tracker.py  # Rolling transcript + chat context
│   ├── llm_client.py       # Anthropic Claude API (streaming)
│   ├── tts.py             # edge-tts + pyttsx3 fallback
│   └── pipeline.py        # Main async orchestrator
├── platforms/
│   ├── base.py            # Abstract PlatformAdapter
│   ├── zoom.py            # Zoom adapter
│   ├── teams.py           # Teams adapter (Graph API)
│   └── google_meet.py     # Google Meet adapter (Calendar API)
├── models/
│   └── schemas.py         # Pydantic data models
├── ui/
│   └── cli.py             # Typer CLI
└── config.py              # pydantic-settings configuration
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests (no API keys required)
pytest tests/ -v

# Run tests excluding slow model loading
pytest tests/ -m "not slow" -v

# Lint and format
ruff check src/ tests/
ruff format src/ tests/
```

## Roadmap

- [ ] Hotword detection ("Hey Assistant") using Porcupine
- [ ] Virtual microphone injection (assistant speaks into meeting)
- [ ] Zoom Meeting SDK integration for direct A/V streams
- [ ] Google Meet Media API integration (currently in preview)
- [ ] Multi-turn conversation memory across meetings
- [ ] Browser extension for web-based meeting capture
- [ ] Overlay UI (transparent, excluded from screen share)
- [ ] Mobile support (iOS/Android)

## License

MIT License — see [LICENSE](LICENSE) for details.
