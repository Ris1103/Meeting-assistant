"""Meeting platform integration adapters.

Each adapter provides:
  - Active meeting detection (process-based)
  - Meeting metadata retrieval (via platform APIs)
  - Real-time chat message streaming
  - Participant list access

Supported platforms:
  - Zoom (ZoomAdapter) — process detection; Meeting SDK for full integration
  - Microsoft Teams (TeamsAdapter) — Microsoft Graph API
  - Google Meet (GoogleMeetAdapter) — Google Calendar + Meet REST API
"""

from meeting_assistant.platforms.base import PlatformAdapter
from meeting_assistant.platforms.google_meet import GoogleMeetAdapter
from meeting_assistant.platforms.teams import TeamsAdapter
from meeting_assistant.platforms.zoom import ZoomAdapter

__all__ = [
    "GoogleMeetAdapter",
    "PlatformAdapter",
    "TeamsAdapter",
    "ZoomAdapter",
]
