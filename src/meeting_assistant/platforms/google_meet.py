"""Google Meet platform adapter.

Integration via Google APIs:
  - Active meeting detection via Chrome/browser process with meet.google.com
  - Meeting metadata: Google Calendar API (find current/upcoming events with Meet link)
  - Chat messages: Not available via public API in MVP (Meet Media API is in preview)
  - Audio: Handled by system loopback (no Meet SDK needed)

Authentication:
  - Uses google-auth + google-auth-oauthlib for OAuth2
  - Requires: GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
  - Scopes: https://www.googleapis.com/auth/calendar.readonly
  - Credentials stored in ~/.config/meeting-assistant/google_token.json

Post-MVP:
  - Google Meet Media API (Developer Preview) for direct A/V stream access
  - Meet Add-on SDK for in-meeting UI (iframe-based)
  - Real-time chat via Meet API when publicly available
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from meeting_assistant.config import Settings
from meeting_assistant.models.schemas import ChatMessage, Platform
from meeting_assistant.platforms.base import PlatformAdapter

logger = logging.getLogger(__name__)

CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"
MEET_SPACES_API_BASE = "https://meet.googleapis.com/v2"
TOKEN_SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
]
TOKEN_FILE = Path.home() / ".config" / "meeting-assistant" / "google_token.json"


class GoogleMeetAdapter(PlatformAdapter):
    """Google Meet adapter using Calendar API for meeting discovery.

    Audio capture is handled by the system loopback device. Chat ingestion
    requires the Meet Media API which is currently in Developer Preview.
    """

    PLATFORM_NAME = "google_meet"
    PLATFORM_ENUM = Platform.GOOGLE_MEET

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._credentials = None
        self._http: httpx.AsyncClient | None = None
        self._access_token: str | None = None
        self._token_expiry: datetime = datetime.min.replace(tzinfo=UTC)

    # -----------------------------------------------------------------------
    # Detection
    # -----------------------------------------------------------------------

    async def detect_active_meeting(self) -> bool:
        """Detect if Google Meet is active in a browser."""
        try:
            import psutil  # type: ignore[import]

            browser_names = ("chrome", "chromium", "firefox", "msedge", "brave")
            for proc in psutil.process_iter(["name", "cmdline"]):
                try:
                    name = (proc.info.get("name") or "").lower()
                    if not any(b in name for b in browser_names):
                        continue
                    cmdline = " ".join(proc.info.get("cmdline") or [])
                    if "meet.google.com" in cmdline:
                        logger.debug("Google Meet detected in browser: %s", name)
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            logger.warning("psutil not installed. Cannot detect Google Meet.")
        return False

    # -----------------------------------------------------------------------
    # Authentication
    # -----------------------------------------------------------------------

    async def _ensure_credentials(self) -> str:
        """Return a valid Google OAuth2 access token."""
        now = datetime.now(UTC)
        if self._access_token and now < self._token_expiry - timedelta(minutes=5):
            return self._access_token

        if not self._config.google_client_id:
            raise RuntimeError(
                "Google Meet integration requires GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET. "
                "See .env.example for setup instructions."
            )

        loop = asyncio.get_running_loop()
        token = await loop.run_in_executor(None, self._acquire_google_token)
        self._access_token = token
        self._token_expiry = now + timedelta(hours=1)
        return token

    def _acquire_google_token(self) -> str:
        """Acquire Google OAuth2 token using google-auth-oauthlib (sync)."""
        try:
            import google.auth.transport.requests  # type: ignore[import]
            from google.oauth2.credentials import Credentials  # type: ignore[import]
            from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "google-auth-oauthlib is not installed. "
                "Install with: pip install google-auth-oauthlib google-auth"
            ) from None

        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

        creds = None
        if TOKEN_FILE.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), TOKEN_SCOPES)
            except Exception:
                creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(google.auth.transport.requests.Request())
            else:
                client_config = {
                    "installed": {
                        "client_id": self._config.google_client_id,
                        "client_secret": self._config.google_client_secret,
                        "redirect_uris": [self._config.google_redirect_uri],
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                }
                flow = InstalledAppFlow.from_client_config(client_config, TOKEN_SCOPES)
                creds = flow.run_local_server(port=8080, open_browser=True)

            TOKEN_FILE.write_text(creds.to_json())

        return creds.token

    # -----------------------------------------------------------------------
    # HTTP helpers
    # -----------------------------------------------------------------------

    async def _get_http(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    async def _api_get(self, url: str, params: dict | None = None) -> Any:
        token = await self._ensure_credentials()
        client = await self._get_http()
        response = await client.get(
            url,
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        response.raise_for_status()
        return response.json()

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    async def get_meeting_metadata(self) -> dict:
        """Find the current Google Meet meeting via Google Calendar API."""
        metadata: dict = {
            "meeting_id": "meet-unknown",
            "title": "Google Meet",
            "participants": [],
            "started_at": datetime.utcnow().isoformat(),
            "host": None,
        }

        if not self._config.google_client_id:
            logger.debug("Google credentials not configured; returning stub metadata.")
            return metadata

        try:
            now = datetime.now(UTC)
            time_min = (now - timedelta(minutes=30)).isoformat()
            time_max = (now + timedelta(minutes=30)).isoformat()

            data = await self._api_get(
                f"{CALENDAR_API_BASE}/calendars/primary/events",
                params={
                    "timeMin": time_min,
                    "timeMax": time_max,
                    "singleEvents": "true",
                    "orderBy": "startTime",
                    "q": "meet.google.com",  # filter for Meet events
                },
            )

            events = data.get("items", [])
            # Find events with a Meet conference link
            for event in events:
                conference = event.get("conferenceData", {})
                entry_points = conference.get("entryPoints", [])
                meet_link = next(
                    (ep["uri"] for ep in entry_points if "meet.google.com" in ep.get("uri", "")),
                    None,
                )
                if meet_link:
                    metadata["title"] = event.get("summary", "Google Meet")
                    metadata["meeting_id"] = meet_link
                    metadata["host"] = event.get("organizer", {}).get("displayName")

                    # Extract attendees
                    attendees = event.get("attendees", [])
                    metadata["participants"] = [
                        a.get("displayName") or a.get("email", "Unknown")
                        for a in attendees
                    ]

                    # Started at
                    start = event.get("start", {})
                    metadata["started_at"] = start.get("dateTime", start.get("date"))
                    logger.info("Google Meet event found: %s", metadata["title"])
                    break

        except Exception as exc:
            logger.warning("Failed to fetch Google Calendar metadata: %s", exc)

        return metadata

    # -----------------------------------------------------------------------
    # Chat
    # -----------------------------------------------------------------------

    async def stream_chat_messages(self) -> AsyncGenerator[ChatMessage, None]:
        """Stream Google Meet chat messages.

        MVP limitation: Google Meet chat is not publicly accessible via the
        standard REST API. The Meet Media API (Developer Preview) may support
        this in future, and a Chrome Extension could expose chat events.

        This generator yields nothing in MVP.
        """
        logger.info(
            "Google Meet chat streaming not available in MVP. "
            "Chat messages will not be included in meeting context. "
            "The Meet Media API (Developer Preview) may provide this in future."
        )
        return
        yield  # type: ignore[misc]

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def shutdown(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()
            self._http = None
