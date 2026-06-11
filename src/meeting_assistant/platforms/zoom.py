"""Zoom meeting platform adapter.

MVP scope:
  - Active meeting detection via psutil process list
  - Meeting metadata: parsed from Zoom window title (best-effort)
  - Chat messages: not available in MVP (Zoom requires SDK or webhook)
  - Audio: handled by system loopback capture (ZoomAdapter provides device hint)

Post-MVP path:
  - Zoom Meeting SDK (C++ native) for direct raw A/V streams
  - Zoom REST API webhooks for real-time chat events
  - Zoom OAuth2 PKCE flow for authenticated API access

Auth config required (for REST API, not needed for MVP):
  - ZOOM_CLIENT_ID
  - ZOOM_CLIENT_SECRET
"""

from __future__ import annotations

import logging
import re
from collections.abc import AsyncGenerator
from datetime import datetime

from meeting_assistant.config import Settings
from meeting_assistant.models.schemas import ChatMessage, Platform
from meeting_assistant.platforms.base import PlatformAdapter

logger = logging.getLogger(__name__)

# Zoom process names across platforms
_ZOOM_PROCESS_NAMES = frozenset({
    "zoom",
    "zoom.exe",
    "zoom us",
    "ZoomOpener",
    "CptHost",         # Zoom internal host process (macOS/Linux)
    "zoomus",          # some Linux builds
})

# Regex to extract meeting ID from Zoom window title
# Zoom titles often look like: "Zoom Meeting - 123 456 7890" or "Meeting with John"
_WINDOW_TITLE_MEETING_ID = re.compile(r"\b(\d{9,11})\b")


class ZoomAdapter(PlatformAdapter):
    """Zoom platform adapter (MVP: process detection + audio loopback).

    Audio is captured via the system loopback device (no Zoom SDK needed).
    Chat messages are not available in MVP without the Zoom Meeting SDK.
    """

    PLATFORM_NAME = "zoom"
    PLATFORM_ENUM = Platform.ZOOM

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._meeting_id: str | None = None
        self._http_session = None  # aiohttp session for REST API calls

    # -----------------------------------------------------------------------
    # Detection
    # -----------------------------------------------------------------------

    async def detect_active_meeting(self) -> bool:
        """Detect if Zoom is running with an active meeting."""
        try:
            import psutil  # type: ignore[import]

            for proc in psutil.process_iter(["name", "status"]):
                try:
                    pname = (proc.info.get("name") or "").lower()
                    if any(z in pname for z in ("zoom", "cpthost")):
                        logger.debug("Zoom process detected: %s", pname)
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            logger.warning("psutil not installed. Cannot detect Zoom process.")
        return False

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    async def get_meeting_metadata(self) -> dict:
        """Return best-effort Zoom meeting metadata.

        In MVP, reads what's available from the process/window title.
        With full SDK integration, this would call the Zoom REST API.
        """
        metadata: dict = {
            "meeting_id": self._meeting_id or "zoom-unknown",
            "title": "Zoom Meeting",
            "participants": [],
            "started_at": datetime.utcnow().isoformat(),
            "host": None,
        }

        # Try to extract meeting ID from window title
        window_id = await self._get_meeting_id_from_window()
        if window_id:
            self._meeting_id = window_id
            metadata["meeting_id"] = window_id

        # If authenticated, fetch from REST API
        if self._config.zoom_client_id and self._config.zoom_client_secret:
            try:
                api_metadata = await self._fetch_meeting_from_api()
                metadata.update(api_metadata)
            except Exception as exc:
                logger.debug("Zoom API metadata fetch failed: %s", exc)

        return metadata

    async def _get_meeting_id_from_window(self) -> str | None:
        """Attempt to extract meeting ID from Zoom window title."""
        try:
            import psutil  # type: ignore[import]

            for proc in psutil.process_iter(["name", "cmdline"]):
                try:
                    name = (proc.info.get("name") or "").lower()
                    if "zoom" in name:
                        cmdline = " ".join(proc.info.get("cmdline") or [])
                        m = _WINDOW_TITLE_MEETING_ID.search(cmdline)
                        if m:
                            return m.group(1)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            pass
        return None

    async def _fetch_meeting_from_api(self) -> dict:
        """Fetch meeting details from Zoom REST API (requires OAuth token)."""
        # This would use Zoom's REST API:
        # GET https://api.zoom.us/v2/meetings/{meetingId}
        # Requires OAuth2 access token
        # Placeholder for full API integration
        return {}

    # -----------------------------------------------------------------------
    # Chat
    # -----------------------------------------------------------------------

    async def stream_chat_messages(self) -> AsyncGenerator[ChatMessage, None]:
        """Stream Zoom chat messages.

        MVP limitation: Zoom does not expose meeting chat via a simple REST API.
        Real-time chat requires the Zoom Meeting SDK or webhook integration.
        This generator yields nothing in MVP.

        Post-MVP: Subscribe to Zoom webhook events with:
          - GET /v2/meetings/{meetingId}/chat/messages (polling)
          - Or Zoom Meeting SDK real-time chat callback
        """
        logger.info(
            "Zoom chat streaming not available in MVP. "
            "Chat messages will not be included in meeting context."
        )
        # Empty async generator
        return
        yield  # type: ignore[misc]

    async def send_chat_message(self, text: str) -> bool:
        """Send a chat message via Zoom REST API (requires OAuth).

        POST https://api.zoom.us/v2/meetings/{meetingId}/chat
        """
        if not self._config.zoom_client_id:
            logger.debug("Zoom credentials not configured; cannot send chat message.")
            return False

        # Placeholder for REST API implementation
        logger.debug("Zoom send_chat_message (stub): %r", text[:60])
        return False

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def shutdown(self) -> None:
        if self._http_session:
            await self._http_session.close()
            self._http_session = None
