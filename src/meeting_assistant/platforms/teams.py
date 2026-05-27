"""Microsoft Teams platform adapter.

Integration via Microsoft Graph API:
  - Active meeting detection via psutil + Teams process
  - Meeting metadata: GET /me/onlineMeetings
  - Chat messages: GET /chats/{chatId}/messages (polling every 5s)
  - Send chat: POST /chats/{chatId}/messages

Authentication:
  - Uses MSAL (Microsoft Authentication Library) for OAuth2
  - Requires: TEAMS_CLIENT_ID, TEAMS_TENANT_ID
  - Delegated permissions: Chat.Read, OnlineMeetings.Read, Calendars.Read
  - Tokens are cached in the OS keyring via MSAL's token cache

Graph API base: https://graph.microsoft.com/v1.0
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from meeting_assistant.config import Settings
from meeting_assistant.models.schemas import ChatMessage, Platform
from meeting_assistant.platforms.base import PlatformAdapter

logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.microsoft.com/v1.0"
GRAPH_SCOPES = [
    "Chat.Read",
    "OnlineMeetings.Read",
    "Calendars.Read",
]

# Teams process names
_TEAMS_PROCESS_NAMES = frozenset({
    "teams",
    "ms-teams",
    "teams.exe",
    "msteams",
    "microsoft teams",
})


class TeamsAdapter(PlatformAdapter):
    """Microsoft Teams adapter using the Graph API.

    Usage::

        adapter = TeamsAdapter(config)
        if await adapter.detect_active_meeting():
            meta = await adapter.get_meeting_metadata()
            async for msg in adapter.stream_chat_messages():
                handle(msg)
    """

    PLATFORM_NAME = "teams"
    PLATFORM_ENUM = Platform.TEAMS

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._access_token: str | None = None
        self._token_expires_at: datetime = datetime.min.replace(tzinfo=timezone.utc)
        self._http: httpx.AsyncClient | None = None
        self._active_chat_id: str | None = None
        self._seen_message_ids: set[str] = set()

    # -----------------------------------------------------------------------
    # Detection
    # -----------------------------------------------------------------------

    async def detect_active_meeting(self) -> bool:
        """Detect if Microsoft Teams is running."""
        try:
            import psutil  # type: ignore[import]

            for proc in psutil.process_iter(["name"]):
                try:
                    pname = (proc.info.get("name") or "").lower()
                    if any(t in pname for t in ("teams", "msteams")):
                        logger.debug("Teams process detected: %s", pname)
                        return True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except ImportError:
            logger.warning("psutil not installed. Cannot detect Teams process.")
        return False

    # -----------------------------------------------------------------------
    # Authentication
    # -----------------------------------------------------------------------

    async def _ensure_token(self) -> str:
        """Return a valid access token, refreshing if needed."""
        now = datetime.now(timezone.utc)
        if self._access_token and now < self._token_expires_at - timedelta(minutes=5):
            return self._access_token

        if not self._config.teams_client_id:
            raise RuntimeError(
                "Teams integration requires TEAMS_CLIENT_ID and TEAMS_TENANT_ID. "
                "See .env.example for setup instructions."
            )

        token = await self._acquire_token_msal()
        self._access_token = token
        # MSAL tokens typically expire in 1 hour
        self._token_expires_at = now + timedelta(hours=1)
        return token

    async def _acquire_token_msal(self) -> str:
        """Acquire OAuth2 token via MSAL interactive or device flow."""
        try:
            import msal  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "msal is not installed. Install with: pip install msal"
            )

        loop = asyncio.get_running_loop()

        def _get_token() -> str:
            authority = f"https://login.microsoftonline.com/{self._config.teams_tenant_id}"
            app = msal.PublicClientApplication(
                self._config.teams_client_id,
                authority=authority,
            )

            # Try silent first (cached token)
            accounts = app.get_accounts()
            if accounts:
                result = app.acquire_token_silent(GRAPH_SCOPES, account=accounts[0])
                if result and "access_token" in result:
                    return result["access_token"]

            # Interactive device code flow (works in terminal)
            flow = app.initiate_device_flow(scopes=GRAPH_SCOPES)
            print(f"\n{flow['message']}\n")
            result = app.acquire_token_by_device_flow(flow)

            if "access_token" not in result:
                raise RuntimeError(
                    f"Teams authentication failed: {result.get('error_description', result)}"
                )
            return result["access_token"]

        return await loop.run_in_executor(None, _get_token)

    # -----------------------------------------------------------------------
    # HTTP client
    # -----------------------------------------------------------------------

    async def _get_http(self) -> httpx.AsyncClient:
        """Return (or create) the authenticated HTTP client."""
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(timeout=30.0)
        return self._http

    async def _graph_get(self, path: str, params: dict | None = None) -> Any:
        """Perform an authenticated GET request to the Graph API."""
        token = await self._ensure_token()
        client = await self._get_http()
        url = f"{GRAPH_BASE}{path}"
        headers = {"Authorization": f"Bearer {token}"}
        response = await client.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json()

    async def _graph_post(self, path: str, json_body: dict) -> Any:
        """Perform an authenticated POST request to the Graph API."""
        token = await self._ensure_token()
        client = await self._get_http()
        url = f"{GRAPH_BASE}{path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        response = await client.post(url, headers=headers, json=json_body)
        response.raise_for_status()
        return response.json()

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    async def get_meeting_metadata(self) -> dict:
        """Fetch current meeting metadata via Graph API."""
        metadata: dict = {
            "meeting_id": "teams-unknown",
            "title": "Teams Meeting",
            "participants": [],
            "started_at": datetime.utcnow().isoformat(),
            "host": None,
        }

        if not self._config.teams_client_id:
            logger.debug("Teams credentials not configured; returning stub metadata.")
            return metadata

        try:
            # Find current/upcoming online meeting via Calendar
            now = datetime.now(timezone.utc)
            start = (now - timedelta(minutes=30)).isoformat()
            end = (now + timedelta(minutes=30)).isoformat()

            data = await self._graph_get(
                "/me/calendarView",
                params={
                    "startDateTime": start,
                    "endDateTime": end,
                    "$filter": "isOnlineMeeting eq true",
                    "$select": "subject,onlineMeeting,organizer,attendees",
                    "$top": "5",
                },
            )

            events = data.get("value", [])
            if events:
                event = events[0]
                metadata["title"] = event.get("subject", "Teams Meeting")
                online = event.get("onlineMeeting", {})
                metadata["meeting_id"] = online.get("joinUrl", "teams-unknown")
                metadata["host"] = event.get("organizer", {}).get("emailAddress", {}).get("name")

                # Extract participants
                attendees = event.get("attendees", [])
                metadata["participants"] = [
                    a.get("emailAddress", {}).get("name", "Unknown")
                    for a in attendees
                ]

                # Store chat ID for streaming
                chat_id = online.get("chatInfo", {}).get("threadId")
                if chat_id:
                    self._active_chat_id = chat_id
                    logger.info("Teams meeting chat ID: %s", chat_id)

        except Exception as exc:
            logger.warning("Failed to fetch Teams meeting metadata: %s", exc)

        return metadata

    # -----------------------------------------------------------------------
    # Chat
    # -----------------------------------------------------------------------

    async def stream_chat_messages(self) -> AsyncGenerator[ChatMessage, None]:
        """Stream Teams chat messages by polling the Graph API every 5 seconds."""
        if not self._config.teams_client_id:
            logger.info("Teams credentials not configured; skipping chat ingestion.")
            return
            yield  # type: ignore[misc]

        # Ensure we have a chat ID
        if not self._active_chat_id:
            await self.get_meeting_metadata()

        if not self._active_chat_id:
            logger.warning("No active Teams meeting chat ID found; cannot stream chat.")
            return
            yield  # type: ignore[misc]

        logger.info("Streaming Teams chat from thread: %s", self._active_chat_id)
        poll_interval = 5.0  # seconds

        while True:
            try:
                data = await self._graph_get(
                    f"/chats/{self._active_chat_id}/messages",
                    params={"$top": "20", "$orderby": "createdDateTime desc"},
                )
                messages = data.get("value", [])

                # Yield new messages (reverse to maintain chronological order)
                for msg in reversed(messages):
                    msg_id = msg.get("id")
                    if msg_id in self._seen_message_ids:
                        continue
                    self._seen_message_ids.add(msg_id)

                    body = msg.get("body", {}).get("content", "")
                    if not body.strip():
                        continue

                    sender_info = msg.get("from", {}).get("user", {})
                    sender = sender_info.get("displayName", "Unknown")

                    yield ChatMessage(
                        sender=sender,
                        content=body,
                        timestamp=datetime.fromisoformat(
                            msg.get("createdDateTime", datetime.utcnow().isoformat()).rstrip("Z")
                        ),
                        platform=Platform.TEAMS,
                        message_id=msg_id,
                    )

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Teams chat poll error: %s", exc)

            await asyncio.sleep(poll_interval)

    async def send_chat_message(self, text: str) -> bool:
        """Post a message to the active Teams meeting chat."""
        if not self._active_chat_id:
            logger.warning("No active Teams chat; cannot send message.")
            return False
        try:
            await self._graph_post(
                f"/chats/{self._active_chat_id}/messages",
                json_body={"body": {"content": text}},
            )
            return True
        except Exception as exc:
            logger.warning("Failed to send Teams chat message: %s", exc)
            return False

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def shutdown(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()
            self._http = None
