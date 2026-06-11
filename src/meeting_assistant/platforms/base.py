"""Abstract base class for meeting platform adapters.

All platform adapters must implement this interface. The audio capture is
handled separately by the unified virtual loopback layer (AudioCaptureManager),
so these adapters focus exclusively on:
  1. Detecting when their platform has an active meeting
  2. Fetching meeting metadata (title, participants, ID)
  3. Streaming chat messages in real time
  4. Posting responses to meeting chat
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from meeting_assistant.models.schemas import ChatMessage, Platform

logger = logging.getLogger(__name__)


class PlatformAdapter(ABC):
    """Abstract interface for meeting platform integrations.

    Concrete implementations should handle authentication lazily (on first API
    call) to avoid blocking startup.

    Usage::

        adapter = ZoomAdapter(config)
        if await adapter.detect_active_meeting():
            metadata = await adapter.get_meeting_metadata()
            async for msg in adapter.stream_chat_messages():
                await context.add_chat_message(msg)
    """

    PLATFORM_NAME: str = ""
    PLATFORM_ENUM: Platform = Platform.GENERIC

    # -----------------------------------------------------------------------
    # Detection
    # -----------------------------------------------------------------------

    @abstractmethod
    async def detect_active_meeting(self) -> bool:
        """Return True if this platform currently has an active meeting.

        Implementation strategies:
          - Check for platform process in the process list (psutil)
          - Check platform API for ongoing meeting
          - Detect platform browser tab (for web-based platforms)
        """

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    @abstractmethod
    async def get_meeting_metadata(self) -> dict:
        """Return meeting metadata as a dict.

        Returned dict should contain (where available):
          - meeting_id (str): Platform-specific meeting identifier
          - title (str): Meeting title
          - participants (list[str]): Display names of participants
          - started_at (str | None): ISO timestamp of meeting start
          - host (str | None): Host display name
        """

    async def get_participant_list(self) -> list[str]:
        """Return current participant display names.

        Default implementation calls get_meeting_metadata(); override for
        more efficient real-time participant tracking.
        """
        metadata = await self.get_meeting_metadata()
        return metadata.get("participants", [])

    # -----------------------------------------------------------------------
    # Chat
    # -----------------------------------------------------------------------

    @abstractmethod
    async def stream_chat_messages(self) -> AsyncGenerator[ChatMessage, None]:
        """Async generator that yields incoming chat messages.

        Implementations should:
          - Yield messages as they arrive (polling or webhook-based)
          - Handle reconnection on transient errors
          - Raise StopAsyncIteration cleanly on meeting end
          - Not yield messages seen before the generator was started (dedup)

        This generator runs indefinitely until cancelled.
        """

    async def send_chat_message(self, text: str) -> bool:
        """Post a text message to the meeting chat.

        Args:
            text: Message to send.

        Returns:
            True if the message was sent successfully, False otherwise.

        Note: Not all platforms support programmatic chat posting without
        a bot integration. Default implementation returns False.
        """
        logger.debug(
            "%s: send_chat_message not implemented; dropping: %r",
            self.PLATFORM_NAME,
            text[:60],
        )
        return False

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Release API sessions, WebSocket connections, and other resources.

        Override in concrete adapters that hold open HTTP sessions or
        WebSocket connections.
        """


# ---------------------------------------------------------------------------
# Null adapter (used when no platform is detected)
# ---------------------------------------------------------------------------


class NullPlatformAdapter(PlatformAdapter):
    """No-op adapter used when no meeting platform is detected.

    Returns empty metadata and an empty async generator for chat messages.
    Useful for testing the pipeline without a live meeting.
    """

    PLATFORM_NAME = "null"
    PLATFORM_ENUM = Platform.GENERIC

    async def detect_active_meeting(self) -> bool:
        return False

    async def get_meeting_metadata(self) -> dict:
        return {
            "meeting_id": "offline-session",
            "title": "Offline Session",
            "participants": [],
            "started_at": None,
            "host": None,
        }

    async def stream_chat_messages(self) -> AsyncGenerator[ChatMessage, None]:
        # Never yields — an empty infinite async generator
        return
        yield  # type: ignore[misc]  # make this an async generator

    async def send_chat_message(self, text: str) -> bool:
        logger.info("NullAdapter: would send chat message: %r", text[:60])
        return True
