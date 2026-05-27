"""Tests for the Google Meet platform adapter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meeting_assistant.config import Settings
from meeting_assistant.platforms.google_meet import GoogleMeetAdapter


class TestGoogleMeetDetection:
    """Tests for Google Meet detection via browser process."""

    @pytest.fixture
    def adapter(self, test_settings: Settings) -> GoogleMeetAdapter:
        return GoogleMeetAdapter(test_settings)

    @pytest.mark.asyncio
    async def test_detect_meet_in_chrome(self, adapter: GoogleMeetAdapter) -> None:
        """Should detect Google Meet when Chrome is running with meet.google.com."""
        mock_proc = MagicMock()
        mock_proc.info = {
            "name": "chrome",
            "cmdline": ["chrome", "--url=https://meet.google.com/abc-defg-hij"],
        }

        with patch("psutil.process_iter", return_value=[mock_proc]):
            result = await adapter.detect_active_meeting()

        assert result is True

    @pytest.mark.asyncio
    async def test_detect_meet_chromium(self, adapter: GoogleMeetAdapter) -> None:
        """Should detect Google Meet in Chromium."""
        mock_proc = MagicMock()
        mock_proc.info = {
            "name": "chromium",
            "cmdline": ["chromium", "meet.google.com/xyz-abc-123"],
        }

        with patch("psutil.process_iter", return_value=[mock_proc]):
            result = await adapter.detect_active_meeting()

        assert result is True

    @pytest.mark.asyncio
    async def test_no_meet_detected(self, adapter: GoogleMeetAdapter) -> None:
        """Should return False when no browser has meet.google.com open."""
        mock_proc = MagicMock()
        mock_proc.info = {
            "name": "chrome",
            "cmdline": ["chrome", "--url=https://www.google.com"],
        }

        with patch("psutil.process_iter", return_value=[mock_proc]):
            result = await adapter.detect_active_meeting()

        assert result is False

    @pytest.mark.asyncio
    async def test_detect_meet_firefox(self, adapter: GoogleMeetAdapter) -> None:
        """Should detect Google Meet in Firefox."""
        mock_proc = MagicMock()
        mock_proc.info = {
            "name": "firefox",
            "cmdline": ["firefox", "meet.google.com/abc-123"],
        }

        with patch("psutil.process_iter", return_value=[mock_proc]):
            result = await adapter.detect_active_meeting()

        assert result is True


class TestGoogleMeetMetadata:
    """Tests for Google Meet metadata retrieval via Calendar API."""

    @pytest.fixture
    def adapter(self, test_settings: Settings) -> GoogleMeetAdapter:
        return GoogleMeetAdapter(test_settings)

    @pytest.mark.asyncio
    async def test_get_meeting_metadata_without_credentials(
        self, adapter: GoogleMeetAdapter
    ) -> None:
        """Without credentials, returns stub metadata."""
        metadata = await adapter.get_meeting_metadata()

        assert isinstance(metadata, dict)
        assert "meeting_id" in metadata
        assert "title" in metadata
        assert "participants" in metadata

    @pytest.mark.asyncio
    async def test_get_meeting_metadata_with_calendar_event(
        self, test_settings: Settings
    ) -> None:
        """Should extract meeting details from Google Calendar API response."""
        settings = Settings(
            anthropic_api_key="test",
            google_client_id="fake-client-id",
            google_client_secret="fake-secret",
        )
        adapter = GoogleMeetAdapter(settings)
        adapter._access_token = "fake-token"

        from datetime import datetime, timezone, timedelta
        adapter._token_expiry = datetime.now(timezone.utc) + timedelta(hours=1)

        mock_event = {
            "summary": "Product Review",
            "organizer": {"displayName": "Alice"},
            "attendees": [
                {"displayName": "Bob", "email": "bob@example.com"},
                {"email": "carol@example.com"},  # no displayName
            ],
            "start": {"dateTime": "2024-01-01T10:00:00Z"},
            "conferenceData": {
                "entryPoints": [
                    {"uri": "https://meet.google.com/abc-defg-hij", "entryPointType": "video"}
                ]
            },
        }

        with patch.object(
            adapter,
            "_api_get",
            return_value={"items": [mock_event]},
        ):
            metadata = await adapter.get_meeting_metadata()

        assert metadata["title"] == "Product Review"
        assert "meet.google.com" in metadata["meeting_id"]
        assert "Bob" in metadata["participants"]
        assert metadata["host"] == "Alice"


class TestGoogleMeetChat:
    """Tests for Google Meet chat (MVP stub)."""

    @pytest.fixture
    def adapter(self, test_settings: Settings) -> GoogleMeetAdapter:
        return GoogleMeetAdapter(test_settings)

    @pytest.mark.asyncio
    async def test_stream_chat_yields_nothing_mvp(self, adapter: GoogleMeetAdapter) -> None:
        """In MVP, chat streaming is not supported — yields nothing."""
        messages = []
        async for msg in adapter.stream_chat_messages():
            messages.append(msg)
        assert messages == []

    @pytest.mark.asyncio
    async def test_shutdown_cleans_up(self, adapter: GoogleMeetAdapter) -> None:
        """shutdown() should close HTTP session cleanly."""
        import httpx
        adapter._http = httpx.AsyncClient()
        await adapter.shutdown()
        assert adapter._http is None
