"""Tests for the Microsoft Teams platform adapter."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from meeting_assistant.config import Settings
from meeting_assistant.models.schemas import Platform
from meeting_assistant.platforms.teams import TeamsAdapter


class TestTeamsAdapterDetection:
    """Tests for Teams process detection."""

    @pytest.fixture
    def adapter(self, test_settings: Settings) -> TeamsAdapter:
        return TeamsAdapter(test_settings)

    @pytest.mark.asyncio
    async def test_detect_active_meeting_teams_running(self, adapter: TeamsAdapter) -> None:
        """Should return True when Teams process is running."""
        mock_proc = MagicMock()
        mock_proc.info = {"name": "teams.exe"}

        with patch("psutil.process_iter", return_value=[mock_proc]):
            result = await adapter.detect_active_meeting()

        assert result is True

    @pytest.mark.asyncio
    async def test_detect_active_meeting_msteams(self, adapter: TeamsAdapter) -> None:
        """Should detect Teams via 'msteams' process name."""
        mock_proc = MagicMock()
        mock_proc.info = {"name": "msteams"}

        with patch("psutil.process_iter", return_value=[mock_proc]):
            result = await adapter.detect_active_meeting()

        assert result is True

    @pytest.mark.asyncio
    async def test_detect_active_meeting_no_teams(self, adapter: TeamsAdapter) -> None:
        """Returns False when Teams is not running."""
        mock_proc = MagicMock()
        mock_proc.info = {"name": "slack"}

        with patch("psutil.process_iter", return_value=[mock_proc]):
            result = await adapter.detect_active_meeting()

        assert result is False


class TestTeamsAdapterMetadata:
    """Tests for Teams meeting metadata retrieval."""

    @pytest.fixture
    def adapter(self, test_settings: Settings) -> TeamsAdapter:
        return TeamsAdapter(test_settings)

    @pytest.mark.asyncio
    async def test_get_meeting_metadata_without_credentials(
        self, adapter: TeamsAdapter
    ) -> None:
        """Without credentials, should return stub metadata."""
        metadata = await adapter.get_meeting_metadata()

        assert isinstance(metadata, dict)
        assert "meeting_id" in metadata
        assert "title" in metadata
        assert "participants" in metadata

    @pytest.mark.asyncio
    async def test_get_meeting_metadata_with_calendar_api(
        self, test_settings: Settings
    ) -> None:
        """With credentials, fetches meeting from Calendar API."""
        settings = Settings(
            anthropic_api_key="test",
            teams_client_id="fake-client-id",
            teams_tenant_id="fake-tenant",
        )
        adapter = TeamsAdapter(settings)
        adapter._access_token = "fake-token"
        adapter._token_expires_at = datetime.now(UTC).replace(
            year=datetime.now(UTC).year + 1
        )

        mock_event = {
            "subject": "Sprint Planning",
            "onlineMeeting": {
                "joinUrl": "https://teams.microsoft.com/l/meetup-join/fake",
                "chatInfo": {"threadId": "19:thread-id@thread.v2"},
            },
            "organizer": {"emailAddress": {"name": "Alice"}},
            "attendees": [
                {"emailAddress": {"name": "Bob"}},
                {"emailAddress": {"name": "Carol"}},
            ],
        }

        mock_response = AsyncMock()
        mock_response.json.return_value = {"value": [mock_event]}
        mock_response.raise_for_status = MagicMock()

        with patch.object(adapter, "_graph_get", return_value={"value": [mock_event]}):
            metadata = await adapter.get_meeting_metadata()

        assert metadata["title"] == "Sprint Planning"
        assert "Bob" in metadata["participants"]
        assert adapter._active_chat_id == "19:thread-id@thread.v2"


class TestTeamsAdapterChat:
    """Tests for Teams chat polling."""

    @pytest.fixture
    def adapter_with_creds(self, test_settings: Settings) -> TeamsAdapter:
        settings = Settings(
            anthropic_api_key="test",
            teams_client_id="fake-id",
            teams_tenant_id="fake-tenant",
        )
        adapter = TeamsAdapter(settings)
        adapter._active_chat_id = "19:fake-thread@thread.v2"
        adapter._access_token = "fake-token"
        adapter._token_expires_at = datetime(2099, 1, 1, tzinfo=UTC)
        return adapter

    @pytest.mark.asyncio
    async def test_stream_chat_yields_new_messages(
        self, adapter_with_creds: TeamsAdapter
    ) -> None:
        """Chat polling should yield new messages and skip already-seen ones."""
        mock_messages = {
            "value": [
                {
                    "id": "msg-001",
                    "body": {"content": "Hello from Teams chat!"},
                    "from": {"user": {"displayName": "Bob"}},
                    "createdDateTime": "2024-01-01T10:00:00Z",
                },
            ]
        }

        yielded = []
        call_count = 0

        async def fake_graph_get(path, params=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_messages
            raise asyncio.CancelledError  # Stop after first poll

        import asyncio
        adapter_with_creds._graph_get = fake_graph_get

        try:
            async for msg in adapter_with_creds.stream_chat_messages():
                yielded.append(msg)
                break  # only need one
        except asyncio.CancelledError:
            pass

        assert len(yielded) == 1
        assert yielded[0].sender == "Bob"
        assert yielded[0].content == "Hello from Teams chat!"
        assert yielded[0].platform == Platform.TEAMS

    @pytest.mark.asyncio
    async def test_stream_chat_deduplicates_messages(
        self, adapter_with_creds: TeamsAdapter
    ) -> None:
        """Same message ID should not be yielded twice."""
        same_message = {
            "id": "msg-dupe",
            "body": {"content": "Duplicate message"},
            "from": {"user": {"displayName": "Alice"}},
            "createdDateTime": "2024-01-01T10:00:00Z",
        }

        call_count = 0

        import asyncio

        async def fake_graph_get(path, params=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return {"value": [same_message]}
            raise asyncio.CancelledError

        adapter_with_creds._graph_get = fake_graph_get

        yielded = []
        try:
            async for msg in adapter_with_creds.stream_chat_messages():
                yielded.append(msg)
        except asyncio.CancelledError:
            pass

        # Should only appear once despite two polls returning it
        assert len(yielded) == 1

    @pytest.mark.asyncio
    async def test_stream_chat_no_credentials(self, test_settings: Settings) -> None:
        """Without credentials, stream yields nothing."""
        adapter = TeamsAdapter(test_settings)
        messages = []
        async for msg in adapter.stream_chat_messages():
            messages.append(msg)
        assert messages == []
