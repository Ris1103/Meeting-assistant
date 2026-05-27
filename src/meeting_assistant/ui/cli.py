"""Command-line interface for the Meeting Assistant.

Commands:
  start       — Start the meeting assistant for a session
  devices     — List available audio input devices
  test-audio  — Test audio capture (loopback + mic)
  test-stt    — Test Whisper STT with a 5-second mic recording
  test-llm    — Test connection to Anthropic Claude API
  setup       — Guided OAuth setup for platform integrations
  transcribe  — Transcribe an audio file offline

Examples:
  meeting-assistant start --platform zoom
  meeting-assistant start --platform teams --meeting-id abc123
  meeting-assistant start --auto-detect
  meeting-assistant devices
  meeting-assistant test-stt
  meeting-assistant transcribe meeting.wav
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from meeting_assistant.config import Settings, settings
from meeting_assistant.models.schemas import Platform

app = typer.Typer(
    name="meeting-assistant",
    help="Cross-platform AI meeting assistant for Zoom, Teams, and Google Meet.",
    add_completion=False,
)
console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


# ---------------------------------------------------------------------------
# Platform helper
# ---------------------------------------------------------------------------


def _get_adapter(platform_name: str, config: Settings):
    """Instantiate the platform adapter for the given platform name."""
    from meeting_assistant.platforms.google_meet import GoogleMeetAdapter
    from meeting_assistant.platforms.teams import TeamsAdapter
    from meeting_assistant.platforms.zoom import ZoomAdapter
    from meeting_assistant.platforms.base import NullPlatformAdapter

    mapping = {
        "zoom": ZoomAdapter,
        "teams": TeamsAdapter,
        "google_meet": GoogleMeetAdapter,
        "meet": GoogleMeetAdapter,
        "none": NullPlatformAdapter,
    }
    cls = mapping.get(platform_name.lower())
    if cls is None:
        console.print(
            f"[red]Unknown platform: {platform_name!r}. "
            f"Choose from: zoom, teams, google_meet, none[/red]"
        )
        raise typer.Exit(code=1)
    if cls == NullPlatformAdapter:
        return NullPlatformAdapter()
    return cls(config)


async def _auto_detect_platform(config: Settings):
    """Auto-detect which meeting platform is currently active."""
    from meeting_assistant.platforms.teams import TeamsAdapter
    from meeting_assistant.platforms.zoom import ZoomAdapter
    from meeting_assistant.platforms.google_meet import GoogleMeetAdapter
    from meeting_assistant.platforms.base import NullPlatformAdapter

    adapters = [
        TeamsAdapter(config),
        ZoomAdapter(config),
        GoogleMeetAdapter(config),
    ]
    for adapter in adapters:
        if await adapter.detect_active_meeting():
            console.print(f"[green]Auto-detected platform: {adapter.PLATFORM_NAME}[/green]")
            return adapter

    console.print("[yellow]No active meeting platform detected. Running in offline mode.[/yellow]")
    return NullPlatformAdapter()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def start(
    platform: str = typer.Option(
        "auto",
        "--platform",
        "-p",
        help="Meeting platform: zoom, teams, google_meet, none, auto",
    ),
    meeting_id: str = typer.Option(
        "",
        "--meeting-id",
        "-m",
        help="Platform-specific meeting identifier",
    ),
    auto_detect: bool = typer.Option(
        False,
        "--auto-detect",
        help="Auto-detect the active meeting platform",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Start the meeting assistant for an active meeting session."""
    _setup_logging(verbose)

    if not settings.has_anthropic_key:
        console.print(
            "[red]Error: ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file or environment variables.[/red]"
        )
        raise typer.Exit(code=1)

    async def _run() -> None:
        from meeting_assistant.core.pipeline import MeetingAssistantPipeline

        platform_name = platform
        if auto_detect or platform_name == "auto":
            adapter = await _auto_detect_platform(settings)
        else:
            adapter = _get_adapter(platform_name, settings)

        # Determine platform enum
        platform_enum = Platform.GENERIC
        for p in Platform:
            if p.value == getattr(adapter, "PLATFORM_NAME", "generic"):
                platform_enum = p
                break

        pipeline = MeetingAssistantPipeline(settings, adapter)

        rprint(
            Panel.fit(
                f"[bold green]Meeting Assistant[/bold green]\n"
                f"Platform: [cyan]{adapter.PLATFORM_NAME}[/cyan]\n"
                f"Trigger keyword: [yellow]{settings.trigger_keyword!r}[/yellow]\n"
                f"Whisper model: [blue]{settings.whisper_model_size}[/blue]\n"
                f"TTS voice: [blue]{settings.tts_voice}[/blue]\n\n"
                f"Say [bold]{settings.trigger_keyword!r}[/bold] to trigger a response.\n"
                f"Press [bold]Ctrl+C[/bold] to stop.",
                title="[bold]Starting[/bold]",
            )
        )

        try:
            await pipeline.start(
                meeting_id=meeting_id or "session-001",
                platform=platform_enum,
            )
            await pipeline.run_until_stopped()
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping…[/yellow]")
        finally:
            await pipeline.stop()

            # Print summary
            console.print("\n[bold]Generating meeting summary…[/bold]")
            try:
                summary = await pipeline.get_summary()
                rprint(Panel(summary, title="[bold]Meeting Summary[/bold]"))

                action_items = await pipeline.get_action_items()
                if action_items:
                    rprint(Panel(
                        "\n".join(f"{i+1}. {item}" for i, item in enumerate(action_items)),
                        title="[bold]Action Items[/bold]",
                    ))
            except Exception as exc:
                console.print(f"[yellow]Could not generate summary: {exc}[/yellow]")

    asyncio.run(_run())


@app.command()
def devices() -> None:
    """List all available audio input/output devices."""
    _setup_logging()
    from meeting_assistant.core.audio_capture import list_audio_devices

    all_devices = list_audio_devices()
    if not all_devices:
        console.print("[red]No audio devices found.[/red]")
        raise typer.Exit(code=1)

    table = Table(title="Available Audio Devices")
    table.add_column("Index", style="cyan", no_wrap=True)
    table.add_column("Name", style="white")
    table.add_column("Input Ch", style="green")
    table.add_column("Output Ch", style="blue")
    table.add_column("Sample Rate", style="yellow")

    for dev in all_devices:
        table.add_row(
            str(dev["index"]),
            dev["name"],
            str(dev["max_input_channels"]),
            str(dev["max_output_channels"]),
            f"{dev['default_samplerate']:.0f} Hz",
        )

    console.print(table)

    # Highlight loopback devices
    console.print("\n[dim]Tip: Loopback devices for meeting audio capture:[/dim]")
    console.print("[dim]  Linux: Look for '.monitor' in the name[/dim]")
    console.print("[dim]  macOS: Install BlackHole — https://github.com/ExistentialAudio/BlackHole[/dim]")
    console.print("[dim]  Windows: WASAPI loopback is auto-detected[/dim]")


@app.command()
def test_audio(
    duration: int = typer.Option(10, help="Recording duration in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Test audio capture — record for N seconds and show audio levels."""
    _setup_logging(verbose)

    async def _run() -> None:
        import numpy as np
        from meeting_assistant.core.audio_capture import AudioCaptureManager

        console.print(f"[cyan]Recording audio for {duration} seconds…[/cyan]")
        console.print("[dim]Speak or play audio to see levels.[/dim]\n")

        manager = AudioCaptureManager(settings)
        await manager.start()

        mic_rms_values = []
        loopback_rms_values = []
        import time
        start = time.monotonic()

        async for chunk in manager.stream():
            arr = __import__("numpy").frombuffer(chunk.data, dtype="float32")
            rms = float(__import__("numpy").sqrt(__import__("numpy").mean(arr ** 2)))
            bar = "█" * int(rms * 200)

            if chunk.source == "mic":
                mic_rms_values.append(rms)
                console.print(f"[green]MIC    [{bar:<30}] {rms:.4f}[/green]", end="\r")
            else:
                loopback_rms_values.append(rms)
                console.print(f"[blue]LOOP   [{bar:<30}] {rms:.4f}[/blue]", end="\r")

            if time.monotonic() - start >= duration:
                break

        await manager.stop()

        console.print("\n\n[bold]Audio capture test complete.[/bold]")
        if mic_rms_values:
            console.print(f"  Mic avg RMS: {sum(mic_rms_values)/len(mic_rms_values):.4f}")
        if loopback_rms_values:
            console.print(f"  Loopback avg RMS: {sum(loopback_rms_values)/len(loopback_rms_values):.4f}")
        else:
            console.print("[yellow]  No loopback audio detected. Check your loopback device setup.[/yellow]")

    asyncio.run(_run())


@app.command()
def test_stt(
    duration: int = typer.Option(5, help="Recording duration before transcription"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Test Whisper STT — record from mic and transcribe."""
    _setup_logging(verbose)

    async def _run() -> None:
        import numpy as np
        from meeting_assistant.core.audio_capture import AudioCaptureManager
        from meeting_assistant.core.transcription import TranscriptionEngine

        console.print(f"[cyan]Recording {duration}s of audio…[/cyan]")
        console.print("[bold]Speak now![/bold]\n")

        manager = AudioCaptureManager(settings)
        await manager.start()

        buffers: list[bytes] = []
        import time
        start = time.monotonic()

        async for chunk in manager.stream():
            if chunk.source == "mic":
                buffers.append(chunk.data)
            if time.monotonic() - start >= duration:
                break

        await manager.stop()

        if not buffers:
            console.print("[red]No audio captured.[/red]")
            return

        audio = np.frombuffer(b"".join(buffers), dtype=np.float32)
        console.print(f"[dim]Captured {len(audio)/settings.audio_sample_rate:.1f}s of audio[/dim]")
        console.print("[cyan]Transcribing…[/cyan]")

        engine = TranscriptionEngine(settings)
        await engine.initialize()
        result = await engine.transcribe(audio)
        await engine.shutdown()

        if result.segments:
            console.print("\n[bold green]Transcript:[/bold green]")
            for seg in result.segments:
                console.print(f"  [{seg.speaker_id}] {seg.text}")
        else:
            console.print("[yellow]No speech detected in the recording.[/yellow]")

    asyncio.run(_run())


@app.command()
def test_llm(
    prompt: str = typer.Argument(
        "In one sentence, what is a meeting assistant?",
        help="Test prompt to send to Claude",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Test connection to Anthropic Claude API."""
    _setup_logging(verbose)

    if not settings.has_anthropic_key:
        console.print("[red]ANTHROPIC_API_KEY not set.[/red]")
        raise typer.Exit(code=1)

    async def _run() -> None:
        from meeting_assistant.core.llm_client import ClaudeClient

        client = ClaudeClient(settings)
        console.print(f"[cyan]Sending test prompt to Claude ({settings.claude_model})…[/cyan]")
        console.print(f"[dim]Prompt: {prompt}[/dim]\n")

        console.print("[bold]Response:[/bold] ", end="")
        async for chunk in client.answer_streaming(prompt, ""):
            console.print(chunk, end="", highlight=False)
        console.print("\n")
        console.print("[green]✓ Claude API connection successful[/green]")

    asyncio.run(_run())


@app.command()
def setup(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Guided setup wizard for platform OAuth integrations."""
    _setup_logging(verbose)

    console.print(
        Panel(
            "[bold]Meeting Assistant Setup Wizard[/bold]\n\n"
            "This wizard will help you configure OAuth credentials for\n"
            "Zoom, Microsoft Teams, and Google Meet integration.\n\n"
            "You'll need API credentials from each platform's developer console.\n"
            "See the README for detailed instructions.",
            title="Setup",
        )
    )

    platforms_to_configure = []

    if typer.confirm("Configure Zoom integration?", default=False):
        platforms_to_configure.append("zoom")
    if typer.confirm("Configure Microsoft Teams integration?", default=False):
        platforms_to_configure.append("teams")
    if typer.confirm("Configure Google Meet integration?", default=False):
        platforms_to_configure.append("google_meet")

    if not platforms_to_configure:
        console.print("[yellow]No platforms selected. You can still use audio-only mode.[/yellow]")
        return

    console.print(
        "\n[green]Add the following to your .env file:[/green]\n"
    )

    if "zoom" in platforms_to_configure:
        console.print(
            "[dim]# Zoom — create OAuth app at https://marketplace.zoom.us/develop/create[/dim]"
        )
        console.print("ZOOM_CLIENT_ID=<your-zoom-client-id>")
        console.print("ZOOM_CLIENT_SECRET=<your-zoom-client-secret>\n")

    if "teams" in platforms_to_configure:
        console.print(
            "[dim]# Teams — register app at https://portal.azure.com[/dim]"
        )
        console.print("TEAMS_CLIENT_ID=<your-azure-app-client-id>")
        console.print("TEAMS_TENANT_ID=<your-tenant-id-or-common>\n")

    if "google_meet" in platforms_to_configure:
        console.print(
            "[dim]# Google — create OAuth credentials at https://console.cloud.google.com[/dim]"
        )
        console.print("GOOGLE_CLIENT_ID=<your-google-client-id>")
        console.print("GOOGLE_CLIENT_SECRET=<your-google-client-secret>\n")


@app.command()
def transcribe(
    file: Path = typer.Argument(..., help="Path to audio file (WAV, MP3, etc.)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save transcript to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Transcribe an audio file offline using Whisper."""
    _setup_logging(verbose)

    if not file.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(code=1)

    async def _run() -> None:
        import numpy as np
        import soundfile as sf  # type: ignore[import]
        from meeting_assistant.core.transcription import TranscriptionEngine

        console.print(f"[cyan]Loading audio: {file}[/cyan]")
        try:
            audio_data, sample_rate = sf.read(str(file), dtype="float32", always_2d=False)
        except Exception as exc:
            console.print(f"[red]Failed to load audio: {exc}[/red]")
            return

        # Resample to 16kHz if needed
        if sample_rate != settings.audio_sample_rate:
            console.print(f"[dim]Resampling from {sample_rate}Hz to {settings.audio_sample_rate}Hz…[/dim]")
            from scipy.signal import resample_poly  # type: ignore[import]
            from math import gcd
            g = gcd(settings.audio_sample_rate, sample_rate)
            audio_data = resample_poly(
                audio_data,
                settings.audio_sample_rate // g,
                sample_rate // g,
            ).astype(np.float32)

        # Mono
        if audio_data.ndim > 1:
            audio_data = audio_data.mean(axis=1)

        console.print(f"[dim]Duration: {len(audio_data)/settings.audio_sample_rate:.1f}s[/dim]")
        console.print("[cyan]Transcribing…[/cyan]")

        engine = TranscriptionEngine(settings)
        await engine.initialize()
        result = await engine.transcribe(audio_data)
        await engine.shutdown()

        lines = []
        for seg in result.segments:
            line = f"[{seg.start:.1f}s] {seg.speaker_id}: {seg.text}"
            lines.append(line)
            console.print(line)

        if output:
            output.write_text("\n".join(lines))
            console.print(f"\n[green]Transcript saved to: {output}[/green]")

    asyncio.run(_run())


if __name__ == "__main__":
    app()
