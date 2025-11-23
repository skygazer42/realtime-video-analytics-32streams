"""
Helper utilities for spawning ffmpeg processes that emulate RTSP cameras.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import time
from collections import deque
from threading import Thread
from typing import Deque, List
from urllib.parse import urlparse, urlunparse

from .config import FFmpegSimulatorConfig, StreamConfig

LOGGER = logging.getLogger(__name__)


class FFmpegStreamError(RuntimeError):
    """Raised when an ffmpeg simulator process cannot be started or terminates unexpectedly."""


class FFmpegStreamSimulator:
    """Lifecycle wrapper around an ffmpeg subprocess that serves RTSP video."""

    def __init__(self, stream: StreamConfig, config: FFmpegSimulatorConfig):
        self.stream = stream
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._stderr_thread: Thread | None = None
        self._stderr_tail: Deque[str] = deque(maxlen=50)

    def start(self) -> None:
        if not self.config.enabled:
            return
        if self._process and self._process.poll() is None:
            LOGGER.debug("ffmpeg simulator for '%s' already running", self.stream.name)
            return

        cmd = self._build_command()
        LOGGER.info("Starting ffmpeg simulator for stream '%s': %s", self.stream.name, _format_cmd(cmd))
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as exc:
            raise FFmpegStreamError(
                "ffmpeg executable not found. Ensure ffmpeg is installed and available in PATH."
            ) from exc

        if self._process.stderr is not None:
            self._stderr_thread = Thread(
                target=self._consume_stderr,
                name=f"ffmpeg-sim-{self.stream.name}",
                daemon=True,
            )
            self._stderr_thread.start()

        # Give ffmpeg a brief moment to validate its inputs.
        time.sleep(0.1)
        code = self._process.poll()
        if code is not None:
            raise FFmpegStreamError(
                f"ffmpeg simulator for stream '{self.stream.name}' exited immediately "
                f"with status {code}. Output:\n{self._collect_stderr_tail()}"
            )

    def stop(self, timeout: float = 5.0) -> None:
        if not self._process:
            return
        if self._process.poll() is None:
            LOGGER.info("Stopping ffmpeg simulator for stream '%s'", self.stream.name)
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                LOGGER.warning(
                    "ffmpeg simulator for stream '%s' did not terminate, sending SIGKILL",
                    self.stream.name,
                )
                self._process.kill()
                try:
                    self._process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    LOGGER.error(
                        "ffmpeg simulator for stream '%s' failed to exit after SIGKILL",
                        self.stream.name,
                    )

        if self._stderr_thread and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1.0)

        self._process = None
        self._stderr_thread = None
        self._stderr_tail.clear()

    def _consume_stderr(self) -> None:
        assert self._process is not None
        assert self._process.stderr is not None
        for line in self._process.stderr:
            if not line:
                continue
            clean = line.rstrip()
            if not clean:
                continue
            self._stderr_tail.append(clean)
            LOGGER.debug("ffmpeg[%s] %s", self.stream.name, clean)

    def _collect_stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    def _build_command(self) -> List[str]:
        cfg = self.config
        input_source = os.path.expanduser(cfg.input)
        if "://" not in input_source and not os.path.exists(input_source):
            LOGGER.warning(
                "Input source '%s' for stream '%s' does not exist; ffmpeg may fail to start",
                cfg.input,
                self.stream.name,
            )

        output_url = self._build_listen_url()
        cmd: List[str] = ["ffmpeg", "-hide_banner", "-loglevel", cfg.log_level]
        if cfg.loop:
            cmd.extend(["-stream_loop", "-1"])
        cmd.extend(["-re", "-i", input_source])

        if cfg.video_codec:
            cmd.extend(["-c:v", cfg.video_codec])
            if cfg.video_codec == "libx264":
                cmd.extend(["-preset", "veryfast", "-tune", "zerolatency"])
        if cfg.audio_enabled:
            cmd.extend(["-c:a", cfg.audio_codec])
        else:
            cmd.append("-an")

        cmd.extend(cfg.extra_args)
        # Align with scripts/rtsp-multistream: force TCP, low muxdelay, explicit listen mode
        cmd.extend(
            [
                "-f",
                "rtsp",
                "-rtsp_transport",
                "tcp",
                "-muxdelay",
                "0.1",
                "-listen",
                "1",
                output_url,
            ]
        )
        return cmd

    def _build_listen_url(self) -> str:
        parsed = urlparse(self.stream.url)
        if parsed.scheme.lower() != "rtsp":
            raise FFmpegStreamError(
                f"ffmpeg simulator only supports RTSP outputs (stream '{self.stream.name}')"
            )
        listen_host = self.config.listen_host or "0.0.0.0"
        port = parsed.port or 8554
        netloc = f"{listen_host}:{port}"
        if parsed.username or parsed.password:
            # Strip credentials for listening socket; ffmpeg listen mode should not include them.
            LOGGER.warning(
                "Ignoring credentials in stream url for ffmpeg simulator on stream '%s'",
                self.stream.name,
            )
        # ParseResult doesn't accept username/password in _replace; override netloc only.
        listen = parsed._replace(netloc=netloc)
        return urlunparse(listen)


def _format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)
