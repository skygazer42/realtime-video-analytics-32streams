"""
Dashboard state management and websocket broadcasting utilities.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Dict, List, Set

from fastapi import WebSocket

from .schemas import DashboardSnapshot, DetectionEvent, WsEnvelope

LOGGER = logging.getLogger(__name__)


class DashboardState:
    """Holds the latest detection event per stream."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._events: Dict[str, DetectionEvent] = {}

    async def update(self, event: DetectionEvent) -> None:
        async with self._lock:
            self._events[event.stream] = event

    async def snapshot(self) -> DashboardSnapshot:
        async with self._lock:
            events = list(self._events.values())
        events.sort(key=lambda evt: evt.received_at, reverse=True)
        return DashboardSnapshot(streams=events)


class ConnectionManager:
    """Tracks websocket connections and handles broadcast messaging."""

    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        LOGGER.info("Websocket connected (%d clients)", len(self.active_connections))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self.active_connections.discard(websocket)
        LOGGER.info(
            "Websocket disconnected (%d clients)", len(self.active_connections)
        )

    async def broadcast_event(self, event: DetectionEvent) -> None:
        envelope = WsEnvelope(type="event", payload=event)
        await self._broadcast(envelope)

    async def broadcast_snapshot(self, snapshot: DashboardSnapshot) -> None:
        envelope = WsEnvelope(type="snapshot", payload=snapshot)
        await self._broadcast(envelope)

    async def _broadcast(self, envelope: WsEnvelope) -> None:
        if not self.active_connections:
            return
        message = envelope.model_dump_json()
        dead: List[WebSocket] = []
        for websocket in list(self.active_connections):
            try:
                await websocket.send_text(message)
            except Exception:
                LOGGER.warning("Websocket send failed, scheduling disconnect")
                dead.append(websocket)
        for websocket in dead:
            await self.disconnect(websocket)
