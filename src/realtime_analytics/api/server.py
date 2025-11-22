"""
FastAPI application serving the realtime dashboard.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import csv
import io
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..config import KafkaSinkConfig
from .kafka_consumer import DetectionConsumer
from .schemas import WsEnvelope
from .state import ConnectionManager, DashboardState

LOGGER = logging.getLogger(__name__)
FAVICON_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="
)


class AppContext:
    """Holds shared dashboard components."""

    def __init__(self, kafka_config: KafkaSinkConfig):
        self.state = DashboardState()
        self.connections = ConnectionManager()
        self.kafka_config = kafka_config
        self.consumer = DetectionConsumer(
            bootstrap_servers=kafka_config.bootstrap_servers,
            topic=kafka_config.topic,
            state=self.state,
            connections=self.connections,
        )

    async def start(self) -> None:
        await self.consumer.start()

    async def stop(self) -> None:
        await self.consumer.stop()


def create_app(
    kafka_config: Optional[KafkaSinkConfig] = None,
    static_dir: Optional[Path] = None,
) -> FastAPI:
    kafka_cfg = kafka_config or KafkaSinkConfig(enabled=False)
    app = FastAPI(title="Realtime Analytics Dashboard")
    context = AppContext(kafka_cfg)

    @app.on_event("startup")
    async def _on_startup() -> None:
        if kafka_cfg.enabled:
            await context.start()
        else:
            LOGGER.info("Kafka disabled; dashboard will run without live updates")

    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        await context.stop()

    router = APIRouter()

    @router.get("/api/snapshot")
    async def get_snapshot():
        snapshot = await context.state.snapshot()
        return Response(content=snapshot.model_dump_json(), media_type="application/json")

    @router.get("/favicon.ico")
    async def favicon() -> Response:
        return Response(content=FAVICON_PNG, media_type="image/png")

    def _filter_streams(streams: Optional[str]):
        if not streams:
            return None
        return {s.strip() for s in streams.split(",") if s.strip()}

    @router.get("/api/export/json")
    async def export_json(streams: Optional[str] = None):
        """
        Export the latest dashboard snapshot as JSON.
        Optional query param `streams=cam01,cam02` to filter by stream names.
        """
        snapshot = await context.state.snapshot()
        allowed = _filter_streams(streams)
        events = (
            [evt for evt in snapshot.streams if evt.stream in allowed]
            if allowed
            else snapshot.streams
        )
        payload = [evt.model_dump() for evt in events]
        return JSONResponse(content=payload)

    @router.get("/api/export/csv")
    async def export_csv(streams: Optional[str] = None):
        """
        Export the latest dashboard snapshot as CSV.
        Columns: stream, frame_id, received_at, track_id, class_id, confidence, x1, y1, x2, y2, action_label, temporal_score
        """
        snapshot = await context.state.snapshot()
        allowed = _filter_streams(streams)
        events = (
            [evt for evt in snapshot.streams if evt.stream in allowed]
            if allowed
            else snapshot.streams
        )
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "stream",
                "frame_id",
                "received_at",
                "track_id",
                "class_id",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
                "action_label",
                "temporal_score",
            ]
        )
        for evt in events:
            ts = evt.received_at.isoformat()
            for track in evt.tracks:
                x1, y1, x2, y2 = track.bbox_xyxy
                writer.writerow(
                    [
                        evt.stream,
                        evt.frame_id,
                        ts,
                        track.track_id,
                        track.class_id,
                        track.confidence,
                        x1,
                        y1,
                        x2,
                        y2,
                        track.action_label or "",
                        track.temporal_score if track.temporal_score is not None else "",
                    ]
                )
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=analytics_snapshot.csv"},
        )

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await context.connections.connect(websocket)
        try:
            snapshot = await context.state.snapshot()
            envelope = WsEnvelope(type="snapshot", payload=snapshot)
            await websocket.send_text(envelope.model_dump_json())
            while True:
                await websocket.receive_text()
        except Exception:
            LOGGER.debug("Websocket receive loop ended")
        finally:
            await context.connections.disconnect(websocket)

    app.include_router(router)

    static_base = static_dir or Path(__file__).resolve().parent / "static"
    modern_index = static_base / "modern-dashboard.html"
    index_file = static_base / "index.html"
    app.mount(
        "/static",
        StaticFiles(directory=str(static_base)),
        name="static",
    )

    @app.get("/", response_class=Response)
    async def serve_index() -> Response:
        # 默认提供可实时联动的 index.html（依赖 /api/snapshot 与 /ws）。
        # 纯静态展示仍可访问 /static/modern-dashboard.html。
        return FileResponse(index_file)

    return app
