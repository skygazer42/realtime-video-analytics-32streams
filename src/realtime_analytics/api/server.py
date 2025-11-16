"""
FastAPI application serving the realtime dashboard.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from ..config import KafkaSinkConfig
from .kafka_consumer import DetectionConsumer
from .schemas import WsEnvelope
from .state import ConnectionManager, DashboardState

LOGGER = logging.getLogger(__name__)


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
        return JSONResponse(snapshot.model_dump())

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
        if modern_index.exists():
            # 始终跳转到新版仪表盘
            return RedirectResponse(url="/static/modern-dashboard.html")
        return FileResponse(index_file)

    return app
