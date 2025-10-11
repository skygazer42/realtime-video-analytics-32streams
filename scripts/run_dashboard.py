#!/usr/bin/env python3
"""
CLI to launch the realtime analytics dashboard (FastAPI + WebSocket).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

from realtime_analytics.api import create_app
from realtime_analytics.config import ConfigError, KafkaSinkConfig, load_config


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime analytics dashboard (FastAPI)"
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Pipeline YAML config to re-use Kafka settings (optional)",
    )
    parser.add_argument(
        "--kafka-bootstrap",
        default="localhost:9092",
        help="Kafka bootstrap servers (overrides config)",
    )
    parser.add_argument(
        "--kafka-topic",
        default="analytics",
        help="Kafka topic name (overrides config)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="HTTP host to bind",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP port to bind",
    )
    parser.add_argument(
        "--no-kafka",
        action="store_true",
        help="Disable Kafka consumer even if configured",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args(argv)


def build_kafka_config(args: argparse.Namespace) -> KafkaSinkConfig:
    kafka_cfg = KafkaSinkConfig(enabled=not args.no_kafka)
    kafka_cfg.bootstrap_servers = args.kafka_bootstrap
    kafka_cfg.topic = args.kafka_topic

    if args.config:
        try:
            pipeline_cfg = load_config(args.config)
        except ConfigError as exc:
            raise SystemExit(f"Failed to load config: {exc}") from exc
        kafka_cfg = pipeline_cfg.kafka
        kafka_cfg.enabled = pipeline_cfg.kafka.enabled and not args.no_kafka
        if args.kafka_bootstrap:
            kafka_cfg.bootstrap_servers = args.kafka_bootstrap
        if args.kafka_topic:
            kafka_cfg.topic = args.kafka_topic
    return kafka_cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )

    kafka_cfg = build_kafka_config(args)
    app = create_app(kafka_config=kafka_cfg)

    try:
        import uvicorn
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Dashboard requires uvicorn. Install optional dependencies with "
            "`uv pip install uvicorn[standard] fastapi` or `pip install realtime-video-analytics-32streams[dashboard]`."
        ) from exc

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
