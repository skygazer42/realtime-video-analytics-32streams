#!/usr/bin/env python3
"""
CLI to launch the realtime analytics dashboard (FastAPI + WebSocket).
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
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
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (optional, logs to file in addition to console)",
    )
    parser.add_argument(
        "--log-format",
        default="standard",
        choices=["standard", "detailed", "json"],
        help="Log format (default: standard)",
    )
    parser.add_argument(
        "--log-rotate",
        action="store_true",
        help="Enable log rotation (only with --log-file, max 10MB x 5 files)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored log output",
    )
    return parser.parse_args(argv)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, use_color: bool = True):
        super().__init__(fmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color and record.levelname in self.COLORS:
            # Add color to level name
            levelname_color = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            record_copy = logging.makeLogRecord(record.__dict__)
            record_copy.levelname = levelname_color
            return super().format(record_copy)
        return super().format(record)


def setup_logging(args: argparse.Namespace) -> None:
    """
    Configure logging with advanced features:
    - Multiple output formats
    - File and console handlers
    - Log rotation
    - Colored output
    """
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)

    # Choose format based on user preference
    if args.log_format == "detailed":
        log_format = (
            "%(asctime)s [%(levelname)-8s] [%(process)d:%(thread)d] "
            "%(name)s:%(funcName)s:%(lineno)d | %(message)s"
        )
    elif args.log_format == "json":
        # Simplified JSON-like format (for machine parsing)
        log_format = (
            '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
            '"message":"%(message)s"}'
        )
    else:  # standard
        log_format = "%(asctime)s [%(levelname)-8s] %(name)s | %(message)s"

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler with color support
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    use_color = not args.no_color and sys.stdout.isatty() and args.log_format != "json"
    if use_color:
        console_formatter = ColoredFormatter(log_format)
    else:
        console_formatter = logging.Formatter(log_format)

    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if requested)
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        if args.log_rotate:
            # Rotating file handler: max 10MB, keep 5 backup files
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8",
            )
        else:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")

        file_handler.setLevel(log_level)
        # File logs should never have color codes
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

        logging.info(f"Logging to file: {log_path}")


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

    # Setup enhanced logging
    setup_logging(args)

    logger = logging.getLogger(__name__)
    logger.info("Starting realtime analytics dashboard")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Binding to {args.host}:{args.port}")

    kafka_cfg = build_kafka_config(args)

    if kafka_cfg.enabled:
        logger.info(f"Kafka enabled: {kafka_cfg.bootstrap_servers} (topic: {kafka_cfg.topic})")
    else:
        logger.info("Kafka disabled (running in standalone mode)")

    app = create_app(kafka_config=kafka_cfg)

    try:
        import uvicorn
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "Dashboard requires uvicorn. Install optional dependencies with "
            "`uv pip install uvicorn[standard] fastapi` or `pip install realtime-video-analytics-32streams[dashboard]`."
        ) from exc

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    except KeyboardInterrupt:
        logger.info("Dashboard shutdown by user")
    except Exception as exc:
        logger.exception("Dashboard failed with error: %s", exc)
        return 1

    logger.info("Dashboard shutdown complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
