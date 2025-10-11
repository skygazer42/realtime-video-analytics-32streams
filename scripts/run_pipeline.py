#!/usr/bin/env python3
"""
CLI entrypoint for launching the realtime analytics pipeline.
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

from realtime_analytics import AnalyticsPipeline, load_config


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime multi-stream analytics pipeline"
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
    )

    config = load_config(args.config)
    pipeline = AnalyticsPipeline(config)

    try:
        import asyncio

        asyncio.run(pipeline.run_forever())
    except KeyboardInterrupt:
        logging.info("Interrupted by user, shutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
