"""
Core package for the realtime multi-stream analytics platform.

Modules expose building blocks for configuring, running, and instrumenting
32 stream capable detection pipelines.
"""

from .config import PipelineConfig, load_config  # noqa: F401
from .detector import Detection, create_detector  # noqa: F401
from .pipeline import AnalyticsPipeline  # noqa: F401
