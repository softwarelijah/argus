"""Pipeline subpackage: end-to-end detection + tracking over video."""

from .async_pipeline import AsyncVideoPipeline, frame_source
from .video_pipeline import PipelineResult, VideoPipeline

__all__ = ["VideoPipeline", "PipelineResult", "AsyncVideoPipeline", "frame_source"]
