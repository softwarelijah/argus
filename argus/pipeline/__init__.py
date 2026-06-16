"""Pipeline subpackage: end-to-end detection + tracking over video."""

from .video_pipeline import PipelineResult, VideoPipeline

__all__ = ["VideoPipeline", "PipelineResult"]
