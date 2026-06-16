"""Argus: real-time aerial detection and multi-object tracking.

Top-level imports are kept dependency-light. The tracking stack (ByteTracker,
KalmanFilter) is pure numpy/scipy and always importable; detector and TensorRT
backends are imported from their subpackages on demand so the package installs
cleanly on CPU-only hosts.
"""

from .tracking import ByteTracker, STrack, TrackerConfig

__version__ = "0.1.0"

__all__ = ["ByteTracker", "TrackerConfig", "STrack", "__version__"]
