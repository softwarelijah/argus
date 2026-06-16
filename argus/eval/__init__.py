"""Evaluation subpackage: MOT metrics and MOTChallenge IO."""

from .mot_io import load_mot, write_mot
from .mot_metrics import MOTResult, evaluate
from .synthetic import generate_mot_scene, run_tracker

__all__ = [
    "evaluate",
    "MOTResult",
    "load_mot",
    "write_mot",
    "generate_mot_scene",
    "run_tracker",
]
