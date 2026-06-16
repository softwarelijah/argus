"""Evaluation subpackage: MOT metrics and MOTChallenge IO."""

from .mot_io import load_mot, write_mot
from .mot_metrics import MOTResult, evaluate

__all__ = ["evaluate", "MOTResult", "load_mot", "write_mot"]
