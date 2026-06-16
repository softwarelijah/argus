"""Base track bookkeeping shared by the tracker."""

from __future__ import annotations

from enum import IntEnum


class TrackState(IntEnum):
    """Lifecycle state of a single track."""

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """Common identity and state machinery for tracks.

    Track ids are handed out from a process-wide counter. Call
    :meth:`reset_id_count` between independent sequences so ids start at 1.
    """

    _count = 0

    def __init__(self) -> None:
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.frame_id = 0
        self.start_frame = 0

    @staticmethod
    def next_id() -> int:
        BaseTrack._count += 1
        return BaseTrack._count

    @staticmethod
    def reset_id_count() -> None:
        BaseTrack._count = 0

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed
