"""Single-target track backed by a Kalman filter."""

from __future__ import annotations

import numpy as np

from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter


class STrack(BaseTrack):
    """A single tracklet.

    Boxes are stored internally in the (top-left x, top-left y, width, height)
    ``tlwh`` convention and converted to the Kalman ``xyah`` representation
    (centre x, centre y, aspect ratio, height) on demand.
    """

    shared_kalman = KalmanFilter()

    def __init__(
        self,
        tlwh: np.ndarray,
        score: float,
        cls: int = 0,
        feat: np.ndarray | None = None,
        feat_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter: KalmanFilter | None = None
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self.is_activated = False

        self.score = float(score)
        self.cls = int(cls)
        self.tracklet_len = 0

        # Appearance features for Re-ID (BoT-SORT style). ``smooth_feat`` is an
        # exponential moving average of observed embeddings; it stays None when
        # Re-ID is disabled, so the motion-only path is unaffected.
        self.curr_feat: np.ndarray | None = None
        self.smooth_feat: np.ndarray | None = None
        self.feat_momentum = feat_momentum
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat: np.ndarray) -> None:
        """Update the L2-normalised appearance embedding (EMA over time)."""
        feat = np.asarray(feat, dtype=np.float32)
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = (
                self.feat_momentum * self.smooth_feat + (1 - self.feat_momentum) * feat
            )
            self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def apply_gmc(self, warp: np.ndarray) -> None:
        """Warp the track state by a 2x3 affine camera-motion transform."""
        if self.mean is None:
            return
        R = warp[:2, :2]
        t = warp[:2, 2]

        # Build the 8x8 operator that rotates/scales position and velocity and
        # leaves aspect/height (and their rates) untouched.
        R8 = np.eye(8, dtype=np.float32)
        R8[:2, :2] = R
        R8[4:6, 4:6] = R

        mean = self.mean.copy()
        mean[:2] = R @ mean[:2] + t
        mean[4:6] = R @ mean[4:6]
        self.mean = mean
        self.covariance = R8 @ self.covariance @ R8.T

    # -- prediction -----------------------------------------------------------
    def predict(self) -> None:
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0  # zero the height velocity when not actively tracked
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(tracks: list[STrack]) -> None:
        if not tracks:
            return
        multi_mean = np.asarray([t.mean.copy() for t in tracks])
        multi_covariance = np.asarray([t.covariance for t in tracks])
        for i, t in enumerate(tracks):
            if t.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance
        )
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            tracks[i].mean = mean
            tracks[i].covariance = cov

    # -- lifecycle ------------------------------------------------------------
    def activate(self, kalman_filter: KalmanFilter, frame_id: int) -> None:
        """Start a new tracklet."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        # The very first frame of a sequence activates immediately; otherwise a
        # track must be re-found before it is considered confirmed.
        self.is_activated = frame_id == 1
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None:
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls

    def update(self, new_track: STrack, frame_id: int) -> None:
        """Update a matched track with an associated detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls

    # -- representations ------------------------------------------------------
    @property
    def tlwh(self) -> np.ndarray:
        """Current box as (top-left x, top-left y, width, height)."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # aspect ratio * height -> width
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self) -> np.ndarray:
        """Current box as (x1, y1, x2, y2)."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xyah(self) -> np.ndarray:
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlwh, dtype=np.float32).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr: np.ndarray) -> np.ndarray:
        ret = np.asarray(tlbr, dtype=np.float32).copy()
        ret[2:] -= ret[:2]
        return ret

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"OT_{self.track_id}_({self.start_frame}-{self.frame_id})"
