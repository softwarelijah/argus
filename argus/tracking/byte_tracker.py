"""ByteTrack multi-object tracker.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box", ECCV 2022. The key idea is a two-stage association: high
confidence detections are matched first, then the remaining low confidence
detections recover tracks that would otherwise be dropped (occlusion, motion
blur), which is exactly the regime aerial small-object footage lives in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import matching
from .basetrack import BaseTrack, TrackState
from .gmc import GMC
from .kalman_filter import KalmanFilter
from .track import STrack


@dataclass
class TrackerConfig:
    """Tunable thresholds for the tracker."""

    track_thresh: float = 0.5  # high/low detection split
    track_buffer: int = 30  # frames a lost track is kept before removal
    match_thresh: float = 0.8  # IoU gate for the first association stage
    new_track_thresh: float = 0.6  # min score to spawn a brand new track
    fuse_score: bool = True
    frame_rate: int = 30

    # Global motion compensation for moving (drone) cameras: "none", "orb", "ecc".
    gmc_method: str = "none"
    gmc_downscale: int = 2

    # Appearance Re-ID (BoT-SORT style). Requires per-detection embeddings.
    with_reid: bool = False
    proximity_thresh: float = 0.5  # IoU gate before trusting appearance
    appearance_thresh: float = 0.25  # cosine-distance gate for appearance
    appearance_weight: float = 0.5  # blend of motion vs appearance cost


def _join_tracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
    exists = {t.track_id for t in tlista}
    res = list(tlista)
    for t in tlistb:
        if t.track_id not in exists:
            exists.add(t.track_id)
            res.append(t)
    return res


def _sub_tracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
    ids_b = {t.track_id for t in tlistb}
    return [t for t in tlista if t.track_id not in ids_b]


def _remove_duplicate_tracks(
    tracks_a: list[STrack], tracks_b: list[STrack]
) -> tuple[list[STrack], list[STrack]]:
    """Drop near-identical tracks, keeping the longer lived one."""
    pdist = matching.iou_distance(tracks_a, tracks_b)
    pairs = np.where(pdist < 0.15)
    dup_a, dup_b = set(), set()
    for p, q in zip(*pairs):
        time_p = tracks_a[p].frame_id - tracks_a[p].start_frame
        time_q = tracks_b[q].frame_id - tracks_b[q].start_frame
        if time_p > time_q:
            dup_b.add(q)
        else:
            dup_a.add(p)
    resa = [t for i, t in enumerate(tracks_a) if i not in dup_a]
    resb = [t for i, t in enumerate(tracks_b) if i not in dup_b]
    return resa, resb


class ByteTracker:
    """Online multi-object tracker following the ByteTrack association scheme."""

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()

        self.tracked_tracks: list[STrack] = []
        self.lost_tracks: list[STrack] = []
        self.removed_tracks: list[STrack] = []

        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        self.max_time_lost = int(self.config.frame_rate / 30.0 * self.config.track_buffer)
        self.gmc = GMC(self.config.gmc_method, downscale=self.config.gmc_downscale)
        BaseTrack.reset_id_count()

    def reset(self) -> None:
        """Clear all state so the tracker can process a fresh sequence."""
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        self.frame_id = 0
        self.gmc.reset()
        BaseTrack.reset_id_count()

    def update(
        self,
        detections: np.ndarray,
        frame: np.ndarray | None = None,
        embeddings: np.ndarray | None = None,
    ) -> list[STrack]:
        """Advance the tracker one frame.

        ``detections`` is an (N, 6) array of ``[x1, y1, x2, y2, score, cls]``.
        ``frame`` is the current image; when ``gmc_method`` is enabled it drives
        camera-motion compensation. ``embeddings`` is an optional (N, D) array of
        per-detection appearance features used for Re-ID when ``with_reid`` is set.
        Returns the list of currently active (confirmed) tracks.
        """
        self.frame_id += 1
        cfg = self.config

        detections = np.asarray(detections, dtype=np.float32).reshape(-1, 6)
        scores = detections[:, 4]
        boxes = detections[:, :4]
        classes = detections[:, 5]

        use_reid = cfg.with_reid and embeddings is not None and len(embeddings) == len(detections)
        embeddings = np.asarray(embeddings, dtype=np.float32) if use_reid else None

        # Split detections into high and low confidence pools.
        remain_high = scores >= cfg.track_thresh
        low_band = (scores > 0.1) & (scores < cfg.track_thresh)

        feats_high = embeddings[remain_high] if use_reid else None
        dets_high = self._to_stracks(
            boxes[remain_high], scores[remain_high], classes[remain_high], feats_high
        )
        dets_low = self._to_stracks(boxes[low_band], scores[low_band], classes[low_band])

        # Partition existing tracks into confirmed and tentative.
        unconfirmed = [t for t in self.tracked_tracks if not t.is_activated]
        tracked = [t for t in self.tracked_tracks if t.is_activated]

        # --- Stage 1: associate high score detections to predicted tracks ---
        track_pool = _join_tracks(tracked, self.lost_tracks)
        STrack.multi_predict(track_pool)

        # Camera motion compensation: warp predictions into the current frame.
        if cfg.gmc_method != "none" and frame is not None:
            warp = self.gmc.apply(frame)
            for track in track_pool:
                track.apply_gmc(warp)
            for track in unconfirmed:
                track.apply_gmc(warp)

        dists = matching.iou_distance(track_pool, dets_high)
        if cfg.fuse_score:
            dists = matching.fuse_score(dists, dets_high)
        if use_reid:
            app_cost = matching.embedding_distance(track_pool, dets_high)
            dists = matching.fuse_motion_appearance(
                dists,
                app_cost,
                proximity_thresh=cfg.proximity_thresh,
                appearance_thresh=cfg.appearance_thresh,
                weight=cfg.appearance_weight,
            )
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=cfg.match_thresh)

        activated, refind = [], []
        for itracked, idet in matches:
            track = track_pool[itracked]
            det = dets_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        # --- Stage 2: associate low score detections to leftover tracks ---
        r_tracked = [track_pool[i] for i in u_track if track_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked, dets_low)
        matches, u_track_low, _ = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked[itracked]
            det = dets_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)

        lost = []
        for i in u_track_low:
            track = r_tracked[i]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        # --- Confirm tentative tracks against remaining high detections ---
        dets_high_left = [dets_high[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, dets_high_left)
        if cfg.fuse_score:
            dists = matching.fuse_score(dists, dets_high_left)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(dets_high_left[idet], self.frame_id)
            activated.append(unconfirmed[itracked])

        removed = []
        for i in u_unconfirmed:
            track = unconfirmed[i]
            track.mark_removed()
            removed.append(track)

        # --- Spawn new tracks from strong, still-unmatched detections ---
        for i in u_detection:
            track = dets_high_left[i]
            if track.score < cfg.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

        # --- Age and reap lost tracks ---
        for track in self.lost_tracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        # --- Merge bookkeeping lists ---
        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.Tracked]
        self.tracked_tracks = _join_tracks(self.tracked_tracks, activated)
        self.tracked_tracks = _join_tracks(self.tracked_tracks, refind)

        self.lost_tracks = _sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks.extend(lost)
        self.lost_tracks = _sub_tracks(self.lost_tracks, self.removed_tracks)

        self.tracked_tracks, self.lost_tracks = _remove_duplicate_tracks(
            self.tracked_tracks, self.lost_tracks
        )
        self.removed_tracks.extend(removed)
        if len(self.removed_tracks) > 1000:
            self.removed_tracks = self.removed_tracks[-1000:]

        return [t for t in self.tracked_tracks if t.is_activated]

    @staticmethod
    def _to_stracks(
        boxes: np.ndarray,
        scores: np.ndarray,
        classes: np.ndarray,
        feats: np.ndarray | None = None,
    ) -> list[STrack]:
        if feats is None:
            return [
                STrack(STrack.tlbr_to_tlwh(box), score, cls)
                for box, score, cls in zip(boxes, scores, classes)
            ]
        return [
            STrack(STrack.tlbr_to_tlwh(box), score, cls, feat)
            for box, score, cls, feat in zip(boxes, scores, classes, feats)
        ]
