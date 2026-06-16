"""Global Motion Compensation (GMC) for moving-camera tracking.

Aerial footage is filmed from a moving drone, so the whole scene shifts between
frames. A constant-velocity Kalman filter assumes a static camera and treats
that global shift as per-object motion, which causes drift and ID switches.

GMC estimates the frame-to-frame camera motion as a 2x3 affine transform and
lets the tracker warp its predictions into the current frame before
association. This is the same idea BoT-SORT uses on top of ByteTrack.

Three backends are provided:
  - "orb"  feature matching + partial-affine estimation (fast, robust)
  - "ecc"  intensity-based enhanced correlation coefficient (accurate, slower)
  - "none" identity (no compensation)
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

IDENTITY_AFFINE = np.eye(2, 3, dtype=np.float32)


class GMC:
    """Estimate the 2x3 affine camera motion between consecutive frames."""

    def __init__(self, method: str = "orb", downscale: int = 2) -> None:
        if method not in {"orb", "ecc", "none"}:
            raise ValueError(f"unknown GMC method: {method!r}")
        self.method = method
        self.downscale = max(1, int(downscale))

        self._prev_gray: np.ndarray | None = None
        self._prev_keypoints = None
        self._prev_descriptors = None

        if method == "orb" and cv2 is not None:
            self._detector = cv2.ORB_create(nfeatures=2000)
            self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        elif method == "ecc" and cv2 is not None:
            self._criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                50,
                1e-4,
            )

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_keypoints = None
        self._prev_descriptors = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return the 2x3 affine mapping the previous frame onto ``frame``."""
        if self.method == "none" or cv2 is None:
            return IDENTITY_AFFINE.copy()
        if self.method == "ecc":
            return self._apply_ecc(frame)
        return self._apply_features(frame)

    # -- preprocessing --------------------------------------------------------
    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self.downscale > 1:
            h, w = gray.shape[:2]
            gray = cv2.resize(gray, (w // self.downscale, h // self.downscale))
        return gray

    # -- ORB feature backend --------------------------------------------------
    def _apply_features(self, frame: np.ndarray) -> np.ndarray:
        gray = self._to_gray(frame)
        keypoints, descriptors = self._detector.detectAndCompute(gray, None)

        if self._prev_descriptors is None or descriptors is None or len(keypoints) < 4:
            self._prev_gray = gray
            self._prev_keypoints = keypoints
            self._prev_descriptors = descriptors
            return IDENTITY_AFFINE.copy()

        knn = self._matcher.knnMatch(self._prev_descriptors, descriptors, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:  # Lowe ratio test
                good.append(m)

        warp = IDENTITY_AFFINE.copy()
        if len(good) >= 4:
            prev_pts = np.float32([self._prev_keypoints[m.queryIdx].pt for m in good])
            curr_pts = np.float32([keypoints[m.trainIdx].pt for m in good])
            affine, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC)
            if affine is not None:
                warp = affine.astype(np.float32)
                if self.downscale > 1:
                    warp[0, 2] *= self.downscale
                    warp[1, 2] *= self.downscale

        self._prev_gray = gray
        self._prev_keypoints = keypoints
        self._prev_descriptors = descriptors
        return warp

    # -- ECC intensity backend ------------------------------------------------
    def _apply_ecc(self, frame: np.ndarray) -> np.ndarray:
        gray = self._to_gray(frame)
        if self._prev_gray is None:
            self._prev_gray = gray
            return IDENTITY_AFFINE.copy()

        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(
                self._prev_gray, gray, warp, cv2.MOTION_AFFINE, self._criteria, None, 1
            )
        except cv2.error:  # pragma: no cover - convergence failure
            warp = IDENTITY_AFFINE.copy()

        if self.downscale > 1:
            warp[0, 2] *= self.downscale
            warp[1, 2] *= self.downscale
        self._prev_gray = gray
        return warp
