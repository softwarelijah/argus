"""Kalman filter for single-object tracking in image space.

The state is an 8 dimensional vector

    (x, y, a, h, vx, vy, va, vh)

where (x, y) is the bounding box centre, ``a`` is the aspect ratio
(width / height), ``h`` is the height and the remaining four entries are the
respective velocities. Object motion follows a constant velocity model. The
bounding box location (x, y, a, h) is taken as a direct observation of the
state (linear observation model).

This implementation mirrors the filter used by SORT / ByteTrack and keeps the
chi-square gating helpers used during data association.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

# Table of the 0.95 quantile of the chi-square distribution with N degrees of
# freedom (used as the Mahalanobis gating threshold). N is the measurement
# dimensionality, here at most 4.
CHI2INV95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilter:
    """A simple constant-velocity Kalman filter over image-space boxes."""

    def __init__(self) -> None:
        ndim, dt = 4, 1.0

        # State transition (constant velocity) and observation matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are scaled relative to the box
        # height; these weights were tuned empirically and match the upstream
        # ByteTrack defaults.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a track from an unassociated measurement.

        ``measurement`` is the box (x, y, a, h). Returns the mean vector (8d)
        and covariance matrix (8x8) of the new track. Velocities start at zero.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the prediction step (advance one frame)."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project state distribution to measurement space."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T
        return mean, covariance + innovation_cov

    def multi_predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised prediction step for a batch of tracks.

        ``mean`` has shape (N, 8) and ``covariance`` shape (N, 8, 8).
        """
        std_pos = np.stack(
            [
                self._std_weight_position * mean[:, 3],
                self._std_weight_position * mean[:, 3],
                np.full(len(mean), 1e-2),
                self._std_weight_position * mean[:, 3],
            ],
            axis=1,
        )
        std_vel = np.stack(
            [
                self._std_weight_velocity * mean[:, 3],
                self._std_weight_velocity * mean[:, 3],
                np.full(len(mean), 1e-5),
                self._std_weight_velocity * mean[:, 3],
            ],
            axis=1,
        )
        combined_std = np.concatenate([std_pos, std_vel], axis=1)
        motion_cov = np.array([np.diag(np.square(s)) for s in combined_std])

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the correction step given a new measurement (x, y, a, h)."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            (covariance @ self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurements: np.ndarray,
        only_position: bool = False,
        metric: str = "maha",
    ) -> np.ndarray:
        """Compute the gating distance between state and measurements.

        Returns an array of length len(measurements). A suitable threshold can
        be taken from :data:`CHI2INV95`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == "gaussian":
            return np.sum(d * d, axis=1)
        if metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
            )
            return np.sum(z * z, axis=0)
        raise ValueError(f"invalid distance metric: {metric!r}")
