import numpy as np

from argus.tracking.kalman_filter import CHI2INV95, KalmanFilter


def test_initiate_sets_zero_velocity():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([100.0, 50.0, 0.5, 40.0]))
    assert mean.shape == (8,)
    assert cov.shape == (8, 8)
    # velocities start at zero
    assert np.allclose(mean[4:], 0.0)
    # position component matches the measurement
    assert np.allclose(mean[:4], [100.0, 50.0, 0.5, 40.0])


def test_predict_advances_position_by_velocity():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([100.0, 50.0, 0.5, 40.0]))
    mean[4] = 5.0  # vx
    mean[5] = -2.0  # vy
    pred_mean, _ = kf.predict(mean, cov)
    assert np.isclose(pred_mean[0], 105.0)
    assert np.isclose(pred_mean[1], 48.0)


def test_update_pulls_state_toward_measurement():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([100.0, 50.0, 0.5, 40.0]))
    mean, cov = kf.predict(mean, cov)
    new_mean, _ = kf.update(mean, cov, np.array([110.0, 50.0, 0.5, 40.0]))
    # corrected x should move from 100 toward the measured 110
    assert 100.0 < new_mean[0] <= 110.0


def test_multi_predict_matches_single_predict():
    kf = KalmanFilter()
    means, covs = [], []
    for i in range(5):
        m, c = kf.initiate(np.array([10.0 * i, 20.0, 0.5, 30.0]))
        m[4] = 1.0
        means.append(m)
        covs.append(c)
    batch_mean = np.stack(means)
    batch_cov = np.stack(covs)

    bm, bc = kf.multi_predict(batch_mean, batch_cov)
    for i in range(5):
        sm, sc = kf.predict(means[i], covs[i])
        assert np.allclose(bm[i], sm, atol=1e-5)
        assert np.allclose(bc[i], sc, atol=1e-4)


def test_gating_distance_zero_for_self():
    kf = KalmanFilter()
    mean, cov = kf.initiate(np.array([100.0, 50.0, 0.5, 40.0]))
    measurement = np.array([[100.0, 50.0, 0.5, 40.0]])
    dist = kf.gating_distance(mean, cov, measurement)
    assert dist[0] < CHI2INV95[4]
