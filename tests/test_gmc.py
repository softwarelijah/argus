import numpy as np

from argus.tracking.gmc import GMC
from argus.tracking.track import STrack


def _textured_frame(seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(0, 255, size=(240, 320, 3))).astype(np.uint8)


def test_gmc_none_returns_identity():
    gmc = GMC("none")
    warp = gmc.apply(_textured_frame())
    assert np.allclose(warp, np.eye(2, 3))


def test_gmc_first_frame_is_identity():
    gmc = GMC("orb", downscale=1)
    warp = gmc.apply(_textured_frame())
    assert np.allclose(warp, np.eye(2, 3))


def test_gmc_orb_recovers_translation():
    gmc = GMC("orb", downscale=1)
    base = _textured_frame(seed=1)
    gmc.apply(base)  # prime with first frame

    shift = 12
    shifted = np.roll(base, shift, axis=1)  # shift right by `shift` px
    warp = gmc.apply(shifted)
    # estimated x translation should be close to the applied shift
    assert abs(warp[0, 2] - shift) < 3.0
    assert abs(warp[1, 2]) < 3.0


def test_apply_gmc_shifts_track_state():
    track = STrack(np.array([100.0, 100.0, 20.0, 40.0]), 0.9)
    from argus.tracking.kalman_filter import KalmanFilter

    track.activate(KalmanFilter(), frame_id=1)
    cx_before, cy_before = track.mean[0], track.mean[1]

    warp = np.array([[1.0, 0.0, 25.0], [0.0, 1.0, -10.0]], dtype=np.float32)
    track.apply_gmc(warp)
    assert np.isclose(track.mean[0], cx_before + 25.0)
    assert np.isclose(track.mean[1], cy_before - 10.0)


def test_invalid_method_raises():
    try:
        GMC("bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown method")
