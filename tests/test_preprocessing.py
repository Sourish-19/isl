"""
Unit Tests for Landmark Preprocessing and Spatial Invariance.
"""

import pytest
import numpy as np
from core.landmark_processor import LandmarkProcessor


def test_calc_landmark_list_dimensions():
    # Synthetic landmark objects
    class DummyLandmark:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    landmarks = [DummyLandmark(i / 20.0, (20 - i) / 20.0) for i in range(21)]
    pixel_coords = LandmarkProcessor.calc_landmark_list(640, 480, landmarks)

    assert len(pixel_coords) == 21
    assert all(len(pt) == 2 for pt in pixel_coords)
    assert 0 <= pixel_coords[0][0] < 640
    assert 0 <= pixel_coords[0][1] < 480


def test_preprocessing_translation_and_normalization():
    # Construct synthetic 21 points
    pts = [[100 + i * 2, 200 + i * 3] for i in range(21)]
    features = LandmarkProcessor.pre_process_landmark(pts)

    assert len(features) == 42
    # Wrist (first point) relative coords should be exactly 0.0, 0.0
    assert features[0] == 0.0
    assert features[1] == 0.0
    # All normalized coordinates must be strictly within [-1.0, 1.0]
    assert all(-1.0 <= f <= 1.0 for f in features)
    # The max absolute value must be exactly 1.0
    assert max(map(abs, features)) == pytest.approx(1.0, rel=1e-5)


def test_translation_invariance():
    # Points placed at position A
    pts_A = [[50 + i * 2, 80 + i * 3] for i in range(21)]
    # Points translated by (dx=300, dy=150)
    pts_B = [[p[0] + 300, p[1] + 150] for p in pts_A]

    features_A = LandmarkProcessor.pre_process_landmark(pts_A)
    features_B = LandmarkProcessor.pre_process_landmark(pts_B)

    # Shifted hand should produce identical normalized feature vectors
    np.testing.assert_allclose(features_A, features_B, atol=1e-5)


def test_scale_invariance():
    pts_base = [[i * 3, i * 4] for i in range(21)]
    # Scale points by 2.5x (e.g. hand closer to camera)
    pts_scaled = [[p[0] * 2.5, p[1] * 2.5] for p in pts_base]

    feat_base = LandmarkProcessor.pre_process_landmark(pts_base)
    feat_scaled = LandmarkProcessor.pre_process_landmark(pts_scaled)

    np.testing.assert_allclose(feat_base, feat_scaled, atol=1e-5)


def test_invalid_landmark_count():
    with pytest.raises(ValueError):
        LandmarkProcessor.pre_process_landmark([[0, 0]] * 10)
