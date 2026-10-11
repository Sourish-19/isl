"""
Unit Tests for Model Engine, Forward Pass, and Confidence Handling.
"""

import pytest
import numpy as np
from core.model_engine import ISLModelEngine


@pytest.fixture
def engine():
    return ISLModelEngine()


def test_model_loading_and_classes(engine):
    assert engine.is_loaded is True
    classes = engine.get_supported_classes()
    assert len(classes) == 35
    assert classes[:9] == ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    assert classes[9:12] == ['A', 'B', 'C']
    assert classes[-1] == 'Z'


def test_forward_pass_dimensions_and_probabilities(engine):
    dummy_input = np.zeros((1, 42), dtype=np.float32)
    probs = engine.forward(dummy_input)

    assert probs.shape == (1, 35)
    # Probabilities must sum to 1.0
    assert np.sum(probs) == pytest.approx(1.0, rel=1e-5)
    assert np.all(probs >= 0.0)


def test_predict_structure_and_confidence_threshold(engine):
    dummy_features = [0.0] * 42
    # High confidence threshold
    pred = engine.predict(dummy_features, confidence_threshold=0.999)

    assert "label" in pred
    assert "confidence" in pred
    assert "is_confident" in pred
    assert "top_candidates" in pred
    assert len(pred["top_candidates"]) == 3
    # When threshold is not met, label should be 'Unrecognized'
    if not pred["is_confident"]:
        assert pred["label"] == "Unrecognized"


def test_invalid_feature_dimensions(engine):
    with pytest.raises(ValueError):
        engine.predict([0.1, 0.2, 0.3])  # Only 3 features instead of 42
