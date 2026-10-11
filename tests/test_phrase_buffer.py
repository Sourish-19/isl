"""
Unit Tests for Phrase Construction Buffer, Temporal Debouncing, and Suggestions.
"""

import pytest
from core.phrase_buffer import PhraseBuffer


@pytest.fixture
def buffer():
    return PhraseBuffer(window_size=4, hold_threshold=5, cooldown_frames=5, confidence_threshold=0.70)


def test_phrase_buffer_manual_operations(buffer):
    buffer.add_character('H')
    buffer.add_character('I')
    assert buffer.get_full_text() == "HI"

    buffer.add_space()
    assert buffer.get_full_text() == "HI"

    buffer.add_character('Y')
    buffer.add_character('O')
    buffer.add_character('U')
    assert buffer.get_full_text() == "HI YOU"

    # Backspace deletes 'U'
    buffer.backspace()
    assert buffer.get_full_text() == "HI YO"

    # Clear deletes everything
    buffer.clear()
    assert buffer.get_full_text() == ""


def test_hold_to_commit_and_duplicate_suppression(buffer):
    pred = {"raw_label": "A", "confidence": 0.95}

    # Frame 1 to 4: holding 'A'
    for _ in range(4):
        res = buffer.process_frame_prediction(pred)
        assert res["text"] == ""  # Not committed yet

    # Frame 5: reaches hold_threshold (5) -> commits 'A'
    res = buffer.process_frame_prediction(pred)
    assert res["text"] == "A"
    assert res["event"] == "Committed 'A'"

    # Frame 6 to 10: continuing to hold 'A' should NOT duplicate type 'A'
    for _ in range(5):
        res = buffer.process_frame_prediction(pred)
        assert res["text"] == "A"  # Still just "A", no spam!


def test_low_confidence_rejection(buffer):
    pred = {"raw_label": "B", "confidence": 0.30}  # Below threshold (0.70)
    for _ in range(10):
        res = buffer.process_frame_prediction(pred)
        assert res["text"] == ""


def test_phrase_suggestions(buffer):
    buffer.add_character('H')
    buffer.add_character('E')
    buffer.add_character('L')
    suggestions = buffer.get_suggestions()

    assert "HELLO" in suggestions
