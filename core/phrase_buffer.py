"""
Phrase Construction, Temporal Smoothing, and Gesture Debounce Buffer.
Provides smooth word-and-sentence assembly from real-time webcam predictions.
Prevents duplicate character spamming while holding a gesture, maintains
sentence text state, offers dictionary auto-suggestions, and manages common ISL phrases.
"""

from typing import List, Dict, Optional, Any
from collections import deque
import time


class PhraseBuffer:
    """
    Manages temporal aggregation of real-time sign predictions into coherent sentences.
    Features:
    - Moving window smoothing / majority voting
    - Hold-to-commit debounce mechanism (prevents jitter and duplicate firing)
    - Full sentence construction (characters, spaces, words, undo/clear)
    - Suggestion engine for target ISL phrases
    """

    DEFAULT_COMMON_PHRASES: List[str] = [
        "HELLO",
        "THANK YOU",
        "PLEASE",
        "SORRY",
        "YES",
        "NO",
        "HELP ME",
        "HOW ARE YOU",
        "WHAT IS YOUR NAME",
        "I NEED WATER",
        "I NEED HELP",
        "GOOD MORNING",
        "WELCOME",
        "NICE TO MEET YOU",
        "TAKE CARE"
    ]

    def __init__(
        self,
        window_size: int = 8,
        hold_threshold: int = 14,
        cooldown_frames: int = 16,
        confidence_threshold: float = 0.70
    ):
        self.window_size = window_size
        self.hold_threshold = hold_threshold
        self.cooldown_frames = cooldown_frames
        self.confidence_threshold = confidence_threshold

        # Sliding window for prediction smoothing
        self.prediction_window: deque = deque(maxlen=window_size)

        # State tracking for gesture stability
        self.current_candidate: Optional[str] = None
        self.stable_frame_count: int = 0
        self.cooldown_remaining: int = 0
        self.last_committed_char: Optional[str] = None

        # Sentence buffer state
        self.current_word: str = ""
        self.completed_words: List[str] = []
        self.history: List[Dict[str, Any]] = []

        # Suggestion / dictionary bank
        self.common_phrases = list(self.DEFAULT_COMMON_PHRASES)

    def reset_stability(self) -> None:
        """Resets the temporal stability counter."""
        self.current_candidate = None
        self.stable_frame_count = 0
        self.prediction_window.clear()

    def process_frame_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingests a single frame prediction, applies smoothing and debounce rules.
        Returns the updated system status and triggers character commitment when held.
        """
        raw_label = prediction.get("raw_label", "")
        confidence = prediction.get("confidence", 0.0)
        is_confident = confidence >= self.confidence_threshold

        event_triggered = None

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

        if is_confident and raw_label != "Unrecognized":
            self.prediction_window.append(raw_label)
        else:
            self.prediction_window.append(None)

        # Smooth prediction by majority vote in sliding window
        valid_votes = [lbl for lbl in self.prediction_window if lbl is not None]
        smoothed_label = None
        if valid_votes:
            # Most frequent label in current window
            smoothed_label = max(set(valid_votes), key=valid_votes.count)
            # Require at least 70% consistency across the sliding window
            smoothed_ratio = valid_votes.count(smoothed_label) / float(len(self.prediction_window))
            if smoothed_ratio < 0.70:
                smoothed_label = None

        # Stability counter and hold-to-commit logic
        if smoothed_label and self.cooldown_remaining == 0:
            if smoothed_label == self.current_candidate:
                self.stable_frame_count += 1
            else:
                self.current_candidate = smoothed_label
                self.stable_frame_count = 1

            # Check if gesture has been held steadily long enough to commit
            if self.stable_frame_count >= self.hold_threshold:
                # Do not re-commit the same letter repeatedly without releasing hand
                if smoothed_label != self.last_committed_char:
                    self.add_character(smoothed_label)
                    event_triggered = f"Committed '{smoothed_label}'"
                    self.last_committed_char = smoothed_label
                    self.cooldown_remaining = self.cooldown_frames
                    self.stable_frame_count = 0
        else:
            if not is_confident:
                # Hand was released or moved out of frame - reset candidate and allow re-commitment
                self.current_candidate = None
                self.stable_frame_count = 0
                self.last_committed_char = None
            else:
                self.stable_frame_count = max(0, self.stable_frame_count - 1)

        progress_pct = min(1.0, self.stable_frame_count / float(self.hold_threshold)) if self.hold_threshold > 0 else 0.0

        return {
            "current_candidate": self.current_candidate,
            "smoothed_label": smoothed_label,
            "confidence": confidence,
            "stability_progress": round(progress_pct, 2),
            "event": event_triggered,
            "text": self.get_full_text(),
            "current_word": self.current_word,
            "suggestions": self.get_suggestions()
        }

    def add_character(self, char: str) -> None:
        """Appends a single character to the active word."""
        if not char:
            return
        char_upper = char.upper()
        self.current_word += char_upper
        self.history.append({
            "action": "ADD_CHAR",
            "value": char_upper,
            "timestamp": time.time(),
            "resulting_text": self.get_full_text()
        })

    def add_space(self) -> None:
        """Finalizes the active word and appends a space."""
        if self.current_word:
            self.completed_words.append(self.current_word)
            self.current_word = ""
            self.history.append({
                "action": "SPACE",
                "value": " ",
                "timestamp": time.time(),
                "resulting_text": self.get_full_text()
            })
        self.reset_stability()

    def backspace(self) -> None:
        """Deletes the last character or word."""
        if len(self.current_word) > 0:
            removed = self.current_word[-1]
            self.current_word = self.current_word[:-1]
            self.history.append({
                "action": "BACKSPACE_CHAR",
                "value": removed,
                "timestamp": time.time(),
                "resulting_text": self.get_full_text()
            })
        elif len(self.completed_words) > 0:
            # Re-open the last completed word for editing
            self.current_word = self.completed_words.pop()
            self.history.append({
                "action": "UNDO_WORD",
                "value": self.current_word,
                "timestamp": time.time(),
                "resulting_text": self.get_full_text()
            })
        self.reset_stability()

    def clear(self) -> None:
        """Clears the entire sentence buffer."""
        self.current_word = ""
        self.completed_words.clear()
        self.reset_stability()
        self.last_committed_char = None
        self.history.append({
            "action": "CLEAR",
            "value": "",
            "timestamp": time.time(),
            "resulting_text": ""
        })

    def add_phrase(self, phrase: str) -> None:
        """Directly inserts a complete phrase or suggested word."""
        if not phrase:
            return
        cleaned = phrase.strip().upper()
        # If there's an active incomplete word, replace or complete it
        words = cleaned.split()
        for w in words:
            if self.current_word:
                self.completed_words.append(self.current_word)
                self.current_word = ""
            self.completed_words.append(w)

        self.reset_stability()
        self.history.append({
            "action": "ADD_PHRASE",
            "value": cleaned,
            "timestamp": time.time(),
            "resulting_text": self.get_full_text()
        })

    def get_full_text(self) -> str:
        """Returns the complete sentence reconstructed from completed and active words."""
        parts = list(self.completed_words)
        if self.current_word:
            parts.append(self.current_word)
        return " ".join(parts)

    def get_suggestions(self, max_suggestions: int = 5) -> List[str]:
        """
        Returns phrase or word suggestions matching the current prefix.
        If no prefix is typed, returns top common emergency/courtesy phrases.
        """
        current = self.current_word.upper()
        if not current:
            return self.common_phrases[:max_suggestions]

        matching = [p for p in self.common_phrases if p.startswith(current)]
        if len(matching) < max_suggestions:
            # Add substring matches
            substring_matches = [p for p in self.common_phrases if current in p and p not in matching]
            matching.extend(substring_matches)

        return matching[:max_suggestions]

    def get_state(self) -> Dict[str, Any]:
        """Provides a serializable snapshot of the phrase buffer for UI/API synchronization."""
        return {
            "text": self.get_full_text(),
            "current_word": self.current_word,
            "completed_words": list(self.completed_words),
            "current_candidate": self.current_candidate,
            "stability_progress": round(min(1.0, self.stable_frame_count / float(self.hold_threshold)), 2),
            "suggestions": self.get_suggestions(),
            "history_length": len(self.history)
        }
