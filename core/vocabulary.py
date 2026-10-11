"""
Vocabulary Registry and Sign Taxonomy for Indian Sign Language Recognition.
Defines explicit taxonomy between:
1. VALIDATED_SIGNS: Static alphanumeric signs with verified dataset samples in keypoint.csv.
2. ARCHITECTURE_CLASSES: Output neurons supported by the pre-trained neural network.
3. VOCABULARY_TARGETS: Configurable vocabulary of everyday ISL signs & words.
4. SHORTCUT_PHRASES: Predefined communication phrases for quick UI insertion.
"""

from typing import List, Dict, Any, Optional
from enum import Enum


class SignType(str, Enum):
    STATIC_ALPHABET = "static_alphabet"
    STATIC_NUMERIC = "static_numeric"
    STATIC_WORD = "static_word"
    DYNAMIC_GESTURE = "dynamic_gesture"
    SHORTCUT_PHRASE = "shortcut_phrase"


class VocabularyRegistry:
    """
    Central registry defining sign definitions, recognition status, and data availability.
    Prevents confusing UI phrase shortcuts with genuine ML-recognized signs.
    """

    # Ground-truth classes present in the repository's keypoint.csv
    EMPIRICALLY_VALIDATED_CLASSES: List[str] = ["A", "B", "C"]

    # All 35 output neurons present in model.h5
    MODEL_ARCHITECTURE_CLASSES: List[str] = [
        "1", "2", "3", "4", "5", "6", "7", "8", "9",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
    ]

    # Target vocabulary for common everyday ISL communication
    TARGET_VOCABULARY: Dict[str, Dict[str, Any]] = {
        # Validated static alphabets
        "A": {
            "type": SignType.STATIC_ALPHABET,
            "has_training_data": True,
            "validated": True,
            "description": "Both hands joined forming a triangular apex with fingertips meeting."
        },
        "B": {
            "type": SignType.STATIC_ALPHABET,
            "has_training_data": True,
            "validated": True,
            "description": "Both hands mirror each other with thumbs and index fingers forming symmetrical loops."
        },
        "C": {
            "type": SignType.STATIC_ALPHABET,
            "has_training_data": True,
            "validated": True,
            "description": "Dominant hand forms a curved 'C' arc facing sideways."
        },

        # Target static words (Awaiting dataset collection)
        "YES": {
            "type": SignType.STATIC_WORD,
            "has_training_data": False,
            "validated": False,
            "description": "Closed fist held upright, tilting slightly forward from the wrist."
        },
        "NO": {
            "type": SignType.STATIC_WORD,
            "has_training_data": False,
            "validated": False,
            "description": "Index and middle fingers snapping downward onto thumb."
        },
        "WATER": {
            "type": SignType.STATIC_WORD,
            "has_training_data": False,
            "validated": False,
            "description": "'W' handshape tapped twice near side of chin or mouth."
        },

        # Target dynamic gestures (Require temporal sequence data)
        "HELLO": {
            "type": SignType.DYNAMIC_GESTURE,
            "has_training_data": False,
            "validated": False,
            "description": "Open flat hand starting at forehead/temple, moving forward and outward."
        },
        "THANK YOU": {
            "type": SignType.DYNAMIC_GESTURE,
            "has_training_data": False,
            "validated": False,
            "description": "Flat palm touching chin and moving forward toward recipient."
        },
        "PLEASE": {
            "type": SignType.DYNAMIC_GESTURE,
            "has_training_data": False,
            "validated": False,
            "description": "Open flat palm rubbed in a circular motion on center of chest."
        },
        "SORRY": {
            "type": SignType.DYNAMIC_GESTURE,
            "has_training_data": False,
            "validated": False,
            "description": "Fist rubbed in a circular motion over heart."
        },
        "HELP ME": {
            "type": SignType.DYNAMIC_GESTURE,
            "has_training_data": False,
            "validated": False,
            "description": "Closed fist with thumb up resting on flat palm of other hand, moving upward."
        }
    }

    # Predefined UI communication shortcuts (Buttons in UI to assist non-verbal typing)
    UI_SHORTCUT_PHRASES: List[str] = [
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
        "TAKE CARE"
    ]

    @classmethod
    def get_sign_status(cls, sign_name: str) -> Dict[str, Any]:
        """Returns the validation status, type, and data availability for a given sign."""
        upper = sign_name.strip().upper()
        if upper in cls.TARGET_VOCABULARY:
            entry = dict(cls.TARGET_VOCABULARY[upper])
            entry["name"] = upper
            return entry

        if upper in cls.MODEL_ARCHITECTURE_CLASSES:
            is_val = upper in cls.EMPIRICALLY_VALIDATED_CLASSES
            return {
                "name": upper,
                "type": SignType.STATIC_NUMERIC if upper.isdigit() else SignType.STATIC_ALPHABET,
                "has_training_data": is_val,
                "validated": is_val,
                "description": f"Indian Sign Language sign '{upper}'."
            }

        return {
            "name": upper,
            "type": SignType.SHORTCUT_PHRASE if upper in cls.UI_SHORTCUT_PHRASES else "unregistered",
            "has_training_data": False,
            "validated": False,
            "description": "Custom communication phrase or unregistered sign."
        }

    @classmethod
    def get_capabilities_summary(cls) -> Dict[str, Any]:
        """Provides an honest summary of current model capabilities vs target vocabulary."""
        return {
            "empirically_validated_classes": cls.EMPIRICALLY_VALIDATED_CLASSES,
            "model_architecture_classes": cls.MODEL_ARCHITECTURE_CLASSES,
            "ui_shortcut_phrases": cls.UI_SHORTCUT_PHRASES,
            "target_vocabulary_count": len(cls.TARGET_VOCABULARY),
            "dynamic_gesture_sequence_support": False,
            "recognition_mode": "Static 2D Hand Landmark Spatial Classification",
            "disclaimer": (
                "The recognition model evaluates 42 static hand joint coordinates. "
                "Only classes 'A', 'B', and 'C' currently possess ground-truth test samples "
                "in keypoint.csv. Multi-word phrases in the UI are helper shortcuts for quick "
                "sentence composition and are not classified as continuous video gestures."
            )
        }
