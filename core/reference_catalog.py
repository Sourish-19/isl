"""
Reverse Communication: Text-to-Sign Reference Module.
Maps user-entered alphanumeric signs, words, and everyday phrases to authentic,
verified Indian Sign Language (ISL) demonstration images and signing guidance.
"""

from typing import List, Dict, Optional, Any
import os


class ISLReferenceCatalog:
    """
    Provides reverse dictionary lookup from text to ISL visual demonstration guides.
    """

    SUPPORTED_SIGNS: List[str] = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
    ]

    SIGN_DESCRIPTIONS: Dict[str, str] = {
        'A': "Both hands joined forming a triangular apex with fingertips meeting, thumbs pointing inward.",
        'B': "Both hands mirror each other with thumbs and index fingers curved forming symmetrical loops.",
        'C': "Dominant hand forms a curved 'C' arc facing sideways toward the opposite side.",
        'D': "Index finger extended vertically, thumb and other fingers forming a base circle.",
        'E': "Fingers curved forward with knuckles prominent.",
        'F': "Index finger of dominant hand crossed horizontally over non-dominant index finger.",
        'G': "Both fists clenched with dominant fist positioned alongside the other.",
        'H': "Open palms horizontal with dominant fingers stroking across non-dominant palm.",
        'I': "Pinky finger extended upright while all other fingers remain closed.",
        'J': "Pinky finger extended upright tracing a downward curved hook.",
        'K': "Dominant index and middle finger pointing up in a 'V' around thumb base.",
        'L': "Thumb and index finger held perpendicular to form an 'L' shape.",
        'M': "Three middle fingers extended downward over thumb.",
        'N': "Two middle fingers extended downward over thumb.",
        'O': "All fingertips touching the thumb tip to form a closed circle.",
        'P': "Index finger extended downward with thumb extended horizontally.",
        'Q': "Thumb and index finger pointing downward forming a downward hook.",
        'R': "Index and middle fingers crossed tightly.",
        'S': "Both forearms crossed horizontally in front of the chest.",
        'T': "Dominant index finger touching non-dominant index finger perpendicular to form a 'T'.",
        'U': "Index and middle fingers held straight together pointing upward.",
        'V': "Index and middle fingers spread apart pointing upward in a 'V' victory shape.",
        'W': "Both hands held open with fingers interlocking or spread in a 'W'.",
        'X': "Both index fingers crossed horizontally forming an 'X'.",
        'Y': "Thumb and pinky finger extended outward, other three fingers closed.",
        'Z': "Dominant index finger pointing horizontally across body toward non-dominant palm.",
        '1': "Single index finger extended upright.",
        '2': "Index and middle finger extended upright.",
        '3': "Thumb, index, and middle fingers extended.",
        '4': "Four fingers extended upright with thumb folded inward.",
        '5': "All five fingers fully spread and extended open.",
        '6': "Thumb and pinky touching, three middle fingers extended.",
        '7': "Thumb and ring finger touching, index, middle, and pinky extended.",
        '8': "Middle finger folded down to thumb, others extended.",
        '9': "Index finger curved to touch thumb, other three fingers extended."
    }

    COMMON_PHRASE_GUIDES: Dict[str, Dict[str, Any]] = {
        "HELLO": {
            "meaning": "Friendly greeting (Open dominant hand raised to forehead/temple and moved forward like a salute).",
            "spelling": ["H", "E", "L", "L", "O"],
            "type": "Courtesy Sign"
        },
        "THANK YOU": {
            "meaning": "Gratitude sign (Flat hand touching chin/lips and extending forward towards recipient).",
            "spelling": ["T", "H", "A", "N", "K", "Y", "O", "U"],
            "type": "Courtesy Sign"
        },
        "PLEASE": {
            "meaning": "Polite request (Open palm rubbed in a circular motion over center of chest).",
            "spelling": ["P", "L", "E", "A", "S", "E"],
            "type": "Courtesy Sign"
        },
        "SORRY": {
            "meaning": "Apology sign (Fist in 'A' or 'S' shape rubbed in a circular motion over heart/chest).",
            "spelling": ["S", "O", "R", "R", "Y"],
            "type": "Courtesy Sign"
        },
        "YES": {
            "meaning": "Affirmation (Fist nodding up and down from the wrist like a nodding head).",
            "spelling": ["Y", "E", "S"],
            "type": "Response Sign"
        },
        "NO": {
            "meaning": "Negation (Index and middle fingers snapping down onto thumb).",
            "spelling": ["N", "O"],
            "type": "Response Sign"
        },
        "HELP ME": {
            "meaning": "Assistance request (Closed fist with thumb up resting on flat palm of other hand, moving upward toward chest).",
            "spelling": ["H", "E", "L", "P", "M", "E"],
            "type": "Emergency / Essential Sign"
        },
        "WATER": {
            "meaning": "Essential need ('W' handshape tapped twice near side of chin or mouth).",
            "spelling": ["W", "A", "T", "E", "R"],
            "type": "Essential Sign"
        }
    }

    @classmethod
    def lookup_sign(cls, char: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves reference details and image URL for a single alphanumeric sign.
        """
        clean_char = char.strip().upper()
        if clean_char not in cls.SUPPORTED_SIGNS:
            return None

        return {
            "sign": clean_char,
            "image_url": f"/static/reference/{clean_char}.png",
            "description": cls.SIGN_DESCRIPTIONS.get(clean_char, "Verified Indian Sign Language gesture."),
            "is_supported": True
        }

    @classmethod
    def lookup_phrase(cls, text: str) -> Dict[str, Any]:
        """
        Retrieves visual fingerspelling sequence and dictionary guidance for a phrase or word.
        """
        clean_text = text.strip().upper()
        phrase_info = cls.COMMON_PHRASE_GUIDES.get(clean_text)

        # Build sequence of valid character images
        sequence = []
        for ch in clean_text:
            if ch in cls.SUPPORTED_SIGNS:
                sequence.append({
                    "char": ch,
                    "image_url": f"/static/reference/{ch}.png",
                    "description": cls.SIGN_DESCRIPTIONS.get(ch, "")
                })

        return {
            "query": clean_text,
            "phrase_metadata": phrase_info,
            "visual_sequence": sequence,
            "total_signs": len(sequence)
        }

    @classmethod
    def get_all_reference_gallery(cls) -> List[Dict[str, Any]]:
        """
        Returns full catalog of all 35 ISL demonstration gestures for the frontend reference gallery.
        """
        gallery = []
        for sign in cls.SUPPORTED_SIGNS:
            gallery.append({
                "sign": sign,
                "image_url": f"/static/reference/{sign}.png",
                "description": cls.SIGN_DESCRIPTIONS.get(sign, "")
            })
        return gallery
