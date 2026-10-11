"""
FastAPI REST and WebSocket API Routes for Indian Sign Language Recognition System.
Provides endpoints for model health, real-time prediction, phrase buffer manipulation,
and reverse text-to-sign reference lookup.
"""

from typing import List, Dict, Optional, Any
import base64
import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from core.landmark_processor import LandmarkProcessor
from core.model_engine import ISLModelEngine
from core.phrase_buffer import PhraseBuffer
from core.reference_catalog import ISLReferenceCatalog
from core.vocabulary import VocabularyRegistry

router = APIRouter()

# Global instances initialized on app startup
model_engine: Optional[ISLModelEngine] = None
phrase_buffer: Optional[PhraseBuffer] = None


def init_api_dependencies(engine: ISLModelEngine, buffer: PhraseBuffer):
    global model_engine, phrase_buffer
    model_engine = engine
    phrase_buffer = buffer


# --- Pydantic Request & Response Schemas ---

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes_count: int
    input_dimension: int
    supported_classes: List[str]
    validated_classes: List[str]


class PredictLandmarksRequest(BaseModel):
    landmarks: List[List[float]] = Field(..., description="21 landmark coordinate pairs [[x, y], ...]")
    confidence_threshold: Optional[float] = Field(0.60, ge=0.0, le=1.0)


class PredictImageRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded JPEG/PNG webcam frame")
    confidence_threshold: Optional[float] = Field(0.60, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    label: str
    raw_label: str
    confidence: float
    is_confident: bool
    class_index: int
    is_empirically_validated: bool
    top_candidates: List[Dict[str, Any]]


class PhraseActionRequest(BaseModel):
    action: str = Field(..., description="Action: 'space', 'backspace', 'clear', 'add_char', 'add_phrase'")
    value: Optional[str] = Field(None, description="Character or phrase to insert")


class PhraseStateResponse(BaseModel):
    text: str
    current_word: str
    completed_words: List[str]
    current_candidate: Optional[str]
    stability_progress: float
    suggestions: List[str]
    history_length: int


# --- API Endpoints ---

@router.get("/health", response_model=HealthResponse)
def get_health():
    """Returns model health, readiness status, and supported class count."""
    if not model_engine or not model_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model engine not loaded")

    return HealthResponse(
        status="healthy",
        model_loaded=model_engine.is_loaded,
        classes_count=model_engine.NUM_CLASSES,
        input_dimension=model_engine.INPUT_DIM,
        supported_classes=model_engine.get_supported_classes(),
        validated_classes=VocabularyRegistry.EMPIRICALLY_VALIDATED_CLASSES
    )


@router.get("/system/capabilities")
def get_system_capabilities():
    """Provides a clear breakdown of validated recognition signs vs helper UI shortcuts."""
    return VocabularyRegistry.get_capabilities_summary()


@router.get("/classes")
def get_classes():
    """Returns list of supported alphanumeric sign classes and common phrases."""
    if not model_engine:
        raise HTTPException(status_code=503, detail="Model engine not initialized")

    return {
        "alphanumeric_classes": model_engine.get_supported_classes(),
        "validated_classes": VocabularyRegistry.EMPIRICALLY_VALIDATED_CLASSES,
        "total_classes": model_engine.NUM_CLASSES,
        "ui_shortcut_phrases": PhraseBuffer.DEFAULT_COMMON_PHRASES,
        "common_phrases": PhraseBuffer.DEFAULT_COMMON_PHRASES
    }


@router.post("/predict/landmarks", response_model=PredictionResponse)
def predict_landmarks(payload: PredictLandmarksRequest):
    """
    Evaluates 21 landmark coordinate pairs [x, y], applies normalization,
    and returns predicted ISL sign label and confidence.
    """
    if not model_engine or not model_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model engine not loaded")

    if len(payload.landmarks) != 21:
        raise HTTPException(status_code=400, detail=f"Expected 21 landmarks, received {len(payload.landmarks)}")

    try:
        # Preprocess: wrist origin translation + max-abs scaling
        normalized_features = LandmarkProcessor.pre_process_landmark(payload.landmarks)
        # Invariant neural inference with anatomical validation
        result = model_engine.predict_invariant(
            normalized_features,
            landmarks_list=payload.landmarks,
            confidence_threshold=payload.confidence_threshold
        )
        raw_lbl = result["raw_label"]
        result["is_empirically_validated"] = raw_lbl in VocabularyRegistry.EMPIRICALLY_VALIDATED_CLASSES
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


_hands_detector = None


def get_hands_detector():
    """Lazily initializes and reuses a persistent MediaPipe Hands instance."""
    global _hands_detector
    if _hands_detector is None:
        import mediapipe as mp
        _hands_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.35,
            min_tracking_confidence=0.35
        )
    return _hands_detector


@router.post("/predict/image")
def predict_image(payload: PredictImageRequest):
    """
    Decodes base64 webcam frame, runs MediaPipe hand landmark extraction,
    and returns detected signs and landmark points.
    """
    if not model_engine or not model_engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model engine not loaded")

    try:
        # Decode base64
        data = payload.image_base64
        if "," in data:
            data = data.split(",", 1)[1]
        img_bytes = base64.b64decode(data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image data")

        h, w, _ = img.shape

        # Mirror frame horizontally to align with selfie perspective and training data
        img_flipped = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB)

        detector = get_hands_detector()
        results = detector.process(img_rgb)

        if not results.multi_hand_landmarks:
            return {
                "hand_detected": False,
                "prediction": None,
                "landmarks": [],
                "message": "No hand detected in frame"
            }

        # Evaluate all detected hands and select the highest confidence prediction
        best_pred = None
        highest_conf = -1.0

        for hand_landmarks in results.multi_hand_landmarks:
            landmarks_list = LandmarkProcessor.calc_landmark_list(w, h, hand_landmarks)
            normalized_features = LandmarkProcessor.pre_process_landmark(landmarks_list)
            pred = model_engine.predict_invariant(
                normalized_features,
                landmarks_list=landmarks_list,
                confidence_threshold=payload.confidence_threshold
            )
            pred["is_empirically_validated"] = pred["raw_label"] in VocabularyRegistry.EMPIRICALLY_VALIDATED_CLASSES

            if pred["confidence"] > highest_conf:
                highest_conf = pred["confidence"]
                best_pred = pred

        # Extract landmarks for frontend visualization (converted back to original camera coordinate space)
        raw_pts = []
        for hand in results.multi_hand_landmarks:
            for lm in hand.landmark:
                raw_pts.append({
                    "x": 1.0 - lm.x,
                    "y": lm.y,
                    "z": lm.z
                })

        return {
            "hand_detected": True,
            "prediction": best_pred,
            "landmarks": raw_pts,
            "message": "Hand successfully tracked"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image inference error: {str(e)}")


# --- Phrase Buffer Endpoints ---

@router.get("/phrase/state", response_model=PhraseStateResponse)
def get_phrase_state():
    """Returns the current state of the sentence construction buffer."""
    if not phrase_buffer:
        raise HTTPException(status_code=503, detail="Phrase buffer not initialized")
    return PhraseStateResponse(**phrase_buffer.get_state())


@router.post("/phrase/action", response_model=PhraseStateResponse)
def execute_phrase_action(payload: PhraseActionRequest):
    """
    Executes a sentence-editing command on the phrase buffer:
    - 'space': Appends word boundary space
    - 'backspace': Deletes last letter or word
    - 'clear': Wipes the buffer
    - 'add_char': Inserts letter
    - 'add_phrase': Inserts complete phrase (e.g. 'THANK YOU')
    """
    if not phrase_buffer:
        raise HTTPException(status_code=503, detail="Phrase buffer not initialized")

    action = payload.action.lower()
    if action == "space":
        phrase_buffer.add_space()
    elif action == "backspace":
        phrase_buffer.backspace()
    elif action == "clear":
        phrase_buffer.clear()
    elif action == "add_char":
        if payload.value:
            phrase_buffer.add_character(payload.value)
    elif action == "add_phrase":
        if payload.value:
            phrase_buffer.add_phrase(payload.value)
    else:
        raise HTTPException(status_code=400, detail=f"Unrecognized action: {action}")

    return PhraseStateResponse(**phrase_buffer.get_state())


@router.post("/phrase/process_frame")
def process_frame_stream(prediction: Dict[str, Any]):
    """
    Feeds a real-time frame prediction into the temporal smoothing and debounce buffer.
    Triggers character commitment when held steadily.
    """
    if not phrase_buffer:
        raise HTTPException(status_code=503, detail="Phrase buffer not initialized")

    status = phrase_buffer.process_frame_prediction(prediction)
    return status


# --- Reverse Text-to-Sign Reference Endpoints ---

@router.get("/reference/sign/{char}")
def lookup_sign_reference(char: str):
    """Retrieves verified ISL visual demonstration for a single alphanumeric sign."""
    ref = ISLReferenceCatalog.lookup_sign(char)
    if not ref:
        raise HTTPException(status_code=404, detail=f"Sign '{char}' not found in ISL catalog")
    return ref


@router.get("/reference/phrase")
def lookup_phrase_reference(text: str = Query(..., description="Word or phrase to look up")):
    """Retrieves visual fingerspelling sequence and signing guidance for a phrase."""
    return ISLReferenceCatalog.lookup_phrase(text)


@router.get("/reference/gallery")
def get_reference_gallery():
    """Returns all 35 authentic ISL gesture demonstrations and descriptions."""
    raw_gallery = ISLReferenceCatalog.get_all_reference_gallery()
    for item in raw_gallery:
        item["is_empirically_validated"] = item["sign"] in VocabularyRegistry.EMPIRICALLY_VALIDATED_CLASSES

    return {
        "total_signs": len(ISLReferenceCatalog.SUPPORTED_SIGNS),
        "validated_signs": VocabularyRegistry.EMPIRICALLY_VALIDATED_CLASSES,
        "gallery": raw_gallery
    }
