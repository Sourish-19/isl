"""
Real-Time Indian Sign Language Detection and Sentence Construction Pipeline.
Upgraded with MediaPipe 3D Landmark Tracking, Vectorized Neural Inference,
Temporal Prediction Smoothing, Gesture Debouncing, and an Interactive On-Screen HUD.
"""

import os
import sys
import copy
import time
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

from core.landmark_processor import LandmarkProcessor
from core.model_engine import ISLModelEngine
from core.phrase_buffer import PhraseBuffer


def draw_hud(
    frame: np.ndarray,
    prediction_info: dict,
    phrase_buffer: PhraseBuffer,
    fps: float
) -> np.ndarray:
    """
    Renders an accessible, modern Heads-Up Display (HUD) overlay directly onto the video frame:
    - Top Status Bar: Detected sign, confidence percentage, stability meter, FPS.
    - Bottom Sentence Bar: Accumulated words/sentence with active typing indicator.
    - Control Shortcuts: Space, Backspace, Clear, Quit.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    # --- Top Header Overlay (Status & Confidence) ---
    header_height = 85
    cv2.rectangle(overlay, (0, 0), (w, header_height), (15, 23, 42), -1)  # Dark navy background

    # Transparency blend
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    label = prediction_info.get("smoothed_label") or "Searching..."
    conf = prediction_info.get("confidence", 0.0)
    stability = prediction_info.get("stability_progress", 0.0)

    # 1. Detected Sign Title
    cv2.putText(
        frame,
        f"SIGN: {label}",
        (20, 42),
        cv2.FONT_HERSHEY_DUPLEX,
        1.1,
        (56, 189, 248) if label != "Searching..." else (148, 163, 184), # Cyan or muted gray
        2,
        cv2.LINE_AA
    )

    # 2. Confidence Indicator
    conf_text = f"Confidence: {int(conf * 100)}%" if label != "Searching..." else "Stand by"
    cv2.putText(
        frame,
        conf_text,
        (20, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (226, 232, 240),
        1,
        cv2.LINE_AA
    )

    # 3. Stability Progress Bar (Hold-to-commit meter)
    bar_x = w - 240
    bar_y = 35
    bar_w = 200
    bar_h = 16
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (51, 65, 85), -1)
    fill_w = int(bar_w * stability)
    if fill_w > 0:
        bar_color = (34, 197, 94) if stability >= 0.95 else (14, 165, 233)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (148, 163, 184), 1)

    cv2.putText(
        frame,
        f"HOLD PROGRESS ({int(stability * 100)}%)",
        (bar_x, bar_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (203, 213, 225),
        1,
        cv2.LINE_AA
    )

    # FPS display
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (w - 75, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (100, 116, 139),
        1,
        cv2.LINE_AA
    )

    # --- Bottom Sentence Bar ---
    sentence_h = 90
    sentence_y = h - sentence_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, sentence_y), (w, h), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    full_text = phrase_buffer.get_full_text()
    display_text = f"> {full_text}" if full_text else "> [Perform sign to begin typing...]"
    text_color = (255, 255, 255) if full_text else (148, 163, 184)

    # Truncate text from left if it exceeds frame width
    if len(display_text) > 45:
        display_text = "> ..." + display_text[-42:]

    cv2.putText(
        frame,
        display_text,
        (20, sentence_y + 36),
        cv2.FONT_HERSHEY_DUPLEX,
        0.85,
        text_color,
        2,
        cv2.LINE_AA
    )

    # Control shortcuts legend
    controls_text = "HOTKEYS: [SPACE] Space  |  [BKSP] Undo  |  [C] Clear  |  [Q] Exit"
    cv2.putText(
        frame,
        controls_text,
        (20, sentence_y + 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (56, 189, 248),
        1,
        cv2.LINE_AA
    )

    return frame


def main():
    print("==================================================================")
    print("  INDIAN SIGN LANGUAGE (ISL) REAL-TIME RECOGNITION SYSTEM")
    print("  Powered by MediaPipe Hands and Deep Neural Network Engine")
    print("==================================================================")

    # Initialize neural model engine and phrase buffer
    try:
        model_engine = ISLModelEngine()
        print(f"Loaded ISL Model with {len(model_engine.get_supported_classes())} supported classes.")
    except Exception as e:
        print(f"[FATAL] Failed to initialize model engine: {e}")
        return

    phrase_buffer = PhraseBuffer(window_size=8, hold_threshold=14, cooldown_frames=16, confidence_threshold=0.65)

    # Initialize MediaPipe Hands solution
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Access webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera 0 could not be opened. Trying Camera index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[FATAL] No accessible webcam found. Please verify camera hardware/permissions.")
            return

    # Set camera resolution for crisp performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    fps = 30.0
    prev_time = time.time()

    print("[INFO] Starting real-time detection stream. Press 'q' or 'ESC' to exit.")

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("[WARNING] Empty frame received from webcam. Retrying...")
                time.sleep(0.05)
                continue

            current_time = time.time()
            fps = 1.0 / max(0.001, (current_time - prev_time))
            prev_time = current_time

            # Flip horizontally for selfie mirror perspective
            image = cv2.flip(image, 1)
            frame_h, frame_w, _ = image.shape

            # Convert BGR to RGB for MediaPipe inference
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            image.flags.writeable = True

            prediction_summary = {"smoothed_label": None, "confidence": 0.0, "stability_progress": 0.0}

            if results.multi_hand_landmarks:
                # Primary hand processing
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # 1. Compute pixel coordinates
                    landmark_list = LandmarkProcessor.calc_landmark_list(frame_w, frame_h, hand_landmarks)

                    # 2. Geometric translation and normalization
                    normalized_features = LandmarkProcessor.pre_process_landmark(landmark_list)

                    # 3. Neural inference
                    pred = model_engine.predict(normalized_features, confidence_threshold=0.60)

                    # 4. Temporal phrase buffer update
                    buf_status = phrase_buffer.process_frame_prediction(pred)

                    prediction_summary = {
                        "smoothed_label": buf_status.get("smoothed_label") or pred.get("label"),
                        "confidence": pred.get("confidence", 0.0),
                        "stability_progress": buf_status.get("stability_progress", 0.0)
                    }

                    # Draw hand skeleton connections
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Draw bounding box around hand
                    bbox = LandmarkProcessor.get_bounding_box(frame_w, frame_h, hand_landmarks)
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (56, 189, 248), 2)
                    hand_label = handedness.classification[0].label
                    cv2.putText(
                        image,
                        f"{hand_label}: {pred['label']}",
                        (bbox[0], max(20, bbox[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (34, 197, 94) if pred['is_confident'] else (239, 68, 68),
                        2,
                        cv2.LINE_AA
                    )
            else:
                phrase_buffer.reset_stability()

            # Render Heads-Up Display overlay
            rendered_frame = draw_hud(image, prediction_summary, phrase_buffer, fps)

            # Display window
            cv2.imshow("Indian Sign Language Recognition System", rendered_frame)

            # Keyboard event listener
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):
                break
            elif key == 32:  # Space key
                phrase_buffer.add_space()
            elif key == 8 or key == 127:  # Backspace
                phrase_buffer.backspace()
            elif key == ord('c') or key == ord('C'):  # Clear
                phrase_buffer.clear()

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed successfully.")
    print("Final Constructed Sentence:", phrase_buffer.get_full_text())


if __name__ == "__main__":
    main()
