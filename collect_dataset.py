"""
Reproducible Dataset Collection Utility for Indian Sign Language Recognition.
Allows users to capture real hand landmarks from a webcam for any alphanumeric sign or custom word.
Validates 42D feature dimensions, enforces geometric normalization, prevents data corruption,
and integrates directly with train_classifier.py without editing source code.
"""

import os
import sys
import time
import csv
import argparse
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from core.landmark_processor import LandmarkProcessor


def inspect_dataset(data_path: str = "keypoint.csv") -> Dict[str, Any]:
    """
    Analyzes keypoint.csv, reports class distribution, dimensions, and data health.
    """
    if not os.path.exists(data_path):
        print(f"[INFO] Dataset '{data_path}' does not exist yet.")
        return {"exists": False, "total_samples": 0, "classes": {}}

    df = pd.read_csv(data_path, header=None)
    total_samples, total_cols = df.shape
    features_count = total_cols - 1

    counts = df[0].value_counts().to_dict()
    nulls = int(df.isnull().sum().sum())

    print("\n================== DATASET HEALTH SUMMARY ==================")
    print(f"  File Path:          {data_path}")
    print(f"  Total Samples:      {total_samples}")
    print(f"  Feature Columns:    {features_count} (Expected: 42)")
    print(f"  Missing / NaN Rows: {nulls}")
    print("  Class Distribution:")
    for cls_name, cnt in sorted(counts.items()):
        print(f"    Class '{cls_name}': {cnt:>5} samples")
    print("============================================================\n")

    return {
        "exists": True,
        "total_samples": total_samples,
        "feature_count": features_count,
        "null_count": nulls,
        "classes": counts
    }


def identify_underrepresented_classes(
    data_path: str = "keypoint.csv",
    target_count: int = 500
) -> List[Tuple[str, int]]:
    """
    Identifies classes with fewer samples than target_count.
    """
    summary = inspect_dataset(data_path)
    if not summary["exists"]:
        return []

    classes = summary["classes"]
    underrepresented = []
    for cls_name, count in classes.items():
        if count < target_count:
            underrepresented.append((cls_name, target_count - count))

    if underrepresented:
        print(f"[NOTICE] Classes needing more samples (Target: {target_count}):")
        for cls_name, needed in underrepresented:
            print(f"  - Class '{cls_name}': needs {needed} more samples")
    else:
        print(f"[OK] All present classes meet or exceed {target_count} samples.")

    return underrepresented


def capture_sign_samples(
    class_name: str,
    target_samples: int = 50,
    data_path: str = "keypoint.csv",
    camera_index: int = 0,
    session_id: Optional[str] = None
) -> int:
    """
    Launches webcam capture to collect genuine MediaPipe hand landmark samples for a specified class.
    Validates features and appends rows to data_path.
    """
    class_name = class_name.strip().upper()
    if not class_name:
        print("[ERROR] Class name cannot be empty.")
        return 0

    print(f"\n[INFO] Initializing webcam capture for class '{class_name}'...")
    print(f"       Target: {target_samples} samples | Destination: '{data_path}'")
    if session_id:
        print(f"       Session ID: '{session_id}'")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam index {camera_index}.")
        return 0

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    collected_samples: List[List[float]] = []

    print("[INFO] Position your hand in front of the camera.")
    print("       Press [SPACE] to start recording samples.")
    print("       Press [Q] or [ESC] to cancel.")

    is_recording = False
    last_sample_time = 0.0
    sample_interval = 0.08  # 80ms between samples allows natural minor variance

    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened() and len(collected_samples) < target_samples:
            success, frame = cap.read()
            if not success:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            frame.flags.writeable = False
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            frame.flags.writeable = True

            hand_found = False
            current_normalized: Optional[List[float]] = None

            if results.multi_hand_landmarks:
                hand_found = True
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                landmark_list = LandmarkProcessor.calc_landmark_list(w, h, hand_landmarks)
                current_normalized = LandmarkProcessor.pre_process_landmark(landmark_list)

            # Header Banner
            cv2.rectangle(frame, (0, 0), (w, 75), (15, 23, 42), -1)

            status_text = f"CLASS: {class_name}  |  Collected: {len(collected_samples)} / {target_samples}"
            cv2.putText(frame, status_text, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.85, (56, 189, 248), 2)

            if not is_recording:
                action_prompt = "Press [SPACE] to Start Recording  |  [Q] Exit"
                cv2.putText(frame, action_prompt, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (203, 213, 225), 1)
            else:
                now = time.time()
                if hand_found and current_normalized:
                    if now - last_sample_time >= sample_interval:
                        # Validation: feature count == 42, values in [-1, 1]
                        if len(current_normalized) == 42 and all(-1.0 <= v <= 1.0 for v in current_normalized):
                            collected_samples.append(current_normalized)
                            last_sample_time = now

                    recording_prompt = f"RECORDING... ({len(collected_samples)}/{target_samples})  |  Hold pose steady!"
                    cv2.putText(frame, recording_prompt, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (34, 197, 94), 2)
                else:
                    cv2.putText(frame, "NO HAND DETECTED - Show hand in camera", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (239, 68, 68), 2)

            cv2.imshow("ISL Dataset Collection Utility", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q') or key == ord('Q'):
                print("[INFO] Collection canceled by user.")
                break
            elif key == 32:  # Space
                is_recording = not is_recording

    cap.release()
    cv2.destroyAllWindows()

    if not collected_samples:
        print("[INFO] No samples recorded.")
        return 0

    # Append to CSV
    print(f"[INFO] Writing {len(collected_samples)} verified samples to '{data_path}'...")
    with open(data_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for sample in collected_samples:
            writer.writerow([class_name, *sample])

    print(f"[SUCCESS] Successfully saved {len(collected_samples)} samples for class '{class_name}'.")
    inspect_dataset(data_path)
    return len(collected_samples)


def interactive_menu():
    print("\n========================================================")
    print("    INDIAN SIGN LANGUAGE DATASET COLLECTION UTILITY")
    print("========================================================")
    print("1. Capture new sign samples from webcam")
    print("2. Inspect current dataset distribution")
    print("3. Check for underrepresented classes (< 500 samples)")
    print("4. Launch classifier training (train_classifier.py)")
    print("5. Exit")

    choice = input("\nEnter choice [1-5]: ").strip()
    if choice == "1":
        cls_name = input("Enter class label to record (e.g., D, E, WATER): ").strip()
        count_str = input("Enter number of samples to record (default 50): ").strip()
        count = int(count_str) if count_str.isdigit() else 50
        capture_sign_samples(cls_name, target_samples=count)
    elif choice == "2":
        inspect_dataset()
    elif choice == "3":
        identify_underrepresented_classes()
    elif choice == "4":
        import subprocess
        subprocess.run([sys.executable, "train_classifier.py"])
    elif choice == "5":
        print("Goodbye.")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISL Dataset Collection Utility")
    parser.add_argument("--class-name", type=str, help="Sign class label to collect (e.g., 'D', 'WATER')")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to collect (default: 50)")
    parser.add_argument("--data-path", type=str, default="keypoint.csv", help="Path to CSV dataset")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--inspect", action="store_true", help="Inspect dataset without recording")

    args = parser.parse_args()

    if args.inspect:
        inspect_dataset(args.data_path)
    elif args.class_name:
        capture_sign_samples(
            class_name=args.class_name,
            target_samples=args.samples,
            data_path=args.data_path,
            camera_index=args.camera_index
        )
    else:
        interactive_menu()
