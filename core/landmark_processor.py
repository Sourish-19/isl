"""
Landmark Processing and Spatial Normalization Module for Indian Sign Language Recognition.
Preserves the exact geometric transformation from the original project while adding
type safety, zero-division guards, and multi-format landmark support.
"""

from typing import List, Tuple, Union, Optional
import copy
import itertools
import numpy as np


class LandmarkProcessor:
    """
    Handles keypoint extraction, origin centering (wrist translation),
    and max-absolute scale normalization for MediaPipe hand landmarks.
    """

    NUM_LANDMARKS = 21
    FEATURE_DIM = 42

    @staticmethod
    def calc_landmark_list(image_width: int, image_height: int, landmarks) -> List[List[int]]:
        """
        Converts MediaPipe normalized landmark coordinates [0, 1] into pixel coordinates.
        """
        landmark_point: List[List[int]] = []

        # Compatible with MediaPipe landmark objects, dicts, and lists
        points = getattr(landmarks, 'landmark', landmarks)
        for landmark in points:
            if hasattr(landmark, 'x'):
                x_val = landmark.x
                y_val = landmark.y
            elif isinstance(landmark, (list, tuple)):
                x_val = landmark[0]
                y_val = landmark[1]
            elif isinstance(landmark, dict):
                x_val = landmark['x']
                y_val = landmark['y']
            else:
                raise TypeError(f"Unsupported landmark type: {type(landmark)}")

            landmark_x = min(int(x_val * image_width), image_width - 1)
            landmark_y = min(int(y_val * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    @classmethod
    def pre_process_landmark(cls, landmark_list: List[List[Union[int, float]]]) -> List[float]:
        """
        Translates landmark coordinates relative to the wrist origin (Landmark 0)
        and scales by max-absolute magnitude into [-1.0, 1.0].
        """
        if not landmark_list or len(landmark_list) != cls.NUM_LANDMARKS:
            raise ValueError(f"Expected {cls.NUM_LANDMARKS} landmarks, got {len(landmark_list) if landmark_list else 0}")

        temp_landmark_list = copy.deepcopy(landmark_list)

        # 1. Translate coordinates relative to wrist joint (Index 0)
        base_x = temp_landmark_list[0][0]
        base_y = temp_landmark_list[0][1]

        for index in range(len(temp_landmark_list)):
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

        # 2. Flatten 2D coordinate pairs into 1D vector (42 features)
        flattened: List[float] = list(itertools.chain.from_iterable(temp_landmark_list))

        # 3. Max-absolute scale normalization
        max_abs = max([abs(val) for val in flattened]) if flattened else 0.0
        if max_abs == 0.0:
            max_abs = 1.0  # Guard against division by zero

        normalized = [val / max_abs for val in flattened]
        return normalized

    @staticmethod
    def get_bounding_box(image_width: int, image_height: int, landmarks) -> Tuple[int, int, int, int]:
        """
        Computes bounding box (x_min, y_min, x_max, y_max) around detected hand.
        """
        points = getattr(landmarks, 'landmark', landmarks)
        x_coords = []
        y_coords = []
        for p in points:
            x_val = getattr(p, 'x', p[0] if isinstance(p, (list, tuple)) else p['x'])
            y_val = getattr(p, 'y', p[1] if isinstance(p, (list, tuple)) else p['y'])
            x_coords.append(x_val * image_width)
            y_coords.append(y_val * image_height)

        x_min = max(0, int(min(x_coords)) - 15)
        y_min = max(0, int(min(y_coords)) - 15)
        x_max = min(image_width, int(max(x_coords)) + 15)
        y_max = min(image_height, int(max(y_coords)) + 15)

        return (x_min, y_min, x_max, y_max)
