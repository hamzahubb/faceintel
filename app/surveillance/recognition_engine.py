"""
Recognition Engine — face detection (IMAGE mode) + hand detection + ArcFace recognition.

Creates its own MediaPipe FaceLandmarker and HandLandmarker instances in IMAGE
mode (no timestamp tracking needed). Uses a shared class-level lock to serialize
ONNX inference calls across multiple camera threads.

This module does NOT modify any existing files — it creates fresh MediaPipe
instances independent of the Flask pipeline's VIDEO-mode detectors.
"""

import os
import sys
import threading
import numpy as np

# Add project root to path so we can import existing modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from detector import download_model, download_hand_model, MODEL_PATH, HAND_MODEL_PATH
from recognizer import get_embedding, compare_with_employees

# Class-level lock shared by ALL RecognitionEngine instances.
# This serializes ONNX inference calls since the ONNX session
# (module-level singleton in recognizer.py) is not thread-safe.
_onnx_lock = threading.Lock()


class RecognitionEngine:
    """
    Wraps face detection, hand detection, and face recognition into a
    single reusable engine for the surveillance pipeline.

    Uses MediaPipe IMAGE mode — each frame is processed independently
    (no temporal tracking, no timestamp management needed).

    Thread safety:
        - MediaPipe instances are per-engine (not shared across threads).
        - ONNX embedding calls are serialized via a shared class-level lock.
        - compare_with_employees() is a pure numpy function (thread-safe).
    """

    def __init__(
        self,
        max_faces: int = 4,
        confidence_threshold: float = 0.62,
        wave_checkout_enabled: bool = True,
    ):
        """
        Initialize the recognition engine.

        Args:
            max_faces: Maximum faces to detect per frame.
            confidence_threshold: Minimum cosine similarity for a match.
            wave_checkout_enabled: Whether to also detect hands for wave checkout.
        """
        self.max_faces = max_faces
        self.confidence_threshold = confidence_threshold
        self.wave_checkout_enabled = wave_checkout_enabled

        # Ensure models are downloaded
        download_model(progress_callback=lambda msg: print(f"[RecognitionEngine] {msg}"))
        if wave_checkout_enabled:
            download_hand_model(progress_callback=lambda msg: print(f"[RecognitionEngine] {msg}"))

        # Create FaceLandmarker in IMAGE mode
        face_base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        face_options = vision.FaceLandmarkerOptions(
            base_options=face_base,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_faces,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

        # Create HandLandmarker in IMAGE mode (optional)
        self._hand_landmarker = None
        if wave_checkout_enabled:
            hand_base = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
            hand_options = vision.HandLandmarkerOptions(
                base_options=hand_base,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=max_faces * 2,  # Up to 2 hands per person
                min_hand_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)

        print(f"[RecognitionEngine] Initialized — max_faces={max_faces}, "
              f"threshold={confidence_threshold}, wave={wave_checkout_enabled}")

    def detect_faces(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Detect faces in a BGR frame using MediaPipe IMAGE mode.
        If no faces are detected at native resolution, retries with a 2x upscaled
        version to help detect small/distant faces in CCTV feeds.

        Returns a list of dicts, each with:
            - "landmarks": list of (x, y, z) normalized coords (468 points)
            - "blendshapes": dict of {name: score} (52 action units)
            - "bounding_box": {originX, originY, width, height} in pixels
        """
        import cv2

        h, w = frame_bgr.shape[:2]

        for scale in (1.0, 2.0):
            if scale == 1.0:
                work_bgr = frame_bgr
            else:
                work_bgr = cv2.resize(
                    frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )

            frame_rgb = work_bgr[:, :, ::-1]  # BGR -> RGB
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # IMAGE mode — no timestamp needed
            result = self._face_landmarker.detect(mp_image)

            if not result.face_landmarks:
                continue

            faces = []
            for i in range(len(result.face_landmarks)):
                # Extract landmarks
                landmarks = [(pt.x, pt.y, pt.z) for pt in result.face_landmarks[i]]

                # Extract blendshapes
                blendshapes = {}
                if result.face_blendshapes and len(result.face_blendshapes) > i:
                    for bs in result.face_blendshapes[i]:
                        blendshapes[bs.category_name] = bs.score

                # Compute bounding box from landmarks
                xs = [pt[0] for pt in landmarks]
                ys = [pt[1] for pt in landmarks]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                # Add padding
                pad_x = (max_x - min_x) * 0.15
                pad_y = (max_y - min_y) * 0.15
                min_x = max(0, min_x - pad_x)
                min_y = max(0, min_y - pad_y)
                max_x = min(1, max_x + pad_x)
                max_y = min(1, max_y + pad_y)

                # Bounding box in original frame pixel dimensions
                bounding_box = {
                    "originX": int(min_x * w),
                    "originY": int(min_y * h),
                    "width": int((max_x - min_x) * w),
                    "height": int((max_y - min_y) * h),
                }

                faces.append({
                    "landmarks": landmarks,
                    "blendshapes": blendshapes,
                    "bounding_box": bounding_box,
                })

            if faces:
                return faces

        return []

    def detect_hands(self, frame_bgr: np.ndarray) -> list[list]:
        """
        Detect hands in a BGR frame using MediaPipe IMAGE mode.

        Returns a list of hand landmark lists, each containing
        21 (x, y, z) normalized tuples.
        Returns empty list if hand detection is disabled.
        """
        if self._hand_landmarker is None:
            return []

        frame_rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self._hand_landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return []

        all_hands = []
        for hand_lms in result.hand_landmarks:
            landmarks = [(pt.x, pt.y, pt.z) for pt in hand_lms]
            all_hands.append(landmarks)

        return all_hands

    def crop_face(
        self, frame: np.ndarray, bounding_box: dict, padding: float = 0.2
    ) -> np.ndarray | None:
        """
        Crop a face region from the frame with extra padding.

        Args:
            frame: Full BGR frame.
            bounding_box: dict with originX, originY, width, height.
            padding: Fractional padding around the detected box.

        Returns:
            BGR face crop, or None if invalid.
        """
        h, w = frame.shape[:2]
        ox = bounding_box["originX"]
        oy = bounding_box["originY"]
        fw = bounding_box["width"]
        fh = bounding_box["height"]

        pad_x = int(fw * padding)
        pad_y = int(fh * padding)

        x1 = max(0, ox - pad_x)
        y1 = max(0, oy - pad_y)
        x2 = min(w, ox + fw + pad_x)
        y2 = min(h, oy + fh + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return crop

    def recognize(
        self, face_crop: np.ndarray, employees: list[dict]
    ) -> dict | None:
        """
        Get face embedding and compare against registered employees.

        Uses a shared lock to serialize ONNX inference calls.

        Args:
            face_crop: BGR face crop image.
            employees: List of employee dicts with 'embedding' bytes.

        Returns:
            dict with {employee_id, full_name, department, confidence}
            or None if no match above threshold.
        """
        if face_crop is None or face_crop.size == 0 or not employees:
            return None

        # Serialize ONNX calls across all threads
        with _onnx_lock:
            embedding = get_embedding(face_crop)

        if embedding is None:
            return None

        match = compare_with_employees(embedding, employees)

        if match and match.get("confidence", 0) >= self.confidence_threshold:
            return match

        return None

    def close(self):
        """Release MediaPipe resources."""
        if self._face_landmarker:
            self._face_landmarker.close()
            self._face_landmarker = None
        if self._hand_landmarker:
            self._hand_landmarker.close()
            self._hand_landmarker = None
        print("[RecognitionEngine] Closed.")
