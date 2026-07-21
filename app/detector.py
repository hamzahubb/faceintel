"""
Face Detector — downloads and initializes the MediaPipe Face Landmarker model,
runs face detection on OpenCV frames, and returns landmarks + blendshapes.
"""

import os
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Model configuration
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")
HAND_MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
FALLBACK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "python_app", "models", "face_landmarker.task")


def download_model(progress_callback=None):
    """Download the face landmarker model if it doesn't exist locally."""
    if os.path.exists(MODEL_PATH):
        if progress_callback:
            progress_callback("Model already cached locally.")
        return MODEL_PATH

    # Check fallback path (if already downloaded in the old folder structure)
    if os.path.exists(FALLBACK_PATH):
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            os.rename(FALLBACK_PATH, MODEL_PATH)
            if progress_callback:
                progress_callback("Found existing model. Migrated to local directory.")
            return MODEL_PATH
        except Exception:
            pass

    os.makedirs(MODEL_DIR, exist_ok=True)

    if progress_callback:
        progress_callback("Downloading face landmarker model (~5 MB)...")

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        if progress_callback:
            progress_callback("Model download complete!")
    except Exception as e:
        if progress_callback:
            progress_callback(f"Download failed: {e}")
        raise

    return MODEL_PATH


def download_hand_model(progress_callback=None):
    """Download the hand landmarker model if it doesn't exist locally."""
    if os.path.exists(HAND_MODEL_PATH):
        if progress_callback:
            progress_callback("Hand model already cached locally.")
        return HAND_MODEL_PATH

    os.makedirs(MODEL_DIR, exist_ok=True)

    if progress_callback:
        progress_callback("Downloading hand landmarker model (~5.6 MB)...")

    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        if progress_callback:
            progress_callback("Hand model download complete!")
    except Exception as e:
        if progress_callback:
            progress_callback(f"Hand model download failed: {e}")
        raise

    return HAND_MODEL_PATH


class FaceDetector:
    """Wraps the MediaPipe FaceLandmarker for synchronous inference."""

    def __init__(self, model_path: str = MODEL_PATH, running_mode=vision.RunningMode.VIDEO):
        self.running_mode = running_mode
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_faces=4,  # Detect up to 4 faces simultaneously
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._frame_count = 0

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """
        Run face detection on a BGR OpenCV frame.

        Returns a list of dicts, each with:
          - "landmarks": list of (x, y, z) normalized coords (468 points)
          - "blendshapes": dict of {name: score} (52 action units)
          - "bounding_box": dict with originX, originY, width, height in pixels
        """
        frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        if self.running_mode == vision.RunningMode.VIDEO:
            self._frame_count += 1
            # MediaPipe VIDEO mode requires monotonically increasing timestamps
            timestamp_ms = int(self._frame_count * 33)  # ~30 FPS
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        else:
            result = self.landmarker.detect(mp_image)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return []

        h, w = frame_bgr.shape[:2]
        faces = []

        for i in range(len(result.face_landmarks)):
            # Extract landmarks
            landmarks = []
            for pt in result.face_landmarks[i]:
                landmarks.append((pt.x, pt.y, pt.z))

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

        return faces

    def close(self):
        """Release MediaPipe resources."""
        if self.landmarker:
            self.landmarker.close()


class HandDetector:
    """Wraps the MediaPipe HandLandmarker for synchronous VIDEO-mode inference."""

    def __init__(self, model_path: str = HAND_MODEL_PATH):
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=4,  # Detect up to 4 hands simultaneously
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self._frame_count = 0

    def detect(self, frame_bgr: np.ndarray) -> list[list]:
        """
        Run hand detection on a BGR OpenCV frame.

        Returns a list of lists, where each sublist contains 21 (x, y, z) 
        normalized landmark tuples for a detected hand.
        """
        frame_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        self._frame_count += 1
        timestamp_ms = int(self._frame_count * 33)

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return []

        all_hands = []
        for i in range(len(result.hand_landmarks)):
            landmarks = []
            for pt in result.hand_landmarks[i]:
                landmarks.append((pt.x, pt.y, pt.z))
            all_hands.append(landmarks)

        return all_hands

    def close(self):
        """Release MediaPipe resources."""
        if self.landmarker:
            self.landmarker.close()


