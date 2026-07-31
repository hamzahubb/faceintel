"""
Hybrid Anti-Spoofing Engine — Dual-Model Pipeline

Model A: MiniFASNetV2 (ONNX) — Passive texture/frequency analysis for print & screen attacks.
Model B: MediaPipe 3D Landmark Liveness — Active depth, EAR blink, and motion delta checks.

A frame must pass BOTH models to be considered live.
Designed for CPU inference at 30+ FPS via onnxruntime.
"""

import os
import time
import urllib.request
import numpy as np
import cv2
from collections import deque

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
ANTI_SPOOF_MODEL_PATH = os.path.join(MODEL_DIR, "anti_spoof_minifasnetv2.onnx")
ANTI_SPOOF_MODEL_URL = (
    "https://github.com/facenox/face-antispoof-onnx/releases/download/v1.0.0/best_model.onnx"
)

# MiniFASNetV2 thresholds
MINIFAS_INPUT_SIZE = (128, 128)
MINIFAS_CROP_SCALE = 2.7  # Scale factor for face crop margin
MINIFAS_REAL_THRESHOLD = 0.50

# Class index of "Real Face" in the model's 2-logit output.
# Calibrated against the 182 enrollment images in dataset/ plus simulated
# screen-replay and print attacks (see scratch/calibrate_minifas.py):
#   idx1-as-real -> real mean p=0.975, screen mean p=0.088  (separation +0.89)
#   idx0-as-real -> real mean p=0.440, i.e. 60% of real faces wrongly blocked.
MINIFAS_REAL_INDEX = 1

# This model expects RGB input with ImageNet normalization. Feeding it BGR/255
# (as an earlier revision did) collapses the separation to near-random.
MINIFAS_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
MINIFAS_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Number of frames that must accumulate before Model A's verdict is enforced.
# A single frame during camera warmup/auto-exposure is not enough evidence to
# reject a legitimate user.
MINIFAS_MIN_FRAMES = 3

# MediaPipe 3D Landmark Liveness thresholds
EAR_BLINK_THRESHOLD = 0.22  # Below this = eye closed
EAR_OPEN_THRESHOLD = 0.28  # Above this = eye open
Z_DEPTH_RATIO_MIN = 1.05  # nose_z / ear_z ratio for real 3D face
MOTION_DELTA_MIN = 0.0008  # Minimum landmark motion between frames (real faces move)
MOTION_DELTA_MAX = 0.08  # Maximum motion (too much = shaking phone/video)

# Temporal tracking
MOTION_HISTORY_SIZE = 5  # Number of frames to track for motion analysis


# ──────────────────────────────────────────────────────────────
# Model A: MiniFASNetV2 ONNX
# ──────────────────────────────────────────────────────────────

_fas_session = None
_fas_input_name = None


def _download_anti_spoof_model():
    """Download MiniFASNetV2 ONNX model if not present."""
    if os.path.exists(ANTI_SPOOF_MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("[AntiSpoof] Downloading MiniFASNetV2 ONNX model (~1.5 MB)...")
    try:
        urllib.request.urlretrieve(ANTI_SPOOF_MODEL_URL, ANTI_SPOOF_MODEL_PATH)
        print("[AntiSpoof] MiniFASNetV2 model download complete!")
    except Exception as e:
        print(f"[AntiSpoof] Model download failed: {e}")
        raise


def _ensure_fas_loaded():
    """Lazily load MiniFASNetV2 ONNX session."""
    global _fas_session, _fas_input_name
    if _fas_session is not None:
        return

    import onnxruntime as ort

    _download_anti_spoof_model()
    print("[AntiSpoof] Loading MiniFASNetV2 ONNX model...")
    _fas_session = ort.InferenceSession(
        ANTI_SPOOF_MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    _fas_input_name = _fas_session.get_inputs()[0].name

    # Determine input shape from model
    input_shape = _fas_session.get_inputs()[0].shape
    print(f"[AntiSpoof] MiniFASNetV2 loaded. Input shape: {input_shape}")

    # Warm up
    h = input_shape[2] if len(input_shape) == 4 else 80
    w = input_shape[3] if len(input_shape) == 4 else 80
    dummy = np.zeros((1, 3, h, w), dtype=np.float32)
    _fas_session.run(None, {_fas_input_name: dummy})
    print("[AntiSpoof] MiniFASNetV2 model ready!")


def _preprocess_minifas(face_bgr: np.ndarray, bbox: dict = None, full_frame: np.ndarray = None) -> np.ndarray:
    """
    Preprocess face for MiniFASNetV2.
    Uses 2.7x crop scale around bounding box for context, then resizes to 80x80 (or model input size).
    """
    if bbox is not None and full_frame is not None:
        # Apply 2.7x scale crop from full frame
        h_frame, w_frame = full_frame.shape[:2]
        ox = bbox.get("originX", 0)
        oy = bbox.get("originY", 0)
        bw = bbox.get("width", 0)
        bh = bbox.get("height", 0)

        # Calculate center and scaled dimensions
        cx = ox + bw / 2
        cy = oy + bh / 2
        max_dim = max(bw, bh)
        scaled_half = int(max_dim * MINIFAS_CROP_SCALE / 2)

        x1 = max(0, int(cx - scaled_half))
        y1 = max(0, int(cy - scaled_half))
        x2 = min(w_frame, int(cx + scaled_half))
        y2 = min(h_frame, int(cy + scaled_half))

        crop = full_frame[y1:y2, x1:x2]
    elif face_bgr is not None:
        # Pad face crop to provide background context for MiniFASNet
        h, w = face_bgr.shape[:2]
        pad_h, pad_w = int(h * 0.35), int(w * 0.35)
        crop = cv2.copyMakeBorder(face_bgr, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REFLECT)
    else:
        crop = None

    if crop is None or crop.size == 0:
        return None

    # Get the model's expected input size
    _ensure_fas_loaded()
    input_shape = _fas_session.get_inputs()[0].shape
    h_target = input_shape[2] if len(input_shape) == 4 else 80
    w_target = input_shape[3] if len(input_shape) == 4 else 80

    # Resize to model input size
    resized = cv2.resize(crop, (w_target, h_target))

    # BGR -> RGB, scale to [0, 1], then ImageNet-normalize (what the net was trained on)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = (rgb - MINIFAS_MEAN) / MINIFAS_STD

    # HWC -> CHW -> NCHW
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0).astype(np.float32)

    return blob


def run_minifas(face_bgr: np.ndarray, bbox: dict = None, full_frame: np.ndarray = None) -> dict:
    """
    Run MiniFASNetV2 inference on a face crop.

    Returns:
        dict with:
            - is_real: bool
            - p_real: float (0-1)
            - p_spoof: float (0-1)
    """
    _ensure_fas_loaded()

    blob = _preprocess_minifas(face_bgr, bbox, full_frame)
    if blob is None:
        return {"is_real": True, "p_real": 1.0, "p_spoof": 0.0}  # Fail open if no input

    try:
        outputs = _fas_session.run(None, {_fas_input_name: blob})
        # Output format varies by model export:
        # Some models output [p_real, p_spoof], others output logits
        raw = outputs[0].flatten()

        if len(raw) >= 2:
            # MiniFASNet class mapping: Index 1 = Real Face (Live), Index 0 = Spoof (Fake)
            exp_vals = np.exp(raw - np.max(raw))  # Numerically stable softmax
            probs = exp_vals / exp_vals.sum()
            p_real = float(probs[MINIFAS_REAL_INDEX])
            p_spoof = float(1.0 - p_real)
        else:
            # Single output — apply sigmoid
            p_real = float(1.0 / (1.0 + np.exp(-raw[0])))
            p_spoof = 1.0 - p_real

        is_real = p_real >= MINIFAS_REAL_THRESHOLD

        return {
            "is_real": is_real,
            "p_real": round(p_real, 4),
            "p_spoof": round(p_spoof, 4),
        }
    except Exception as e:
        print(f"[AntiSpoof] MiniFASNetV2 inference error: {e}")
        return {"is_real": True, "p_real": 1.0, "p_spoof": 0.0}  # Fail open


# ──────────────────────────────────────────────────────────────
# Model B: MediaPipe 3D Landmark Liveness
# ──────────────────────────────────────────────────────────────

def compute_ear(landmarks: list, side: str = "left") -> float:
    """
    Compute Eye Aspect Ratio (EAR) from MediaPipe face landmarks.
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    Low EAR = eyes closed (blink), High EAR = eyes open.
    Photos have constant EAR; real faces vary due to blinks.

    MediaPipe landmark indices:
      Left eye:  33 (outer), 133 (inner), 159 (top), 145 (bottom), 160 (top2), 144 (bottom2)
      Right eye: 362 (outer), 263 (inner), 386 (top), 374 (bottom), 385 (top2), 380 (bottom2)
    """
    if not landmarks or len(landmarks) < 400:
        return 0.3  # Default open

    try:
        if side == "left":
            p1 = landmarks[33]   # outer corner
            p4 = landmarks[133]  # inner corner
            p2 = landmarks[160]  # top-left
            p6 = landmarks[144]  # bottom-left
            p3 = landmarks[159]  # top-right
            p5 = landmarks[145]  # bottom-right
        else:
            p1 = landmarks[362]  # outer corner
            p4 = landmarks[263]  # inner corner
            p2 = landmarks[385]  # top-left
            p6 = landmarks[380]  # bottom-left
            p3 = landmarks[386]  # top-right
            p5 = landmarks[374]  # bottom-right

        def dist(a, b):
            return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

        vertical1 = dist(p2, p6)
        vertical2 = dist(p3, p5)
        horizontal = dist(p1, p4)

        if horizontal < 1e-6:
            return 0.3

        ear = (vertical1 + vertical2) / (2.0 * horizontal)
        return float(ear)
    except (IndexError, TypeError):
        return 0.3


def compute_z_depth_ratio(landmarks: list) -> float:
    """
    Compute 3D depth ratio: nose_tip_z / avg_ear_z.
    Real faces have a prominent nose (ratio > 1.05).
    Flat screens have ratio ~1.0.

    MediaPipe landmark indices:
      Nose tip: 1
      Left ear tragion: 234
      Right ear tragion: 454
    """
    if not landmarks or len(landmarks) < 455:
        return 1.2  # Default real

    try:
        nose_z = abs(landmarks[1][2])  # Nose tip z-coordinate
        left_ear_z = abs(landmarks[234][2])  # Left ear
        right_ear_z = abs(landmarks[454][2])  # Right ear
        avg_ear_z = (left_ear_z + right_ear_z) / 2.0

        if avg_ear_z < 1e-8:
            return 1.2  # Default real if no depth data

        ratio = nose_z / avg_ear_z
        return float(ratio)
    except (IndexError, TypeError):
        return 1.2


def compute_landmark_motion(current_landmarks: list, previous_landmarks: list) -> float:
    """
    Compute average motion delta between two frames' landmarks.
    Real faces have subtle micro-saccades (0.001-0.02 normalized).
    Static photos have ~0 motion. Phone videos may have different patterns.

    Uses a subset of stable landmarks (nose, eyes, mouth corners) for efficiency.
    """
    MOTION_INDICES = [1, 4, 5, 6, 33, 133, 159, 145, 362, 263, 386, 374, 61, 291]

    if (not current_landmarks or not previous_landmarks or
            len(current_landmarks) < 400 or len(previous_landmarks) < 400):
        return 0.005  # Default mid-range

    try:
        deltas = []
        for idx in MOTION_INDICES:
            if idx < len(current_landmarks) and idx < len(previous_landmarks):
                curr = current_landmarks[idx]
                prev = previous_landmarks[idx]
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                deltas.append((dx**2 + dy**2) ** 0.5)

        if not deltas:
            return 0.005

        return float(np.mean(deltas))
    except (IndexError, TypeError):
        return 0.005


# ──────────────────────────────────────────────────────────────
# Hybrid Anti-Spoof Engine (Singleton)
# ──────────────────────────────────────────────────────────────

class HybridAntiSpoof:
    """
    Dual-model anti-spoofing engine combining:
    - Model A: MiniFASNetV2 (passive texture analysis)
    - Model B: MediaPipe 3D Landmark Liveness (active depth/motion)

    Both models must pass for a frame to be considered live.
    """

    def __init__(self):
        self._prev_landmarks = None
        self._motion_history = deque(maxlen=MOTION_HISTORY_SIZE)
        self._ear_history = deque(maxlen=10)
        self._fas_history = deque(maxlen=5)
        self._last_check_time = 0
        self._model_loaded = False

    def preload(self):
        """Pre-load the MiniFASNetV2 model (call during app startup)."""
        try:
            _ensure_fas_loaded()
            self._model_loaded = True
            print("[AntiSpoof] Hybrid engine ready.")
        except Exception as e:
            print(f"[AntiSpoof] WARNING: MiniFASNetV2 failed to load: {e}")
            print("[AntiSpoof] Falling back to Model B only (3D landmark liveness).")
            self._model_loaded = False

    def check(self, face_crop: np.ndarray, landmarks: list,
              blendshapes: dict = None, bbox: dict = None,
              full_frame: np.ndarray = None) -> dict:
        result = {
            "is_spoof": False,
            "is_live": True,
            "reason": "REAL_FACE",
            "model_a_score": 1.0,
            "model_b_depth": 1.2,
            "model_b_ear_left": 0.3,
            "model_b_ear_right": 0.3,
            "model_b_motion": 0.005,
            "method": "none",
        }

        if face_crop is None or face_crop.size == 0:
            return result

        reasons = []
        methods = []

        # ── Model A: MiniFASNetV2 with 5-Frame Rolling Window ──
        if self._model_loaded:
            fas_result = run_minifas(face_crop, bbox, full_frame)
            self._fas_history.append(fas_result["p_real"])

            # Compute smoothed real face probability across recent frames
            smoothed_p_real = float(np.mean(self._fas_history))
            result["model_a_score"] = round(smoothed_p_real, 4)
            n_frames = len(self._fas_history)
            print(f"[AntiSpoof ModelA] MiniFAS: frame_p_real={fas_result['p_real']:.4f} "
                  f"smoothed_p_real={smoothed_p_real:.4f} frames={n_frames}")

            # Enforce only once enough frames have accumulated. A lone warmup frame
            # (auto-exposure still settling) must not reject a legitimate user.
            if n_frames < MINIFAS_MIN_FRAMES:
                result["warming_up"] = True
            elif smoothed_p_real < MINIFAS_REAL_THRESHOLD:
                reasons.append(f"MiniFASNetV2: spoof probability {1.0 - smoothed_p_real:.1%}")
                methods.append("MiniFASNetV2")

        # ── Screen Glass Glare & FFT Moiré Subpixel Grid ──
        if face_crop is not None and face_crop.size > 0:
            try:
                from liveness import check_screen_spoof
                screen_spoof_res = check_screen_spoof(face_crop)
                if screen_spoof_res["is_spoof"]:
                    reasons.append(screen_spoof_res["reason"])
                    methods.append("ScreenMoiréGlare")
            except Exception as e:
                print(f"[AntiSpoof] Screen check error: {e}")

        # ── Model B: MediaPipe 3D Landmark Liveness ──
        model_b_fail = False

        # B.1: 3D Z-Depth Ratio
        if landmarks and len(landmarks) >= 455:
            z_ratio = compute_z_depth_ratio(landmarks)
            result["model_b_depth"] = round(z_ratio, 4)

            # Compute z-std across facial landmarks, normalized by face width so the
            # verdict does not change with how far the user sits from the camera.
            # A raw z_std threshold rejects anyone standing slightly further back.
            zs = [pt[2] for pt in landmarks]
            z_std = float(np.std(zs))
            xs = [pt[0] for pt in landmarks]
            face_width = float(max(xs) - min(xs))
            z_ratio_norm = z_std / face_width if face_width > 1e-6 else 1.0

            print(f"[AntiSpoof ModelB] 3D Depth: z_ratio={z_ratio:.4f} z_std={z_std:.6f} "
                  f"face_w={face_width:.4f} z_norm={z_ratio_norm:.4f}")

            # MediaPipe fits a canonical 3D mesh, so even a flat photo yields some z.
            # Keep this gate conservative — it catches only degenerate/planar fits,
            # while Model A carries the real screen/print discrimination.
            if z_ratio_norm < 0.02:
                reasons.append(f"Flat screen/photo detected (z_norm={z_ratio_norm:.4f})")
                methods.append("3D_Depth")
                model_b_fail = True

        # B.2: EAR (Eye Aspect Ratio) — track variance over time
        if landmarks and len(landmarks) >= 400:
            ear_left = compute_ear(landmarks, "left")
            ear_right = compute_ear(landmarks, "right")
            result["model_b_ear_left"] = round(ear_left, 4)
            result["model_b_ear_right"] = round(ear_right, 4)

            avg_ear = (ear_left + ear_right) / 2.0
            self._ear_history.append(avg_ear)

            print(f"[AntiSpoof ModelB] EAR: left={ear_left:.4f} right={ear_right:.4f} avg={avg_ear:.4f}")

        # B.3: Motion Delta
        if landmarks and len(landmarks) >= 400:
            if self._prev_landmarks is not None:
                motion = compute_landmark_motion(landmarks, self._prev_landmarks)
                self._motion_history.append(motion)
                result["model_b_motion"] = round(motion, 6)
                print(f"[AntiSpoof ModelB] Motion: delta={motion:.6f}")
            self._prev_landmarks = landmarks[:]

        # ── Combined Anti-Spoof Decision ──
        if reasons:
            result["is_spoof"] = True
            result["is_live"] = False
            result["reason"] = " | ".join(reasons)
            result["method"] = "+".join(methods)
        else:
            result["is_spoof"] = False
            result["is_live"] = True
            result["reason"] = "REAL_FACE"
            result["method"] = "passed"

        return result

    def reset(self):
        """Reset temporal state (call when camera restarts or user changes)."""
        self._prev_landmarks = None
        self._motion_history.clear()
        self._ear_history.clear()
        self._fas_history.clear()  # else stale scores bleed into the next login attempt
        self._last_check_time = 0


# ──────────────────────────────────────────────────────────────
# Singleton Access
# ──────────────────────────────────────────────────────────────

_engine = None


def get_anti_spoof_engine() -> HybridAntiSpoof:
    """Get or create the singleton HybridAntiSpoof engine."""
    global _engine
    if _engine is None:
        _engine = HybridAntiSpoof()
        _engine.preload()
    return _engine
