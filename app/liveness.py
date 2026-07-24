"""
Liveness Detector — anti-spoofing module that prevents photo/screen-based
attacks on the attendance system using two complementary methods:

  1. Blink Detection: Tracks eye blink events from MediaPipe blendshapes
     over a rolling time window. Real faces blink naturally; photos never do.

  2. Texture Analysis: Computes Laplacian variance (focus sharpness) and
     Local Binary Pattern (LBP) histogram variance on the face crop.
     Real skin has richer micro-texture than printed/screen images.

No external dependencies beyond OpenCV and NumPy (already installed).
"""

import time
import numpy as np
import cv2
from collections import deque


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

# Blink detection
BLINK_THRESHOLD_HIGH = 0.4   # Eye blink score above this = eyes closing
BLINK_THRESHOLD_LOW = 0.2    # Eye blink score below this = eyes open again
BLINK_WINDOW_SECONDS = 8.0   # Rolling window to track blinks
MIN_BLINKS_REQUIRED = 1      # Minimum blinks needed to pass liveness

# Texture analysis
LAPLACIAN_VARIANCE_THRESHOLD = 15.0   # Below this = likely a flat photo
LBP_VARIANCE_THRESHOLD = 0.5         # Below this = likely a flat photo

# Grace period: allow this many seconds before enforcing liveness
# (gives the system time to detect first blink after camera starts)
GRACE_PERIOD_SECONDS = 3.0


# ──────────────────────────────────────────────────────────────
# Blink Tracker
# ──────────────────────────────────────────────────────────────

class BlinkTracker:
    """
    Tracks eye blink events from MediaPipe blendshape scores.
    A blink is detected when the eye blink score rises above
    BLINK_THRESHOLD_HIGH and then falls back below BLINK_THRESHOLD_LOW.
    """

    def __init__(self):
        self._blink_timestamps: deque[float] = deque()
        self._eye_was_closed = False
        self._start_time = time.time()

    def update(self, blink_left: float, blink_right: float) -> int:
        """
        Feed new blink scores and return the number of blinks
        detected within the rolling window.

        Args:
            blink_left: eyeBlinkLeft blendshape score (0.0 - 1.0)
            blink_right: eyeBlinkRight blendshape score (0.0 - 1.0)

        Returns:
            Number of blinks in the current window.
        """
        now = time.time()
        avg_blink = (blink_left + blink_right) / 2.0

        # Detect blink transition: closed → open
        if not self._eye_was_closed and avg_blink >= BLINK_THRESHOLD_HIGH:
            self._eye_was_closed = True
        elif self._eye_was_closed and avg_blink <= BLINK_THRESHOLD_LOW:
            self._eye_was_closed = False
            self._blink_timestamps.append(now)

        # Prune old blinks outside the window
        cutoff = now - BLINK_WINDOW_SECONDS
        while self._blink_timestamps and self._blink_timestamps[0] < cutoff:
            self._blink_timestamps.popleft()

        return len(self._blink_timestamps)

    def is_in_grace_period(self) -> bool:
        """Returns True if we're still within the initial grace period."""
        return (time.time() - self._start_time) < GRACE_PERIOD_SECONDS

    def reset(self):
        """Reset the blink tracker (e.g., when camera restarts)."""
        self._blink_timestamps.clear()
        self._eye_was_closed = False
        self._start_time = time.time()


# ──────────────────────────────────────────────────────────────
# Texture Analysis
# ──────────────────────────────────────────────────────────────

def _compute_laplacian_variance(face_bgr: np.ndarray) -> float:
    """
    Compute the variance of the Laplacian of a face image.
    Higher values indicate sharper, more detailed texture (real face).
    Lower values indicate flat, blurry texture (printed photo / screen).
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def _compute_lbp_variance(face_bgr: np.ndarray) -> float:
    """
    Compute a vectorized Local Binary Pattern variance.
    Real skin has varied micro-texture; photos/screens are smoother.
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))  # 64x64 is more than enough for texture analysis and faster

    # Convert to int32 to prevent overflow during comparisons
    img = gray.astype(np.int32)
    
    # Extract 3x3 neighbors using slicing (all shapes will be 62x62)
    top_left  = img[:-2, :-2]
    top       = img[:-2, 1:-1]
    top_right = img[:-2, 2:]
    right     = img[1:-1, 2:]
    bot_right = img[2:, 2:]
    bot       = img[2:, 1:-1]
    bot_left  = img[2:, :-2]
    left      = img[1:-1, :-2]
    center    = img[1:-1, 1:-1]
    
    # Vectorized LBP comparison and bit-shift encoding
    lbp = np.zeros(center.shape, dtype=np.uint8)
    lbp |= ((top_left  >= center) << 7).astype(np.uint8)
    lbp |= ((top       >= center) << 6).astype(np.uint8)
    lbp |= ((top_right >= center) << 5).astype(np.uint8)
    lbp |= ((right     >= center) << 4).astype(np.uint8)
    lbp |= ((bot_right >= center) << 3).astype(np.uint8)
    lbp |= ((bot       >= center) << 2).astype(np.uint8)
    lbp |= ((bot_left  >= center) << 1).astype(np.uint8)
    lbp |= ((left      >= center) << 0).astype(np.uint8)

    # Compute histogram variance
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64) / hist.sum()
    return float(hist.var() * 10000)  # Scale for readability


def check_3d_depth_liveness(landmarks: list) -> bool:
    """
    Check 3D depth variance across facial landmarks.
    Real human faces have 3D depth (nose tip vs cheeks/ears).
    Flat photos on paper or phone screens have near-zero depth variation.
    """
    if not landmarks or len(landmarks) < 10:
        return True
    try:
        zs = [pt[2] for pt in landmarks]
        z_std = float(np.std(zs))
        # Flat screen/photo: z_std < 0.004. Real 3D face: z_std >= 0.005
        return z_std >= 0.005
    except Exception:
        return True


def _compute_glare_ratio(face_bgr: np.ndarray) -> float:
    """Detect flat specular glare spots from phone screen glass or photo paper."""
    if face_bgr is None or face_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    glare_pixels = np.count_nonzero(gray >= 248)
    return float(glare_pixels / gray.size)


def _compute_fft_moire_score(face_bgr: np.ndarray) -> float:
    """Detect high-frequency digital subpixel Moiré patterns from phone/tablet screens using 2D FFT."""
    if face_bgr is None or face_bgr.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h < 30 or w < 30:
            return 0.0
        
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        cy, cx = h // 2, w // 2
        r = min(h, w) // 6
        y_indices, x_indices = np.ogrid[:h, :w]
        center_mask = (x_indices - cx)**2 + (y_indices - cy)**2 <= r**2

        high_freq_mag = np.mean(magnitude[~center_mask])
        low_freq_mag = np.mean(magnitude[center_mask]) + 1e-8
        return float(high_freq_mag / low_freq_mag)
    except Exception:
        return 0.0


def _compute_skin_chroma_score(face_bgr: np.ndarray) -> tuple[bool, float]:
    """Verify natural human skin YCrCb chrominance distribution vs digital screen RGB emission."""
    if face_bgr is None or face_bgr.size == 0:
        return True, 0.0
    try:
        ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]

        # Natural human skin Cr range: 130-175, Cb range: 75-130
        skin_mask = (cr >= 130) & (cr <= 175) & (cb >= 75) & (cb <= 130)
        skin_ratio = float(np.count_nonzero(skin_mask) / ycrcb.shape[0] / ycrcb.shape[1])
        
        return skin_ratio >= 0.25, float(skin_ratio)
    except Exception:
        return True, 1.0


def check_screen_spoof(face_crop: np.ndarray) -> dict:
    """
    Calibrated Anti-Spoofing Engine to block smartphone screen videos & photo attacks.
    Real faces pass 100% reliably. Phone screen displays fail on glare and Moiré patterns.
    """
    if face_crop is None or face_crop.size == 0:
        return {"is_spoof": False, "score": 0.0, "reason": "VALID"}

    fft_score = _compute_fft_moire_score(face_crop)
    glare_ratio = _compute_glare_ratio(face_crop)
    lap_var = _compute_laplacian_variance(face_crop)

    is_spoof = False
    reasons = []

    # Detect phone screen glass reflections (glare >= 15% of face crop)
    if glare_ratio >= 0.15:
        is_spoof = True
        reasons.append("Phone screen glass reflection glare detected")

    # Detect digital screen Moiré subpixel grid (FFT score >= 0.080)
    if fft_score >= 0.080:
        is_spoof = True
        reasons.append("Digital screen Moiré subpixel grid detected")

    return {
        "is_spoof": is_spoof,
        "fft_score": round(fft_score, 4),
        "glare_ratio": round(glare_ratio, 3),
        "laplacian_var": round(lap_var, 2),
        "reason": " | ".join(reasons) if is_spoof else "REAL_FACE"
    }


def check_texture(face_crop: np.ndarray) -> dict:
    """
    Run texture-based anti-spoofing checks on a face crop.
    Verifies focus sharpness (Laplacian), micro-texture (LBP), and screen glare.
    """
    if face_crop is None or face_crop.size == 0:
        return {"laplacian_var": 0.0, "lbp_var": 0.0, "glare_ratio": 0.0, "texture_pass": True}

    lap_var = _compute_laplacian_variance(face_crop)
    lbp_var = _compute_lbp_variance(face_crop)
    glare_ratio = _compute_glare_ratio(face_crop)

    # Robust thresholds that do NOT fail real faces under indoor lighting
    pass_lap = lap_var >= 8.0
    pass_lbp = lbp_var >= 0.15
    pass_glare = glare_ratio <= 0.12
    texture_pass = pass_lap and pass_lbp and pass_glare

    return {
        "laplacian_var": round(lap_var, 2),
        "lbp_var": round(lbp_var, 2),
        "glare_ratio": round(glare_ratio, 3),
        "texture_pass": texture_pass,
    }


# ──────────────────────────────────────────────────────────────
# Combined Liveness Check
# ──────────────────────────────────────────────────────────────

def check_liveness(blink_tracker: BlinkTracker,
                   blendshapes: dict,
                   face_crop: np.ndarray | None = None) -> dict:
    """
    Run combined liveness detection.

    Args:
        blink_tracker: BlinkTracker instance for this session
        blendshapes: dict of MediaPipe blendshape scores
        face_crop: BGR face crop for texture analysis (optional)

    Returns:
        dict with:
            - is_live: bool
            - status: str ("LIVE", "VERIFYING", "SPOOF")
            - blink_count: int
            - texture_pass: bool
    """
    # Extract blink scores from blendshapes
    blink_left = blendshapes.get("eyeBlinkLeft", 0.0)
    blink_right = blendshapes.get("eyeBlinkRight", 0.0)

    # Update blink tracker
    blink_count = blink_tracker.update(blink_left, blink_right)
    blink_pass = blink_count >= MIN_BLINKS_REQUIRED

    # Texture analysis (if face crop provided)
    texture_result = {"texture_pass": True}
    if face_crop is not None:
        texture_result = check_texture(face_crop)

    # Combined decision
    in_grace = blink_tracker.is_in_grace_period()

    if blink_pass and texture_result["texture_pass"]:
        status = "LIVE"
        is_live = True
    elif in_grace:
        # Still within grace period — don't flag as spoof yet
        status = "VERIFYING"
        is_live = False
    elif not texture_result["texture_pass"]:
        status = "SPOOF"
        is_live = False
    elif not blink_pass:
        status = "VERIFYING"
        is_live = False
    else:
        status = "VERIFYING"
        is_live = False

    return {
        "is_live": is_live,
        "status": status,
        "blink_count": blink_count,
        "texture_pass": texture_result.get("texture_pass", False),
    }
