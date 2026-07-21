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


def check_texture(face_crop: np.ndarray) -> dict:
    """
    Run texture-based anti-spoofing checks on a face crop.

    Returns:
        dict with keys:
            - laplacian_var: float
            - lbp_var: float
            - texture_pass: bool (True if both checks pass)
    """
    if face_crop is None or face_crop.size == 0:
        return {"laplacian_var": 0.0, "lbp_var": 0.0, "texture_pass": False}

    lap_var = _compute_laplacian_variance(face_crop)
    lbp_var = _compute_lbp_variance(face_crop)

    texture_pass = (lap_var >= LAPLACIAN_VARIANCE_THRESHOLD and
                    lbp_var >= LBP_VARIANCE_THRESHOLD)

    return {
        "laplacian_var": round(lap_var, 2),
        "lbp_var": round(lbp_var, 2),
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
