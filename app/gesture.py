"""
Gesture Detector — detects hand-wave gestures for checkout validation.

Tracks the wrist landmark's horizontal position over a rolling window
and detects side-to-side oscillation (wave pattern) by counting
significant direction changes.

No external dependencies beyond NumPy (already installed).
"""

from collections import deque
import time


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────

# Rolling window size (number of frames to track)
WAVE_WINDOW_FRAMES = 25  # ~0.8 seconds at 30 FPS

# Minimum horizontal displacement to count as a direction change
# (in normalized coordinates 0.0 - 1.0, so 0.03 = ~3% of frame width)
MIN_DIRECTION_CHANGE = 0.03

# Minimum number of direction reversals to qualify as a wave
MIN_REVERSALS = 3

# Minimum total horizontal amplitude (peak-to-peak) for the wave
MIN_AMPLITUDE = 0.08

# Cooldown: once a wave is detected, ignore subsequent waves for this duration
WAVE_COOLDOWN_SECONDS = 5.0


# ──────────────────────────────────────────────────────────────
# Wave Tracker
# ──────────────────────────────────────────────────────────────

class WaveTracker:
    """
    Tracks hand position over time and detects wave gestures.

    A wave is defined as rapid horizontal back-and-forth movement
    of the hand with at least MIN_REVERSALS direction changes
    and MIN_AMPLITUDE total swing within the tracking window.
    """

    def __init__(self):
        self._x_history: deque[float] = deque(maxlen=WAVE_WINDOW_FRAMES)
        self._last_wave_time: float = 0.0
        self._no_hand_count: int = 0

    def update(self, hand_landmarks: list | None) -> bool:
        """
        Feed new hand landmark data and check for wave gesture.

        Args:
            hand_landmarks: List of (x, y, z) tuples for 21 hand landmarks,
                           or None if no hand is detected in this frame.

        Returns:
            True if a wave gesture was just completed, False otherwise.
        """
        if hand_landmarks is None:
            self._no_hand_count += 1
            # If hand disappears for too long, reset tracking
            if self._no_hand_count > 10:
                self._x_history.clear()
            return False

        self._no_hand_count = 0

        # Use wrist landmark (index 0) x-coordinate for tracking
        wrist_x = hand_landmarks[0][0]
        self._x_history.append(wrist_x)

        # Need enough data points to analyze
        if len(self._x_history) < 8:
            return False

        # Check cooldown
        now = time.time()
        if now - self._last_wave_time < WAVE_COOLDOWN_SECONDS:
            return False

        # Analyze the x-position history for wave pattern
        is_wave = self._detect_wave()

        if is_wave:
            self._last_wave_time = now
            self._x_history.clear()
            print("[Gesture] 👋 Wave detected!")
            return True

        return False

    def _detect_wave(self) -> bool:
        """Analyze x-position history for oscillation pattern."""
        positions = list(self._x_history)

        # Count significant direction reversals
        reversals = 0
        last_direction = 0  # -1 = left, +1 = right, 0 = undecided

        for i in range(1, len(positions)):
            delta = positions[i] - positions[i - 1]

            # Only count if the movement is significant enough
            if abs(delta) < MIN_DIRECTION_CHANGE * 0.3:
                continue

            current_direction = 1 if delta > 0 else -1

            if last_direction != 0 and current_direction != last_direction:
                reversals += 1

            last_direction = current_direction

        # Check amplitude (peak-to-peak swing)
        amplitude = max(positions) - min(positions)

        return reversals >= MIN_REVERSALS and amplitude >= MIN_AMPLITUDE

    def reset(self):
        """Reset the wave tracker state."""
        self._x_history.clear()
        self._last_wave_time = 0.0
        self._no_hand_count = 0
