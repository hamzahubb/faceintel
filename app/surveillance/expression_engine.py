"""
Expression Engine — thin wrapper around classifier.py for the surveillance module.

Keeps imports clean and allows future extension (e.g., expression history
per camera, aggregation, or smoothing across frames).
"""

import sys
import os

# Add project root to path so we can import classifier module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classifier import classify_emotions, get_dominant_emotion


class ExpressionEngine:
    """
    Wraps the existing rule-based emotion classifier for use in the
    surveillance pipeline.

    Thread-safe: classify_emotions and get_dominant_emotion are pure
    functions with no shared state.
    """

    def analyze(self, blendshapes: dict) -> dict | None:
        """
        Classify facial expressions from MediaPipe blendshapes.

        Args:
            blendshapes: Dictionary of {blendshape_name: score} from face detection.

        Returns:
            dict with keys:
                - scores: dict of {emotion_key: probability} (7 keys)
                - dominant_key: str (e.g., "happy")
                - dominant_label: str (e.g., "Happy")
                - dominant_emoji: str (e.g., "😊")
                - dominant_score: float (probability of dominant emotion)
            or None if classification fails.
        """
        if not blendshapes:
            return None

        scores = classify_emotions(blendshapes)
        if scores is None:
            return None

        dominant = get_dominant_emotion(scores)
        if dominant is None:
            return None

        return {
            "scores": scores,
            "dominant_key": dominant["key"],
            "dominant_label": dominant["label"],
            "dominant_emoji": dominant["emoji"],
            "dominant_score": dominant["score"],
        }
