"""
Emotion Classifier — converts 52 MediaPipe blendshape coefficients into
7 emotion probability scores using a rule-based heuristic with dead-zone
gating and winner-take-all sharpening.
"""

# Emotion configuration: labels, emojis, colors (hex for CTk widgets)
EMOTIONS = {
    "happy":     {"label": "Happy",     "emoji": "😊", "color": "#22c55e"},
    "sad":       {"label": "Sad",       "emoji": "😢", "color": "#3b82f6"},
    "angry":     {"label": "Angry",     "emoji": "😠", "color": "#ef4444"},
    "surprised": {"label": "Surprised", "emoji": "😲", "color": "#f59e0b"},
    "neutral":   {"label": "Neutral",   "emoji": "😐", "color": "#94a3b8"},
    "fearful":   {"label": "Fearful",   "emoji": "😨", "color": "#a855f7"},
    "disgusted": {"label": "Disgusted", "emoji": "🤢", "color": "#84cc16"},
}

EMOTION_KEYS = list(EMOTIONS.keys())


def _gate(value: float, threshold: float = 0.12) -> float:
    """Dead-zone threshold: values below the threshold are zeroed out.
    Prevents resting-face noise from triggering emotions like angry."""
    return max(0.0, value - threshold) if value > threshold else 0.0


def _get_bs(blendshapes: dict, name: str) -> float:
    """Safely retrieve a blendshape value by name."""
    return blendshapes.get(name, 0.0)


def classify_emotions(blendshapes: dict) -> dict | None:
    """
    Classify 7 emotions from a dict of {blendshape_name: score}.

    Returns a dict of {emotion_key: probability} or None if input is empty.
    """
    if not blendshapes:
        return None

    # Extract key blendshape values
    mouth_smile_l = _get_bs(blendshapes, "mouthSmileLeft")
    mouth_smile_r = _get_bs(blendshapes, "mouthSmileRight")
    mouth_frown_l = _get_bs(blendshapes, "mouthFrownLeft")
    mouth_frown_r = _get_bs(blendshapes, "mouthFrownRight")
    brow_down_l   = _get_bs(blendshapes, "browDownLeft")
    brow_down_r   = _get_bs(blendshapes, "browDownRight")
    brow_inner_up = _get_bs(blendshapes, "browInnerUp")
    brow_outer_up_l = _get_bs(blendshapes, "browOuterUpLeft")
    brow_outer_up_r = _get_bs(blendshapes, "browOuterUpRight")
    eye_wide_l    = _get_bs(blendshapes, "eyeWideLeft")
    eye_wide_r    = _get_bs(blendshapes, "eyeWideRight")
    jaw_open      = _get_bs(blendshapes, "jawOpen")
    cheek_squint_l = _get_bs(blendshapes, "cheekSquintLeft")
    cheek_squint_r = _get_bs(blendshapes, "cheekSquintRight")
    nose_sneer_l  = _get_bs(blendshapes, "noseSneerLeft")
    nose_sneer_r  = _get_bs(blendshapes, "noseSneerRight")
    mouth_upper_up_l = _get_bs(blendshapes, "mouthUpperUpLeft")
    mouth_upper_up_r = _get_bs(blendshapes, "mouthUpperUpRight")
    mouth_stretch_l = _get_bs(blendshapes, "mouthStretchLeft")
    mouth_stretch_r = _get_bs(blendshapes, "mouthStretchRight")
    mouth_press_l = _get_bs(blendshapes, "mouthPressLeft")
    mouth_press_r = _get_bs(blendshapes, "mouthPressRight")
    eye_squint_l  = _get_bs(blendshapes, "eyeSquintLeft")
    eye_squint_r  = _get_bs(blendshapes, "eyeSquintRight")
    mouth_pucker  = _get_bs(blendshapes, "mouthPucker")
    mouth_shrug_lower = _get_bs(blendshapes, "mouthShrugLower")
    mouth_shrug_upper = _get_bs(blendshapes, "mouthShrugUpper")

    # Derived features (average left/right for symmetry)
    smile       = (mouth_smile_l + mouth_smile_r) / 2
    frown       = (mouth_frown_l + mouth_frown_r) / 2
    brow_down   = (brow_down_l + brow_down_r) / 2
    brow_up     = (brow_outer_up_l + brow_outer_up_r) / 2
    eye_wide    = (eye_wide_l + eye_wide_r) / 2
    cheek_squint = (cheek_squint_l + cheek_squint_r) / 2
    nose_sneer  = (nose_sneer_l + nose_sneer_r) / 2
    mouth_upper_up = (mouth_upper_up_l + mouth_upper_up_r) / 2
    mouth_stretch = (mouth_stretch_l + mouth_stretch_r) / 2
    mouth_press = (mouth_press_l + mouth_press_r) / 2
    eye_squint  = (eye_squint_l + eye_squint_r) / 2
    shrug       = (mouth_shrug_lower + mouth_shrug_upper) / 2

    # --- Gated features: remove resting-face noise ---
    # Gating limits are adjusted to make micro-expressions (like sad or angry)
    # highly sensitive while keeping neutral stable at rest.
    g_brow_down    = _gate(brow_down, 0.12)
    g_eye_squint   = _gate(eye_squint, 0.15)
    g_mouth_press  = _gate(mouth_press, 0.12)
    g_nose_sneer   = _gate(nose_sneer, 0.10)
    g_frown        = _gate(frown, 0.03)  # reduced from 0.08 for high sad sensitivity
    g_smile        = _gate(smile, 0.05)
    g_eye_wide     = _gate(eye_wide, 0.05)
    g_jaw_open     = _gate(jaw_open, 0.05)
    g_brow_up      = _gate(brow_up, 0.05)
    g_brow_inner_up = _gate(brow_inner_up, 0.04)  # reduced from 0.10 for high sad sensitivity
    g_mouth_stretch = _gate(mouth_stretch, 0.08)
    g_mouth_upper_up = _gate(mouth_upper_up, 0.08)
    g_shrug        = _gate(shrug, 0.04)  # new feature for sad detection

    # Compute raw scores for each emotion
    raw = {}

    # Happy: smile + cheek squint, penalized by frown and brow down
    raw["happy"] = max(0.0,
        (g_smile * 1.8 + cheek_squint * 0.4)
        * (1.0 - g_frown * 2.5)
        * (1.0 - g_brow_down * 2.0)
    )

    # Sad: frown + inner brow raise + shrug, penalized by smile
    raw["sad"] = max(0.0,
        (g_frown * 2.0 + g_brow_inner_up * 1.5 + g_shrug * 1.0 + _gate(mouth_pucker, 0.10) * 0.3)
        * (1.0 - g_smile * 4.0)
    )

    # Angry: requires strong brow down as primary signal
    raw["angry"] = max(0.0,
        (g_brow_down * 2.0 + g_mouth_press * 0.5 + g_nose_sneer * 0.8)
        * (1.0 - g_smile * 4.0)
        * (1.0 if g_brow_down > 0.03 else 0.0)
    )

    # Surprised: jaw open + eye wide + brow raise
    raw["surprised"] = max(0.0,
        (g_jaw_open * 1.2 + g_eye_wide * 1.2 + g_brow_up * 0.8)
        - g_brow_down * 1.5
    )

    # Fearful: eye wide + inner brow up + mouth stretch
    raw["fearful"] = max(0.0,
        (g_eye_wide * 0.8 + g_brow_inner_up * 1.0 + g_mouth_stretch * 0.8)
        - g_jaw_open * 0.6
    )

    # Disgusted: nose sneer + upper lip raise
    raw["disgusted"] = max(0.0,
        (g_nose_sneer * 1.6 + g_mouth_upper_up * 1.0)
        * (1.0 - g_smile * 4.0)
    )

    # Neutral: high when overall expressiveness is low
    expressiveness = sum(raw.values())
    raw["neutral"] = max(0.0, 1.0 - expressiveness * 2.0)

    # --- Winner-take-all sharpening (power-of-5) ---
    power = 5
    processed = {k: v ** power for k, v in raw.items()}
    total = sum(processed.values())

    if total > 0:
        scores = {k: v / total for k, v in processed.items()}
    else:
        scores = {k: (1.0 if k == "neutral" else 0.0) for k in raw}

    return scores


def get_dominant_emotion(scores: dict) -> dict | None:
    """Return the emotion with the highest score as a dict with key, label, emoji, color, score."""
    if not scores:
        return None
    key = max(scores, key=scores.get)
    config = EMOTIONS[key]
    return {
        "key": key,
        "label": config["label"],
        "emoji": config["emoji"],
        "color": config["color"],
        "score": scores[key],
    }


def sort_emotions_by_score(scores: dict) -> list[dict]:
    """Return a list of emotion dicts sorted by score descending."""
    result = []
    for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        config = EMOTIONS[key]
        result.append({
            "key": key,
            "label": config["label"],
            "emoji": config["emoji"],
            "color": config["color"],
            "score": score,
        })
    return result


class EmotionSmoother:
    """Exponential Moving Average smoother to prevent rapid flickering."""

    def __init__(self, alpha: float = 0.45):
        self.alpha = alpha
        self.prev_scores: dict | None = None

    def smooth(self, new_scores: dict) -> dict:
        if self.prev_scores is None:
            self.prev_scores = dict(new_scores)
            return new_scores

        smoothed = {}
        for key in new_scores:
            smoothed[key] = (
                self.alpha * new_scores[key]
                + (1 - self.alpha) * self.prev_scores.get(key, 0.0)
            )

        self.prev_scores = dict(smoothed)
        return smoothed

    def reset(self):
        self.prev_scores = None
