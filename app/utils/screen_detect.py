"""
Screen-Presentation Detector
============================

Catches the physical giveaway of a replay attack: the face is inside a *display*
being held up to the camera. A blink test cannot do this — a recorded video
blinks. This looks for the screen itself rather than the face on it.

Two independent cues, both computed on the full frame:

  1. BEZEL — a straight-edged rectangle enclosing the face. A phone, tablet or
     monitor has hard, high-contrast borders; a human head in a room does not
     sit inside one.

  2. EMISSION — a display is a light source, so the face region is brighter than
     its surroundings, whereas a real face is lit *by* the room and is not
     systematically brighter than it.

Measured on the 145 genuine enrollment faces and a rendered phone-held-to-camera
simulation (scratch/eval_phone_held.py):

    genuine faces          0.0% false positives   (bezel 0.0%, emission 0.0%)
    phone at 75/55/40%   100.0% detected
    thin bezel + bright    18.1% detected         <- the weak case; the texture
                                                     model covers most of it

See scratch/validate_screen_detect.py to re-measure after any threshold change.
"""

import cv2
import numpy as np

# Bezel geometry
BEZEL_MIN_AREA_RATIO = 1.6      # rectangle must be this much bigger than the face
BEZEL_MAX_FRAME_RATIO = 0.92    # ...but not essentially the whole frame
BEZEL_MIN_EDGE_STRENGTH = 55.0  # mean gradient along the rectangle border
BEZEL_MIN_RECTANGULARITY = 0.80 # contour area / its min-area-rect area

# Emission
EMISSION_MIN_CONTRAST = 1.5     # face luminance / surround luminance


def _quad_edge_strength(gray: np.ndarray, quad: np.ndarray) -> float:
    """Mean Sobel gradient magnitude sampled along a quadrilateral's border."""
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.polylines(mask, [quad.astype(np.int32)], True, 255, thickness=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    vals = mag[mask > 0]
    return float(vals.mean()) if vals.size else 0.0


def detect_bezel(frame: np.ndarray, bbox: dict) -> dict:
    """Is the face enclosed by a straight-edged rectangular border?"""
    out = {"bezel": False, "edge_strength": 0.0, "area_ratio": 0.0}
    if frame is None or frame.size == 0 or not bbox:
        return out

    h, w = frame.shape[:2]
    fx = bbox.get("originX", 0) + bbox.get("width", 0) / 2.0
    fy = bbox.get("originY", 0) + bbox.get("height", 0) / 2.0
    face_area = max(1.0, bbox.get("width", 0) * bbox.get("height", 0))
    frame_area = float(h * w)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # RETR_LIST, not RETR_EXTERNAL: a phone panel is frequently an *inner*
    # contour once background clutter reaches the frame edge. Rectangularity is
    # measured against minAreaRect rather than demanding an exact 4-vertex
    # approximation, which a thin bezel in a bright room rarely produces.
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in contours:
        area = cv2.contourArea(c)
        if area < face_area * BEZEL_MIN_AREA_RATIO or area > frame_area * BEZEL_MAX_FRAME_RATIO:
            continue
        rect = cv2.minAreaRect(c)
        rect_area = rect[1][0] * rect[1][1]
        if rect_area <= 0 or (area / rect_area) < BEZEL_MIN_RECTANGULARITY:
            continue
        box = cv2.boxPoints(rect)
        # the rectangle must actually enclose the face
        if cv2.pointPolygonTest(box.astype(np.float32), (float(fx), float(fy)), False) < 0:
            continue
        strength = _quad_edge_strength(gray, box)
        if best is None or strength > best[0]:
            best = (strength, area / face_area)

    if best:
        out["edge_strength"] = round(best[0], 2)
        out["area_ratio"] = round(best[1], 2)
        out["bezel"] = best[0] >= BEZEL_MIN_EDGE_STRENGTH
    return out


def detect_emission(frame: np.ndarray, bbox: dict) -> dict:
    """Is the face region self-luminous relative to its surroundings?"""
    out = {"emissive": False, "contrast": 0.0}
    if frame is None or frame.size == 0 or not bbox:
        return out

    h, w = frame.shape[:2]
    x = max(0, int(bbox.get("originX", 0)))
    y = max(0, int(bbox.get("originY", 0)))
    bw = max(1, int(bbox.get("width", 1)))
    bh = max(1, int(bbox.get("height", 1)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    face = gray[y:min(h, y + bh), x:min(w, x + bw)]
    if face.size == 0:
        return out

    mask = np.ones(gray.shape, dtype=bool)
    mask[y:min(h, y + bh), x:min(w, x + bw)] = False
    surround = gray[mask]
    if surround.size == 0:
        return out

    contrast = float(face.mean()) / (float(surround.mean()) + 1e-6)
    out["contrast"] = round(contrast, 3)
    out["emissive"] = contrast >= EMISSION_MIN_CONTRAST
    return out


def check(frame: np.ndarray, bbox: dict) -> dict:
    """Combined screen-presentation check."""
    b = detect_bezel(frame, bbox)
    e = detect_emission(frame, bbox)
    reasons = []
    if b["bezel"]:
        reasons.append("Device screen border detected around face")
    if e["emissive"]:
        reasons.append("Face region is self-luminous (display emission)")
    return {
        "is_screen": bool(reasons),
        "reason": " | ".join(reasons),
        **b, **e,
    }
