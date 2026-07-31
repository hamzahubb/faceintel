"""
Auth Blueprint — Login, Signup, Face Login, and Session Management.
Protects all existing routes via before_app_request hook.
"""

import base64
import time
import numpy as np
import cv2
from functools import wraps
from flask import Blueprint, render_template, request, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash

from database import (
    create_user, get_user_by_username, get_user_by_id,
    get_all_users_with_embedding, get_all_employees, save_employee
)
from recognizer import get_embedding, cosine_similarity
from utils import liveness_pipeline as liveness

auth_bp = Blueprint("auth", __name__)

# Cosine similarity threshold for face login matching.
FACE_LOGIN_THRESHOLD = 0.48

# Below this, the face is confidently nobody on file -> genuine unregistered
# person. Between this and FACE_LOGIN_THRESHOLD the face partially matches an
# enrolled identity, which is what a screen replay of that person looks like.
# Measured: true strangers peak at 0.43 and mostly sit under 0.32, while genuine
# live matches start at 0.638 — so this band belongs to degraded replays.
PARTIAL_MATCH_FLOOR = 0.35

# Consecutive frames a registered identity must hold before login is granted.
MATCH_STREAK_REQUIRED = 2

# Per-client identity-match streaks. All liveness state lives in the liveness
# pipeline; this tracks recognition only, keeping the two concerns separate.
_match_state = {}


def _match_streak(client_key: str, score: float) -> int:
    """Count consecutive frames this client has matched an identity."""
    st = _match_state.setdefault(client_key, {"streak": 0, "scores": []})
    st["scores"] = (st["scores"] + [score])[-4:]
    # A wobbling score means the match has not settled; make it start over.
    if len(st["scores"]) >= 3 and float(np.std(st["scores"][-3:])) > 0.15:
        st["streak"] = 0
    else:
        st["streak"] += 1
    return st["streak"]


def _reset_match_state(client_key: str):
    _match_state.pop(client_key, None)


def _detect_face(img: np.ndarray):
    """
    STAGE 1 — face detection only.

    Deliberately does NOT extract an embedding. Recognition is a separate stage
    that must not run until the liveness pipeline has cleared the frame.

    Returns {"bbox", "face_crop", "blendshapes", "landmarks"} or None.
    """
    if img is None or img.size == 0:
        return None

    bbox = None
    face_crop = None
    blendshapes = {}
    landmarks = []

    # 1. Try MediaPipe detector
    try:
        import app as main_app
        if main_app.detector is not None:
            faces = main_app.detector.detect(img)
            if faces:
                largest = max(faces, key=lambda f: f["bounding_box"]["width"] * f["bounding_box"]["height"])
                bbox = largest["bounding_box"]
                blendshapes = largest.get("blendshapes", {})
                landmarks = largest.get("landmarks", [])
                face_crop = main_app.crop_face(img, bbox)
    except Exception as e:
        print(f"[Auth] MediaPipe detection error: {e}")

    # 2. Fallback: Haar Cascade
    if face_crop is None:
        try:
            import app as main_app
            main_app.ensure_haar_cascade()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = main_app.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                bbox = {"originX": int(x), "originY": int(y), "width": int(w), "height": int(h)}
                face_crop = main_app.crop_face(img, bbox)
        except Exception as e:
            print(f"[Auth] Haar detection error: {e}")

    # 3. Fallback: center crop
    if face_crop is None:
        h, w = img.shape[:2]
        min_dim = min(h, w)
        cy, cx = h // 2, w // 2
        bbox = {"originX": cx - min_dim//2, "originY": cy - min_dim//2, "width": min_dim, "height": min_dim}
        face_crop = img[max(0, cy - min_dim//2):min(h, cy + min_dim//2), max(0, cx - min_dim//2):min(w, cx + min_dim//2)]

    if face_crop is not None and face_crop.size > 0:
        return {"bbox": bbox, "face_crop": face_crop,
                "blendshapes": blendshapes, "landmarks": landmarks}

    return None


def _recognise(face_crop: np.ndarray):
    """
    STAGE 3 — identity matching.

    Only ever called on a frame the liveness pipeline has ruled LIVE.
    Returns (full_name, user_id, username, score).
    """
    embedding = get_embedding(face_crop)
    if embedding is None:
        return None, None, None, 0.0

    best_name = best_id = best_username = None
    best_score = 0.0

    for emp in get_all_employees():
        if not emp.get("embedding"):
            continue
        try:
            stored = np.frombuffer(emp["embedding"], dtype=np.float32)
            if stored.shape[0] != 512:
                continue
            score = cosine_similarity(embedding, stored)
            if score > best_score:
                best_score = score
                best_name = emp["full_name"]
                best_id = f"emp_{emp['employee_id']}"
                best_username = emp["employee_id"]
        except Exception as e:
            print(f"[Auth] Error comparing employee {emp.get('employee_id')}: {e}")

    for u in get_all_users_with_embedding():
        if not u.get("face_embedding"):
            continue
        try:
            stored = np.frombuffer(u["face_embedding"], dtype=np.float32)
            if stored.shape[0] != 512:
                continue
            score = cosine_similarity(embedding, stored)
            if score > best_score:
                best_score = score
                best_name = u["full_name"]
                best_id = u["id"]
                best_username = u["username"]
        except Exception as e:
            print(f"[Auth] Error comparing user {u.get('username')}: {e}")

    return best_name, best_id, best_username, best_score


def _detect_and_get_embedding(img: np.ndarray):
    """Back-compat shim for signup, which has no liveness requirement."""
    det = _detect_face(img)
    if det is None:
        return None, None, {}, []
    return (get_embedding(det["face_crop"]), det["bbox"],
            det["blendshapes"], det["landmarks"])


# ──────────────────────────────────────────────────────────────
# Before-request hook — protects ALL routes automatically
# ──────────────────────────────────────────────────────────────

@auth_bp.before_app_request
def require_login():
    """Redirect unauthenticated users to /login for all protected routes."""
    allowed_prefixes = ("/login", "/signup", "/api/auth/", "/static/")
    if any(request.path.startswith(p) for p in allowed_prefixes):
        return None
    if request.path == "/favicon.ico":
        return None
    if "user_id" not in session:
        return redirect("/login")
    return None


# ──────────────────────────────────────────────────────────────
# Page Routes
# ──────────────────────────────────────────────────────────────

@auth_bp.route("/login")
def login_page():
    """Render the login/signup page."""
    if "user_id" in session:
        return redirect("/")
    return render_template("login.html")


@auth_bp.route("/signup")
def signup_page():
    """Render the login page in signup mode."""
    if "user_id" in session:
        return redirect("/")
    return render_template("login.html", signup=True)


# ──────────────────────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────────────────────

@auth_bp.route("/api/auth/signup", methods=["POST"])
def api_signup():
    """Create a new user account with optional face embedding."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""
    full_name = (data.get("full_name") or "").strip()
    face_image_b64 = data.get("face_image")

    if not username or not password or not full_name:
        return jsonify({"error": "Username, password, and full name are required"}), 400
    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    existing = get_user_by_username(username)
    if existing:
        return jsonify({"error": "Username already taken"}), 409

    pw_hash = generate_password_hash(password)

    # Extract face embedding from captured face image
    face_embedding_bytes = None
    if face_image_b64:
        try:
            img_data = base64.b64decode(face_image_b64.split(",")[-1])
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                embedding, _, _, _ = _detect_and_get_embedding(img)
                if embedding is not None:
                    face_embedding_bytes = embedding.tobytes()
        except Exception as e:
            print(f"[Auth] Face embedding extraction failed during signup: {e}")

    success = create_user(username, pw_hash, full_name, face_embedding_bytes)
    if not success:
        return jsonify({"error": "Failed to create account. Please try again."}), 500

    # Also register as employee so they appear in "registered employees" list
    try:
        image_count = 1 if face_embedding_bytes is not None else 0
        save_employee(
            employee_id=username,
            full_name=full_name,
            department="User Account",
            embedding_bytes=face_embedding_bytes,
            image_count=image_count
        )
        # Clear main app's employee cache
        import app as main_app
        main_app._cache_last_refresh = 0
    except Exception as e:
        print(f"[Auth Error] Failed to create corresponding employee: {e}")

    user = get_user_by_username(username)
    if user:
        session["user_id"] = user["id"]
        session["username"] = user["username"]
        session["full_name"] = user["full_name"]

    return jsonify({
        "success": True,
        "message": "Account created successfully!",
        "has_face": face_embedding_bytes is not None,
    })


@auth_bp.route("/api/auth/login", methods=["POST"])
def api_login():
    """Authenticate with username and password."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    username = (data.get("username") or "").strip().lower()
    password = data.get("password") or ""

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    user = get_user_by_username(username)
    if not user:
        return jsonify({"error": "Invalid username or password"}), 401

    if not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid username or password"}), 401

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["full_name"] = user["full_name"]

    return jsonify({
        "success": True,
        "message": f"Welcome back, {user['full_name']}!",
    })


@auth_bp.route("/api/auth/face_login", methods=["POST"])
def api_face_login():
    """
    Face login — a single deterministic pipeline.

        Face Detection
              |
        Liveness / Spoof Verification      <- gatekeeper, no identity involved
              |
        SPOOF  -> stop, spoof modal, recognition never runs
        PENDING-> keep watching
        LIVE   -> Face Recognition
                    |
              registered -> log in
              unknown    -> unregistered modal

    Exactly one modal can result from a frame: the branches are mutually
    exclusive by construction, not by precedence patching.
    """
    data = request.get_json()
    if not data or not data.get("image"):
        return jsonify({"error": "No image provided"}), 400

    try:
        img_b64 = data["image"].split(",")[-1]
        img_array = np.frombuffer(base64.b64decode(img_b64), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        client_key = request.remote_addr or "unknown"

        # ── STAGE 1: Face detection ───────────────────────────────
        det = _detect_face(img)
        if det is None:
            return jsonify({
                "success": False, "face_detected": False,
                "error": "No face detected in camera view.",
            }), 400

        bbox = det["bbox"]
        face_crop = det["face_crop"]

        if face_crop.shape[0] < 45 or face_crop.shape[1] < 45:
            return jsonify({
                "success": False, "face_detected": False,
                "error": "Face region too small.",
            }), 200

        # Ignore dark camera-warmup frames before judging anything
        if float(np.mean(cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY))) < 18.0:
            return jsonify({
                "success": False, "face_detected": False,
                "error": "Camera stream warming up...",
            }), 200

        # ── STAGE 2: Liveness / spoof verification ────────────────
        # The gatekeeper. Knows nothing about identity.
        verdict = liveness.evaluate(
            client_key=client_key,
            frame=img,
            bbox=bbox,
            face_crop=face_crop,
            blendshapes=det["blendshapes"],
            landmarks=det["landmarks"],
        )
        print(f"[Auth] Liveness verdict: {verdict.state} - {verdict.reason}", flush=True)

        # SPOOF -> stop here. Recognition is never invoked, so the unregistered
        # modal is structurally unreachable for a spoof.
        if verdict.is_spoof:
            _reset_match_state(client_key)
            return jsonify({
                "success": False, "face_detected": True, "bbox": bbox,
                "reason": "spoof",
                "error": f"⚠️ Spoof detected — {verdict.reason}.",
                "confidence": 0.0,
            }), 200

        # PENDING -> gathering evidence; no identity decision yet.
        # Report what is actually being waited on rather than always asking for
        # a blink, which is misleading while frames are still being collected.
        if verdict.is_pending:
            if verdict.detail.get("needed"):
                msg = "🔒 Verifying liveness — hold still..."
            elif "blink" in verdict.reason.lower():
                msg = "👁️ Please blink naturally to confirm you are live."
            else:
                msg = "🔒 Liveness inconclusive — move into better light and hold still."
            return jsonify({
                "success": False, "face_detected": True, "bbox": bbox,
                "reason": "blink_required",
                "error": msg,
                "confidence": 0.0,
            }), 200

        # ── STAGE 3: Face recognition (LIVE frames only) ──────────
        name, user_id, username, score = _recognise(face_crop)
        print(f"[Auth] Recognition: name={name} score={score:.4f} "
              f"threshold={FACE_LOGIN_THRESHOLD}", flush=True)

        # ── STAGE 4: Decision ─────────────────────────────────────
        # Three outcomes, mutually exclusive by score band.
        #
        # Measured on this dataset (scratch/match_distributions.py), after
        # excluding two duplicate enrollments that were inflating the tail:
        #     genuine live match   p5   = 0.638   -> logs in
        #     true stranger        max  = 0.43, typically < 0.32
        #     a replay of an enrolled user degrades into the band between
        #
        # So a score in [PARTIAL_MATCH_FLOOR, FACE_LOGIN_THRESHOLD) is too high
        # to be a stranger and too low to be a live enrolled user — it is the
        # signature of an enrolled face arriving through a screen. Reporting
        # that as "Unregistered Person" is wrong and is what caused the two
        # modals to overlap. It is treated as a spoof instead.
        if not name or score < PARTIAL_MATCH_FLOOR:
            # Confidently nobody on file: a genuine unregistered person.
            _reset_match_state(client_key)
            return jsonify({
                "success": False, "face_detected": True, "bbox": bbox,
                "reason": "unregistered",
                "error": f"Face detected, but unregistered ({round(score * 100, 1)}% match).",
                "confidence": round(score * 100, 1),
            }), 200

        if score < FACE_LOGIN_THRESHOLD:
            # Partial match to an enrolled identity — degraded, as a screen
            # replay degrades it. Never the unregistered modal.
            _reset_match_state(client_key)
            liveness.flag_partial_match(client_key)
            print(f"[Auth] Partial match {score:.3f} to '{name}' — treating as replay, "
                  f"not unregistered", flush=True)
            return jsonify({
                "success": False, "face_detected": True, "bbox": bbox,
                "reason": "spoof",
                "error": "⚠️ Spoof detected — Face matches an enrolled user only "
                         "partially, consistent with a screen replay.",
                "confidence": 0.0,
            }), 200

        # Registered. Require a couple of consistent frames so a borderline
        # match cannot log in off a single lucky frame.
        streak = _match_streak(client_key, score)
        if streak < MATCH_STREAK_REQUIRED:
            return jsonify({
                "success": False, "face_detected": True, "bbox": bbox,
                "reason": "verifying",
                "error": f"🔒 Verifying identity... ({streak}/{MATCH_STREAK_REQUIRED})",
                "confidence": round(score * 100, 1),
            }), 200

        _reset_match_state(client_key)
        liveness.clear_client(client_key)   # full drop: this login succeeded

        session["user_id"] = user_id
        session["username"] = username
        session["full_name"] = name
        return jsonify({
            "success": True,
            "face_detected": True,
            "bbox": bbox,
            "employee_name": name,
            "message": f"Welcome back, {name}!",
            "confidence": round(score * 100, 1),
        })

    except Exception as e:
        import traceback
        print(f"[Auth] Face login error: {e}", flush=True)
        traceback.print_exc()
        return jsonify({"error": "Face login processing failed."}), 500



@auth_bp.route("/api/auth/reset_face_session", methods=["POST"])
def api_reset_face_session():
    """Clear all per-client liveness and recognition state for a fresh attempt."""
    client_key = request.remote_addr or "unknown"
    liveness.reset_client(client_key)
    _reset_match_state(client_key)
    return jsonify({"success": True})


@auth_bp.route("/api/auth/logout", methods=["POST"])
def api_logout():
    """Clear session and log out."""
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully."})


@auth_bp.route("/api/auth/me", methods=["GET"])
def api_me():
    """Return the current logged-in user info."""
    if "user_id" not in session:
        return jsonify({"error": "Not logged in"}), 401
    return jsonify({
        "user_id": session["user_id"],
        "username": session["username"],
        "full_name": session["full_name"],
    })
