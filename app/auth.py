"""
Auth Blueprint — Login, Signup, Face Login, and Session Management.
Protects all existing routes via before_app_request hook.
"""

import base64
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
from liveness import check_texture, check_3d_depth_liveness, check_screen_spoof

auth_bp = Blueprint("auth", __name__)

# Cosine similarity threshold for face login matching (0.60 ensures strangers are NEVER accepted)
FACE_LOGIN_THRESHOLD = 0.60


def _detect_and_get_embedding(img: np.ndarray):
    """Detect face, crop region, extract 512-d ArcFace embedding, and return (embedding, bbox, blendshapes, landmarks)."""
    if img is None or img.size == 0:
        return None, None, {}, []

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
        emb = get_embedding(face_crop)
        return emb, bbox, blendshapes, landmarks

    return None, None, {}, []


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
    """Authenticate using webcam face recognition with multi-layer anti-spoofing."""
    data = request.get_json()
    if not data or not data.get("image"):
        return jsonify({"error": "No image provided"}), 400

    try:
        import time
        import app as main_app

        img_b64 = data["image"].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Extract face embedding, bbox, blendshapes, landmarks
        embedding, bbox, blendshapes, landmarks = _detect_and_get_embedding(img)
        if embedding is None:
            return jsonify({
                "success": False,
                "face_detected": False,
                "error": "No face detected in camera view.",
            }), 400

        face_crop = main_app.crop_face(img, bbox)
        now_ts = time.time()

        # ──────────────────────────────────────────────────────────
        # LAYER 1: 3D Depth Liveness (flat screens have no depth)
        # ──────────────────────────────────────────────────────────
        if landmarks and len(landmarks) >= 10:
            zs = [pt[2] for pt in landmarks]
            z_std = float(np.std(zs))
            z_range = float(max(zs) - min(zs))
            print(f"[Auth L1] 3D Depth: z_std={z_std:.6f} z_range={z_range:.6f}")
            # Phone screens produce flattened 3D: z_std typically < 0.003
            # Real faces produce z_std typically > 0.008
            if z_std < 0.004:
                print(f"[Auth L1] BLOCKED — Flat screen/photo detected (z_std={z_std:.6f})")
                return jsonify({
                    "success": False, "face_detected": True, "bbox": bbox,
                    "reason": "spoof",
                    "error": "⚠️ Spoof detected — Flat screen or printed photo rejected.",
                    "confidence": 0.0
                }), 200

        # ──────────────────────────────────────────────────────────
        # LAYER 2: Screen Spoof Detection (Moiré, glare)
        # ──────────────────────────────────────────────────────────
        if face_crop is not None:
            screen_spoof_res = check_screen_spoof(face_crop)
            print(f"[Auth L2] Screen: fft={screen_spoof_res.get('fft_score', 0):.4f} glare={screen_spoof_res.get('glare_ratio', 0):.4f}")
            if screen_spoof_res["is_spoof"]:
                print(f"[Auth L2] BLOCKED — {screen_spoof_res['reason']}")
                return jsonify({
                    "success": False, "face_detected": True, "bbox": bbox,
                    "reason": "spoof",
                    "error": f"⚠️ Spoof detected — {screen_spoof_res['reason']}.",
                    "confidence": 0.0
                }), 200

        # ──────────────────────────────────────────────────────────
        # LAYER 3: Texture Analysis (LBP + Laplacian)
        # ──────────────────────────────────────────────────────────
        if face_crop is not None:
            texture_res = check_texture(face_crop)
            print(f"[Auth L3] Texture: lap={texture_res['laplacian_var']:.2f} lbp={texture_res['lbp_var']:.4f} glare={texture_res['glare_ratio']:.4f} pass={texture_res['texture_pass']}")
            if not texture_res["texture_pass"]:
                print(f"[Auth L3] BLOCKED — Texture analysis failed (screen/photo texture)")
                return jsonify({
                    "success": False, "face_detected": True, "bbox": bbox,
                    "reason": "spoof",
                    "error": "⚠️ Spoof detected — Abnormal face texture (screen or photo).",
                    "confidence": 0.0
                }), 200

        # ──────────────────────────────────────────────────────────
        # LAYER 4: Multi-Frame Temporal Consistency (anti video replay)
        # Stores grayscale face hash from each frame. Real faces show
        # natural micro-variations; videos shown on phones show 
        # characteristic screen refresh artifacts.
        # ──────────────────────────────────────────────────────────
        spoof_score = 0
        if face_crop is not None and face_crop.size > 0:
            gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray_crop, (32, 32)).astype(np.float32).flatten()

            # Store frame hashes for temporal analysis
            prev_hashes = session.get("_face_frame_hashes", [])
            prev_timestamps = session.get("_face_frame_times", [])

            if prev_hashes and prev_timestamps:
                # Compare with most recent stored frame
                prev_arr = np.array(prev_hashes[-1], dtype=np.float32)
                frame_diff = float(np.mean(np.abs(gray_small - prev_arr)))
                time_gap = now_ts - prev_timestamps[-1]

                print(f"[Auth L4] Temporal: frame_diff={frame_diff:.2f} time_gap={time_gap:.3f}s")

                # If frame is EXACTLY identical (diff < 0.5), it's a static image
                if frame_diff < 0.5 and time_gap > 0.05:
                    spoof_score += 1

            # Keep last 5 hashes (rolling window)
            prev_hashes.append(gray_small.tolist())
            prev_timestamps.append(now_ts)
            if len(prev_hashes) > 5:
                prev_hashes = prev_hashes[-5:]
                prev_timestamps = prev_timestamps[-5:]
            session["_face_frame_hashes"] = prev_hashes
            session["_face_frame_times"] = prev_timestamps

        # ──────────────────────────────────────────────────────────
        # LAYER 5: Mandatory Live Eye-Blink (Rolling 4-Second Window)
        # ──────────────────────────────────────────────────────────
        blink_left = blendshapes.get("eyeBlinkLeft", 0.0)
        blink_right = blendshapes.get("eyeBlinkRight", 0.0)
        max_blink = max(blink_left, blink_right)

        eye_was_closed = session.get("face_eye_closed", False)
        last_blink_time = session.get("last_blink_time", 0.0)

        # Dynamic blink transition: open -> closed (>= 0.22) -> open (<= 0.12)
        if not eye_was_closed and max_blink >= 0.22:
            session["face_eye_closed"] = True
        elif eye_was_closed and max_blink <= 0.12:
            session["face_eye_closed"] = False
            session["last_blink_time"] = now_ts
            last_blink_time = now_ts
            print(f"[Auth L5] 👁️ Live blink transition verified at t={now_ts:.2f}")

        has_fresh_blink = (now_ts - last_blink_time) <= 4.0

        if not has_fresh_blink:
            return jsonify({
                "success": False, "face_detected": True, "bbox": bbox,
                "reason": "blink_required", "blink_required": True,
                "error": "👁️ Real live face required — Please blink your eyes to verify liveness.",
                "confidence": 0.0
            }), 200

        # ──────────────────────────────────────────────────────────
        # LAYER 6: Skin Chrominance Verification
        # Real skin has natural YCrCb values; screens emit different
        # chromatic signatures.
        # ──────────────────────────────────────────────────────────
        if face_crop is not None:
            from liveness import _compute_skin_chroma_score
            skin_pass, skin_ratio = _compute_skin_chroma_score(face_crop)
            print(f"[Auth L6] Skin Chroma: ratio={skin_ratio:.4f} pass={skin_pass}")
            if not skin_pass:
                spoof_score += 1

        # If accumulated spoof_score is high enough, reject
        if spoof_score >= 2:
            print(f"[Auth] Multi-signal spoof rejection: spoof_score={spoof_score}")
            return jsonify({
                "success": False, "face_detected": True, "bbox": bbox,
                "reason": "spoof",
                "error": "⚠️ Spoof detected — Multiple anti-spoofing signals triggered.",
                "confidence": 0.0
            }), 200

        # ──────────────────────────────────────────────────────────
        # FACE MATCHING — Compare against all registered identities
        # ──────────────────────────────────────────────────────────
        best_match_name = None
        best_user_id = None
        best_username = None
        best_score = 0.0

        # 1. Compare against registered employees table (from /register page)
        employees = get_all_employees()
        for emp in employees:
            if emp.get("embedding"):
                try:
                    stored = np.frombuffer(emp["embedding"], dtype=np.float32)
                    if stored.shape[0] == 512:
                        score = cosine_similarity(embedding, stored)
                        if score > best_score:
                            best_score = score
                            best_match_name = emp["full_name"]
                            best_user_id = f"emp_{emp['employee_id']}"
                            best_username = emp["employee_id"]
                except Exception as e:
                    print(f"[Auth] Error comparing employee {emp.get('employee_id')}: {e}")

        # 2. Compare against registered user accounts (from /login signup page)
        users = get_all_users_with_embedding()
        for u in users:
            if u.get("face_embedding"):
                try:
                    stored = np.frombuffer(u["face_embedding"], dtype=np.float32)
                    if stored.shape[0] == 512:
                        score = cosine_similarity(embedding, stored)
                        if score > best_score:
                            best_score = score
                            best_match_name = u["full_name"]
                            best_user_id = u["id"]
                            best_username = u["username"]
                except Exception as e:
                    print(f"[Auth] Error comparing user {u.get('username')}: {e}")

        print(f"[Auth Match] Best: name={best_match_name} score={best_score:.4f} threshold={FACE_LOGIN_THRESHOLD}")

        if best_match_name and best_score >= FACE_LOGIN_THRESHOLD:
            # Clear anti-spoofing session state on successful login
            session.pop("face_eye_closed", None)
            session.pop("face_blink_count", None)
            session.pop("_face_frame_hashes", None)
            session.pop("_face_frame_times", None)

            session["user_id"] = best_user_id
            session["username"] = best_username
            session["full_name"] = best_match_name
            return jsonify({
                "success": True,
                "face_detected": True,
                "bbox": bbox,
                "employee_name": best_match_name,
                "message": f"Welcome back, {best_match_name}!",
                "confidence": round(best_score * 100, 1),
            })
        else:
            is_spoof = bool(best_match_name)
            reason = "spoof" if is_spoof else "unregistered"
            error_msg = (
                f"⚠️ SPOOF DETECTED ({round(best_score*100, 1)}% match). Phone screen or video attack rejected."
                if is_spoof
                else f"Face detected, but unrecognised ({round(best_score*100, 1)}% match)."
            )

            return jsonify({
                "success": False,
                "face_detected": True,
                "bbox": bbox,
                "reason": reason,
                "error": error_msg,
                "confidence": round(best_score * 100, 1),
            }), 401

    except Exception as e:
        print(f"[Auth] Face login error: {e}")
        return jsonify({"error": "Face login processing failed."}), 500


@auth_bp.route("/api/auth/reset_face_session", methods=["POST"])
def api_reset_face_session():
    """Reset all anti-spoofing and blink session state when starting face login."""
    session["face_eye_closed"] = False
    session["face_blink_count"] = 0
    session["last_blink_time"] = 0.0
    session.pop("_face_frame_hashes", None)
    session.pop("_face_frame_times", None)
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
