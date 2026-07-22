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
from liveness import check_texture

auth_bp = Blueprint("auth", __name__)

# Cosine similarity threshold for face login matching
FACE_LOGIN_THRESHOLD = 0.60


def _detect_and_get_embedding(img: np.ndarray):
    """Detect face, crop region, extract 512-d ArcFace embedding, and return (embedding, bbox)."""
    if img is None or img.size == 0:
        return None, None

    bbox = None
    face_crop = None

    # 1. Try MediaPipe detector
    try:
        import app as main_app
        if main_app.detector is not None:
            faces = main_app.detector.detect(img)
            if faces:
                largest = max(faces, key=lambda f: f["bounding_box"]["width"] * f["bounding_box"]["height"])
                bbox = largest["bounding_box"]
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
        return emb, bbox

    return None, None


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
                embedding, _ = _detect_and_get_embedding(img)
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
    """Authenticate using webcam face recognition against both User accounts and Employee records."""
    data = request.get_json()
    if not data or not data.get("image"):
        return jsonify({"error": "No image provided"}), 400

    try:
        img_b64 = data["image"].split(",")[-1]
        img_data = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Extract face embedding with proper face cropping & return bbox
        embedding, bbox = _detect_and_get_embedding(img)
        if embedding is None:
            return jsonify({
                "success": False,
                "face_detected": False,
                "error": "No face detected in camera view.",
            }), 400

        # Perform anti-spoofing texture liveness check on face crop
        import app as main_app
        face_crop = main_app.crop_face(img, bbox)
        if face_crop is not None:
            texture_res = check_texture(face_crop)
            if not texture_res["texture_pass"]:
                print(f"[Auth Liveness] Spoof rejected - Laplacian: {texture_res['laplacian_var']}, LBP: {texture_res['lbp_var']}")
                return jsonify({
                    "success": False,
                    "face_detected": True,
                    "bbox": bbox,
                    "error": "Liveness check failed (anti-spoofing alert). Please present a real face.",
                    "confidence": 0.0
                }), 401

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

        if best_match_name and best_score >= FACE_LOGIN_THRESHOLD:
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
            return jsonify({
                "success": False,
                "face_detected": True,
                "bbox": bbox,
                "error": f"Face detected, but unrecognised ({round(best_score*100, 1)}% match).",
                "confidence": round(best_score * 100, 1),
            }), 401

    except Exception as e:
        print(f"[Auth] Face login error: {e}")
        return jsonify({"error": "Face login processing failed."}), 500


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
