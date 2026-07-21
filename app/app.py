"""
Flask Server — Infigo FaceIntel
Serves the web interface and provides APIs for:
  - /process_frame  : real-time face detection + liveness + recognition + wave checkout + emotions
  - /register       : employee registration page
  - /attendance     : attendance dashboard page
  - /api/register_employee : register a new employee with face images
  - /api/employees  : list registered employees
  - /api/delete_employee : remove an employee
  - /api/attendance : get attendance data for a date
  - /api/reset_liveness : reset session tracking state
"""

import os
import json
import base64
import threading
import time
from datetime import datetime, date, timedelta
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, make_response, Response

from classifier import classify_emotions, get_dominant_emotion
from detector import (
    download_model, download_hand_model,
    FaceDetector, HandDetector,
    MODEL_PATH, HAND_MODEL_PATH,
    vision,
    MODEL_DIR,
)
from database import (
    init_tables, save_employee, get_all_employees, get_employee_list,
    delete_employee, employee_exists, log_recognition,
    check_in_employee, update_check_out, get_attendance_by_date,
    get_all_employee_ids, get_employee_attendance_today,
    get_all_logs,
)
from recognizer import get_embedding, compare_with_employees, compute_average_embedding, save_face_images, cosine_similarity
from werkzeug.utils import secure_filename
from gesture import WaveTracker
from tracker import MultiFaceTracker
from auth import auth_bp

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.secret_key = 'infigo-faceintel-secret-2026'
app.register_blueprint(auth_bp)

# ──────────────────────────────────────────────────────────────
# Global State
# ──────────────────────────────────────────────────────────────
model_lock = threading.Lock()
hand_lock = threading.Lock()
detector = None
hand_detector = None

HAAR_CASCADE_PATH = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")
face_cascade = None

def ensure_haar_cascade():
    global face_cascade
    if face_cascade is not None:
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(HAAR_CASCADE_PATH):
        print("[AI] Downloading Haar Cascade XML...")
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        import urllib.request
        urllib.request.urlretrieve(url, HAAR_CASCADE_PATH)
        print("[AI] Haar Cascade XML downloaded.")
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# Cache of employee embeddings (refreshed periodically)
_employee_cache = []
_cache_lock = threading.Lock()
_cache_last_refresh = 0
CACHE_REFRESH_INTERVAL = 10  # seconds

# Logging throttle: prevent spamming logs for the same person
_last_log_time = {}
LOG_THROTTLE_SECONDS = 5

# Attendance throttle: prevent spamming attendance DB updates
_last_attendance_update = {}
ATTENDANCE_THROTTLE_SECONDS = 60

# Multi-Person Face and Hand Tracker instance
_face_tracker = MultiFaceTracker()
_recognition_lock = threading.Lock()
_pipeline_lock = threading.Lock()  # Lock to prevent race conditions on global face tracker
RECOGNITION_THROTTLE_FRAMES = 5   # Run recognition every ~5 frames for faster response seconds)

# Office hours configuration (for late detection)
OFFICE_START_HOUR = 9   # 09:00 AM
OFFICE_START_MINUTE = 0


def init_detector():
    """Download models and initialize both face and hand detectors."""
    global detector, hand_detector
    print("[AI] Initializing face landmarker...")
    download_model(progress_callback=print)
    detector = FaceDetector(MODEL_PATH)
    print("[AI] Face landmarker ready!")

    print("[AI] Initializing hand landmarker...")
    download_hand_model(progress_callback=print)
    hand_detector = HandDetector(HAND_MODEL_PATH)
    print("[AI] Hand landmarker ready!")


def refresh_employee_cache():
    """Reload employee embeddings from the database if stale."""
    global _employee_cache, _cache_last_refresh
    now = time.time()
    if now - _cache_last_refresh < CACHE_REFRESH_INTERVAL:
        return
    with _cache_lock:
        _employee_cache = get_all_employees()
        _cache_last_refresh = now
        print(f"[Cache] Refreshed employee cache: {len(_employee_cache)} employees loaded.")


# ──────────────────────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────────────────────
print("[DB] Initializing database tables...")
init_tables()
init_detector()
refresh_employee_cache()


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def decode_base64_image(base64_str: str) -> np.ndarray | None:
    """Decode a base64 image string from the browser into a BGR OpenCV image."""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"[Error] Failed to decode base64 frame: {e}")
        return None


def crop_face(frame: np.ndarray, bounding_box: dict, padding: float = 0.2) -> np.ndarray | None:
    """Crop the face region from a frame using the bounding box with extra padding."""
    h, w = frame.shape[:2]
    x = bounding_box["originX"]
    y = bounding_box["originY"]
    bw = bounding_box["width"]
    bh = bounding_box["height"]

    # Add padding for better embedding quality
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return None

    return face_crop


def _determine_attendance_status() -> str:
    """Determine if the current time is 'present' or 'late'."""
    now = datetime.now()
    office_start = now.replace(hour=OFFICE_START_HOUR, minute=OFFICE_START_MINUTE, second=0, microsecond=0)
    if now > office_start:
        return "late"
    return "present"


# ──────────────────────────────────────────────────────────────
# Routes — Pages
# ──────────────────────────────────────────────────────────────
@app.route("/")
def index():
    """Serve the main detection web page."""
    response = make_response(render_template("index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/register")
def register_page():
    """Serve the employee registration web page."""
    return render_template("register.html")


@app.route("/attendance")
def attendance_page():
    """Serve the attendance dashboard web page."""
    return render_template("attendance.html")


_attendance_cache = {}
_attendance_cache_time = {}

def _get_cached_attendance_today(emp_id: str, force_refresh: bool = False):
    """Retrieve today's attendance record with a 3-second in-memory TTL cache."""
    now = time.time()
    if not force_refresh and emp_id in _attendance_cache:
        if now - _attendance_cache_time.get(emp_id, 0) < 3.0:
            return _attendance_cache[emp_id]
    record = get_employee_attendance_today(emp_id)
    _attendance_cache[emp_id] = record
    _attendance_cache_time[emp_id] = now
    return record


# ──────────────────────────────────────────────────────────────
# API — Real-Time Frame Processing
# ──────────────────────────────────────────────────────────────
@app.route("/process_frame", methods=["POST"])
def process_frame():
    """
    Multi-Person pipeline:
      1. MediaPipe face detection on up to 4 faces
      2. Multi-face tracking to associate detections across frames
      3. Throttled recognition per face track
      4. Hand detection + closest-face association for wave-to-checkout
      5. Attendance log + emotion snapshot per face track
      6. Return a list of all active faces
    """
    global detector, hand_detector, _face_tracker

    if detector is None:
        return jsonify({"error": "AI model is not ready yet."}), 503

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Missing image data."}), 400

    frame = decode_base64_image(data["image"])
    if frame is None:
        return jsonify({"error": "Invalid image data."}), 400

    with _pipeline_lock:
        # Step 1: MediaPipe multi-face detection
        detected_faces = []
        with model_lock:
            try:
                detected_faces = detector.detect(frame)
            except Exception as e:
                print(f"[Error] MediaPipe face detection failed: {e}")

        if not detected_faces:
            _face_tracker.reset()
            return jsonify({"faces": []})

        # Step 2: Update tracks with detected faces
        active_tracks = _face_tracker.update_tracks(detected_faces)

        # Step 3: Run hand detection ONLY if a recognized face is on screen
        hand_landmarks_list = []
        if hand_detector is not None:
            needs_hand_scan = any(
                t.last_recognized_person.get("is_recognized") for t in active_tracks
            )
            if needs_hand_scan:
                with hand_lock:
                    try:
                        hand_landmarks_list = hand_detector.detect(frame)
                    except Exception as e:
                        print(f"[Error] MediaPipe hand detection failed: {e}")

        h, w = frame.shape[:2]
        _face_tracker.associate_hands(hand_landmarks_list, w, h)

        faces_response = []

        # Step 4: Process recognition and attendance for each face
        for face_data, track in zip(detected_faces, active_tracks):
            # Crop face for embedding
            face_crop = crop_face(frame, face_data["bounding_box"])

            # Emotions
            scores = classify_emotions(face_data["blendshapes"])
            dominant = get_dominant_emotion(scores) if scores else None

            # Face Recognition (throttled per face track)
            with _recognition_lock:
                should_recognize = (
                    not track.last_recognized_person["is_recognized"] or
                    track.recognition_frame_counter >= RECOGNITION_THROTTLE_FRAMES or
                    track.recognition_frame_counter == 0
                )

                if should_recognize and face_crop is not None:
                    track.recognition_frame_counter = 0
                    try:
                        embedding = get_embedding(face_crop)
                        if embedding is not None:
                            refresh_employee_cache()
                            with _cache_lock:
                                employees = list(_employee_cache)
                            match = compare_with_employees(embedding, employees)
                            if match:
                                track.last_recognized_person = {
                                    "employee_id": match["employee_id"],
                                    "employee_name": match["full_name"],
                                    "department": match.get("department", ""),
                                    "recognition_confidence": match["confidence"],
                                    "is_recognized": True,
                                }
                                if dominant:
                                    _throttled_log(
                                        match["employee_id"],
                                        match["full_name"],
                                        dominant["label"],
                                        match["confidence"],
                                    )
                            else:
                                track.last_recognized_person = {
                                    "employee_id": None,
                                    "employee_name": "Unknown Person",
                                    "department": None,
                                    "recognition_confidence": 0.0,
                                    "is_recognized": False,
                                }
                    except Exception as e:
                        print(f"[Error] Face recognition failed for track {track.track_id}: {e}")
                else:
                    track.recognition_frame_counter += 1

                person_info = track.last_recognized_person.copy()

            # Attendance + Wave Checkout logic per face track
            attendance_info = {
                "attendance_action": None,     # "check_in", "already_in", "checked_out"
                "requires_wave": False,
                "wave_detected": False,
                "hand_visible": False,
                "check_in_time": None,
                "check_out_time": None,
            }

            if person_info["is_recognized"]:
                emp_id = person_info["employee_id"]
                today_record = _get_cached_attendance_today(emp_id)

                if today_record is None:
                    # NOT checked in yet today — auto check-in immediately
                    _handle_check_in(emp_id, person_info["employee_name"], person_info.get("department", ""))
                    attendance_info["attendance_action"] = "check_in"
                    attendance_info["requires_wave"] = False
                    refreshed = _get_cached_attendance_today(emp_id, force_refresh=True)
                    if refreshed:
                        attendance_info["check_in_time"] = refreshed["check_in"]

                elif today_record["check_out"] is None:
                    # Already checked in, no check-out yet today
                    attendance_info["attendance_action"] = "already_in"
                    attendance_info["requires_wave"] = True
                    attendance_info["check_in_time"] = today_record["check_in"]
                    attendance_info["hand_visible"] = track.hand_visible
                    attendance_info["wave_detected"] = track.wave_detected

                    if track.wave_detected:
                        update_check_out(emp_id)
                        attendance_info["attendance_action"] = "checked_out"
                        attendance_info["requires_wave"] = False
                        refreshed = _get_cached_attendance_today(emp_id, force_refresh=True)
                        if refreshed:
                            attendance_info["check_out_time"] = refreshed["check_out"]
                        print(f"[Attendance] 👋 Wave checkout for {person_info['employee_name']} ({emp_id})")

                else:
                    # Already checked in and checked out
                    attendance_info["attendance_action"] = "checked_out"
                    attendance_info["requires_wave"] = False
                    attendance_info["check_in_time"] = today_record["check_in"]
                    attendance_info["check_out_time"] = today_record["check_out"]

            faces_response.append({
                "track_id": track.track_id,
                "bounding_box": face_data["bounding_box"],
                "landmarks": face_data["landmarks"],
                "scores": scores,
                "dominant": dominant,
                "attendance": attendance_info,
                **person_info,
            })

        return jsonify({"faces": faces_response})

def _throttled_log(employee_id: str, name: str, expression: str, confidence: float):
    """Log recognition to DB, but no more than once per LOG_THROTTLE_SECONDS per person."""
    global _last_log_time
    now = time.time()
    last = _last_log_time.get(employee_id, 0)
    if now - last >= LOG_THROTTLE_SECONDS:
        log_recognition(employee_id, name, expression, confidence)
        _last_log_time[employee_id] = now


def _handle_check_in(employee_id: str, employee_name: str, department: str):
    """Handle automatic attendance check-in with throttling."""
    global _last_attendance_update
    now = time.time()
    last = _last_attendance_update.get(employee_id, 0)

    if now - last < ATTENDANCE_THROTTLE_SECONDS:
        return

    _last_attendance_update[employee_id] = now
    status = _determine_attendance_status()
    check_in_employee(employee_id, employee_name, department, status)


@app.route("/api/reset_liveness", methods=["POST"])
def api_reset_liveness():
    """Reset the tracking states."""
    global _face_tracker
    _face_tracker.reset()
    return jsonify({"success": True, "message": "All session trackers reset."})


# ──────────────────────────────────────────────────────────────
# API — Employee Registration
# ──────────────────────────────────────────────────────────────
@app.route("/api/register_employee", methods=["POST"])
def api_register_employee():
    """
    Register a new employee.
    Expects JSON with:
      - employee_id: str
      - full_name: str
      - department: str (optional)
      - images: list of base64 image strings (20-30 face captures)
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request data."}), 400

    emp_id = data.get("employee_id", "").strip()
    full_name = data.get("full_name", "").strip()
    department = data.get("department", "").strip()
    images_b64 = data.get("images", [])

    # Validation
    if not emp_id:
        return jsonify({"error": "Employee ID is required."}), 400
    if not full_name:
        return jsonify({"error": "Full Name is required."}), 400
    if len(images_b64) < 5:
        return jsonify({"error": f"At least 5 face images required. Got {len(images_b64)}."}), 400

    if employee_exists(emp_id):
        return jsonify({"error": f"Employee ID '{emp_id}' is already registered."}), 409

    # Decode and crop faces from base64 frames
    face_images = []
    skipped_count = 0
    for b64 in images_b64:
        img = decode_base64_image(b64)
        if img is not None:
            detect_res = None
            with model_lock:
                try:
                    detect_res = detector.detect(img)
                except Exception as e:
                    print(f"[Registration Error] MediaPipe detection failed: {e}")

            if detect_res:
                crop = crop_face(img, detect_res[0]["bounding_box"])
                if crop is not None:
                    face_images.append(crop)
                else:
                    skipped_count += 1
            else:
                skipped_count += 1

    if len(face_images) < 5:
        return jsonify({
            "error": f"Could not detect faces in enough frames. Got {len(face_images)} valid faces, skipped {skipped_count}."
        }), 400

    print(f"[Registration] Processing {len(face_images)} face crops for '{full_name}' ({emp_id}). Skipped {skipped_count} frames.")

    saved_paths = save_face_images(emp_id, face_images)
    print(f"[Registration] Saved {len(saved_paths)} images to dataset/{emp_id}/")

    avg_embedding = compute_average_embedding(face_images)
    if avg_embedding is None:
        return jsonify({"error": "Failed to extract face embeddings. Ensure clear face visibility."}), 500

    embedding_bytes = avg_embedding.tobytes()
    success = save_employee(emp_id, full_name, department, embedding_bytes, len(face_images))

    if success:
        global _cache_last_refresh
        _cache_last_refresh = 0
        return jsonify({
            "success": True,
            "message": f"Employee '{full_name}' registered successfully with {len(face_images)} face images.",
        })
    else:
        return jsonify({"error": "Database error while saving employee."}), 500


@app.route("/api/employees", methods=["GET"])
def api_employees():
    """Return list of registered employees (without embedding data)."""
    employees = get_employee_list()
    return jsonify({"employees": employees})


@app.route("/api/delete_employee", methods=["POST"])
def api_delete_employee():
    """Delete an employee by employee_id."""
    data = request.get_json()
    if not data or "employee_id" not in data:
        return jsonify({"error": "Missing employee_id."}), 400

    emp_id = data["employee_id"]
    success = delete_employee(emp_id)

    if success:
        global _cache_last_refresh
        _cache_last_refresh = 0
        return jsonify({"success": True, "message": f"Employee '{emp_id}' deleted."})
    else:
        return jsonify({"error": f"Employee '{emp_id}' not found."}), 404


# ──────────────────────────────────────────────────────────────
# API — Attendance
# ──────────────────────────────────────────────────────────────
@app.route("/api/attendance", methods=["GET"])
def api_attendance():
    """
    Get attendance data for a date.
    Query param: ?date=YYYY-MM-DD (defaults to today)
    """
    date_str = request.args.get("date", "")
    if date_str:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400
    else:
        target_date = date.today()

    attendance = get_attendance_by_date(target_date)
    all_employee_ids = get_all_employee_ids()

    present_ids = set()
    late_count = 0
    for record in attendance:
        present_ids.add(record["employee_id"])
        if record.get("status") == "late":
            late_count += 1

    present_count = len(present_ids)
    total_employees = len(all_employee_ids)
    absent_count = total_employees - present_count

    summary = {
        "present": present_count,
        "late": late_count,
        "absent": max(0, absent_count),
        "total": total_employees,
    }

    return jsonify({
        "attendance": attendance,
        "summary": summary,
    })


# ──────────────────────────────────────────────────────────────
# API — Video Detection
# ──────────────────────────────────────────────────────────────
@app.route("/video_detection", methods=["GET"])
def video_detection():
    return render_template("video_detection.html")


verification_detector = None
verification_lock = threading.Lock()

def ensure_verification_detector():
    global verification_detector
    with verification_lock:
        if verification_detector is not None:
            return
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        download_model(progress_callback=print)
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.55,
            min_face_presence_confidence=0.55,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        verification_detector = vision.FaceLandmarker.create_from_options(options)

def verify_face_crop(face_crop):
    """
    Returns True if verification_detector finds a valid face inside face_crop,
    otherwise False.
    """
    ensure_verification_detector()
    if face_crop is None or face_crop.size == 0:
        return False
    try:
        import mediapipe as mp
        # BGR to RGB
        crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        result = verification_detector.detect(mp_image)
        if result.face_landmarks and len(result.face_landmarks) > 0:
            return True
    except Exception as e:
        print(f"[Verification Error] Failed to verify face crop: {e}")
    return False


@app.route("/api/video_detect", methods=["POST"])
def api_video_detect():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
    os.makedirs(temp_dir, exist_ok=True)

    filename = secure_filename(file.filename)
    filepath = os.path.join(temp_dir, filename)
    file.save(filepath)

    results = []

    try:
        ext = os.path.splitext(filename)[1].lower()
        is_video = ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]

        refresh_employee_cache()
        with _cache_lock:
            employees = list(_employee_cache)

        ensure_haar_cascade()

        # Relative matching helper function with lower threshold of 0.50 and 0.02 minimum margin
        def match_employee_video(embedding, threshold=0.50, min_margin=0.02):
            best_match = None
            best_score = -1.0
            second_best_score = -1.0
            
            for emp in employees:
                stored_emb = np.frombuffer(emp["embedding"], dtype=np.float32)
                if stored_emb.shape[0] != 512:
                    continue
                score = cosine_similarity(embedding, stored_emb)
                if score > best_score:
                    second_best_score = best_score
                    best_score = score
                    best_match = emp
                elif score > second_best_score:
                    second_best_score = score
            
            margin = best_score - second_best_score
            if best_match and best_score >= threshold and margin >= min_margin:
                return {
                    "employee_id": best_match["employee_id"],
                    "full_name": best_match["full_name"],
                    "department": best_match.get("department", "General"),
                    "confidence": round(best_score, 4),
                }
            return None

        if not is_video:
            # Process static image
            frame = cv2.imread(filepath)
            if frame is None:
                return jsonify({"error": "Could not read uploaded image."}), 400

            # Run Haar Cascade face detection with high sensitivity
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(30, 30))
            
            detected_faces = []
            for (x, y, w, h) in faces:
                # Align Haar Cascade box to match tight face landmarker crop (shift down 12%, scale down to 90%)
                w_new = int(w * 0.90)
                h_new = int(h * 0.90)
                x_new = x + int((w - w_new) / 2)
                y_new = y + int(h * 0.12)
                
                detected_faces.append({
                    "bounding_box": {
                        "originX": int(x_new),
                        "originY": int(y_new),
                        "width": int(w_new),
                        "height": int(h_new)
                    }
                })

            for idx, face_data in enumerate(detected_faces):
                bbox = face_data["bounding_box"]
                # Crop without padding for strict face verification
                face_crop_tight = crop_face(frame, bbox, padding=0.0)
                if face_crop_tight is None or not verify_face_crop(face_crop_tight):
                    continue

                # Crop with padding for embedding extraction
                face_crop = crop_face(frame, bbox, padding=0.2)
                if face_crop is None:
                    continue

                embedding = get_embedding(face_crop)
                match = None
                if embedding is not None:
                    match = match_employee_video(embedding)

                _, buffer = cv2.imencode('.jpg', face_crop)
                crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

                if match:
                    results.append({
                        "face_crop": crop_b64,
                        "is_stranger": False,
                        "employee_id": match["employee_id"],
                        "employee_name": match["full_name"],
                        "department": match["department"],
                        "confidence": match["confidence"],
                        "first_seen_seconds": 0.0
                    })
                else:
                    results.append({
                        "face_crop": crop_b64,
                        "is_stranger": True,
                        "employee_id": "—",
                        "employee_name": "Stranger",
                        "department": "—",
                        "confidence": 0.0,
                        "first_seen_seconds": 0.0,
                        "embedding": embedding.tolist() if embedding is not None else None
                    })
        else:
            # Process video
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                print(f"[Video Detection Error] Could not open video file: {filepath}")
                return jsonify({"error": "Could not open video file."}), 400

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            sample_interval = max(1, int(fps * 0.5))

            print(f"[Video Detection] Opened video successfully. File: {filename}, FPS: {fps}, sample_interval: {sample_interval}")

            all_detected = []
            frame_idx = 0
            read_count = 0
            
            # Auto-detection of video rotation and blank frames
            detected_rotation = None
            rotation_checked = False

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                read_count += 1

                if frame_idx % sample_interval == 0 and frame_idx > 0:
                    sec = round(frame_idx / fps, 1)
                    if frame is not None:
                        # Make sure frame is contiguous in memory
                        frame = np.ascontiguousarray(frame)

                        # Check for empty/black frames
                        max_pixel = np.max(frame)
                        if max_pixel == 0:
                            print(f"[Video Detection Warning] Frame {frame_idx} is completely black/empty!")

                        # Apply rotation if saved
                        if detected_rotation is not None:
                            frame = cv2.rotate(frame, detected_rotation)

                        # Run Haar Cascade with high sensitivity
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(30, 30))

                        # Auto-orient if no faces detected yet and rotation not checked
                        if len(faces) == 0 and not rotation_checked and max_pixel > 0:
                            # Try 90 degrees clockwise
                            f_90 = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                            g_90 = cv2.cvtColor(f_90, cv2.COLOR_BGR2GRAY)
                            faces_90 = face_cascade.detectMultiScale(g_90, scaleFactor=1.08, minNeighbors=4, minSize=(30, 30))
                            if len(faces_90) > 0:
                                detected_rotation = cv2.ROTATE_90_CLOCKWISE
                                frame = f_90
                                faces = faces_90
                                rotation_checked = True
                                print(f"[Video Detection] Detected rotation: 90 degrees clockwise")
                            else:
                                # Try 180 degrees
                                f_180 = cv2.rotate(frame, cv2.ROTATE_180)
                                g_180 = cv2.cvtColor(f_180, cv2.COLOR_BGR2GRAY)
                                faces_180 = face_cascade.detectMultiScale(g_180, scaleFactor=1.08, minNeighbors=4, minSize=(30, 30))
                                if len(faces_180) > 0:
                                    detected_rotation = cv2.ROTATE_180
                                    frame = f_180
                                    faces = faces_180
                                    rotation_checked = True
                                    print(f"[Video Detection] Detected rotation: 180 degrees")
                                else:
                                    # Try 90 degrees counter-clockwise (270 degrees)
                                    f_270 = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                                    g_270 = cv2.cvtColor(f_270, cv2.COLOR_BGR2GRAY)
                                    faces_270 = face_cascade.detectMultiScale(g_270, scaleFactor=1.08, minNeighbors=4, minSize=(30, 30))
                                    if len(faces_270) > 0:
                                        detected_rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
                                        frame = f_270
                                        faces = faces_270
                                        rotation_checked = True
                                        print(f"[Video Detection] Detected rotation: 90 degrees counter-clockwise")
                        elif len(faces) > 0 and not rotation_checked:
                            rotation_checked = True
                            print(f"[Video Detection] Confirmed upright orientation (no rotation needed)")

                        detected_faces = []
                        for (fx, fy, fw, fh) in faces:
                            # Align Haar Cascade box to match tight face landmarker crop (shift down 12%, scale down to 90%)
                            fw_new = int(fw * 0.90)
                            fh_new = int(fh * 0.90)
                            fx_new = fx + int((fw - fw_new) / 2)
                            fy_new = fy + int(fh * 0.12)

                            detected_faces.append({
                                "bounding_box": {
                                    "originX": int(fx_new),
                                    "originY": int(fy_new),
                                    "width": int(fw_new),
                                    "height": int(fh_new)
                                }
                            })

                        print(f"[Video Detection] Frame {frame_idx} (time: {sec}s, shape: {frame.shape}, max: {max_pixel}): detected {len(detected_faces)} faces")
                    else:
                        detected_faces = []
                        print(f"[Video Detection] Frame {frame_idx} (time: {sec}s): frame was None")

                    for face_data in detected_faces:
                        bbox = face_data["bounding_box"]
                        # Crop without padding for strict face verification
                        face_crop_tight = crop_face(frame, bbox, padding=0.0)
                        if face_crop_tight is None or not verify_face_crop(face_crop_tight):
                            continue

                        # Crop with padding for embedding extraction
                        face_crop = crop_face(frame, bbox, padding=0.2)
                        if face_crop is None:
                            continue

                        embedding = get_embedding(face_crop)
                        if embedding is None:
                            continue

                        _, buffer = cv2.imencode('.jpg', face_crop)
                        crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

                        all_detected.append({
                            "embedding": embedding,
                            "crop_b64": crop_b64,
                            "sec": sec,
                            "x_coord": bbox["originX"]
                        })

                frame_idx += 1

            cap.release()
            print(f"[Video Detection] Finished video loop. Total frames read: {read_count}, raw faces: {len(all_detected)}")

            # Group raw faces into clusters by cosine similarity (threshold >= 0.55)
            clusters = []
            for face in all_detected:
                best_cluster_idx = -1
                best_sim = -1.0
                for idx, cluster in enumerate(clusters):
                    max_cluster_sim = max(cosine_similarity(face["embedding"], item["embedding"]) for item in cluster)
                    if max_cluster_sim > best_sim:
                        best_sim = max_cluster_sim
                        best_cluster_idx = idx
                if best_sim >= 0.55:
                    clusters[best_cluster_idx].append(face)
                else:
                    clusters.append([face])

            print(f"[Video Detection] Grouped into {len(clusters)} unique face clusters.")

            # Group clusters into employee matches and stranger matches
            employee_clusters = []
            stranger_clusters = []

            for cluster in clusters:
                best_db_match = None
                best_db_score = -1.0
                best_cluster_face = cluster[0]
                earliest_sec = min(item["sec"] for item in cluster)

                for face in cluster:
                    # Compare against employees
                    for emp in employees:
                        stored_emb = np.frombuffer(emp["embedding"], dtype=np.float32)
                        if stored_emb.shape[0] != 512:
                            continue
                        score = cosine_similarity(face["embedding"], stored_emb)
                        if score > best_db_score:
                            best_db_score = score
                            best_db_match = emp
                            best_cluster_face = face

                match = None
                if best_db_match:
                    match = match_employee_video(best_cluster_face["embedding"])

                if match:
                    employee_clusters.append((match, best_cluster_face, earliest_sec))
                else:
                    stranger_clusters.append((best_cluster_face, earliest_sec))

            # Identify each cluster
            unique_employees = {}
            unique_strangers = []

            # 1. Process employee clusters first
            for match, best_cluster_face, earliest_sec in employee_clusters:
                emp_id = match["employee_id"]
                if emp_id in unique_employees:
                    if match["confidence"] > unique_employees[emp_id]["confidence"]:
                        unique_employees[emp_id]["crop_b64"] = best_cluster_face["crop_b64"]
                        unique_employees[emp_id]["confidence"] = match["confidence"]
                        unique_employees[emp_id]["embedding"] = best_cluster_face["embedding"]
                        unique_employees[emp_id]["x_coord"] = best_cluster_face["x_coord"]
                    if earliest_sec < unique_employees[emp_id]["first_seen"]:
                        unique_employees[emp_id]["first_seen"] = earliest_sec
                else:
                    unique_employees[emp_id] = {
                        "crop_b64": best_cluster_face["crop_b64"],
                        "employee_name": match["full_name"],
                        "department": match["department"],
                        "confidence": match["confidence"],
                        "first_seen": earliest_sec,
                        "embedding": best_cluster_face["embedding"],
                        "x_coord": best_cluster_face["x_coord"]
                    }

            # 2. Process stranger clusters second
            for best_cluster_face, earliest_sec in stranger_clusters:
                # Deduplicate strangers against recognized employees first (raised threshold to 0.52 to prevent different individuals from being merged)
                is_duplicate_employee = False
                for emp_id, emp_data in unique_employees.items():
                    # Spatial coordinate guard: if the spatial difference is > 400 pixels, they CANNOT be the same person!
                    if abs(best_cluster_face["x_coord"] - emp_data["x_coord"]) > 400:
                        continue
                    sim = cosine_similarity(best_cluster_face["embedding"], emp_data["embedding"])
                    if sim >= 0.52:
                        is_duplicate_employee = True
                        if earliest_sec < emp_data["first_seen"]:
                            emp_data["first_seen"] = earliest_sec
                        break

                if is_duplicate_employee:
                    continue

                # Deduplicate strangers (ensure no duplicate stranger cards, raised threshold to 0.52 to prevent merging different strangers)
                is_duplicate_stranger = False
                for st in unique_strangers:
                    # Spatial coordinate guard: if the spatial difference is > 400 pixels, they CANNOT be the same person!
                    if abs(best_cluster_face["x_coord"] - st["x_coord"]) > 400:
                        continue
                    sim = cosine_similarity(best_cluster_face["embedding"], st["embedding"])
                    if sim >= 0.52:
                        is_duplicate_stranger = True
                        if earliest_sec < st["first_seen"]:
                            st["first_seen"] = earliest_sec
                        break

                if not is_duplicate_stranger:
                    unique_strangers.append({
                        "crop_b64": best_cluster_face["crop_b64"],
                        "first_seen": earliest_sec,
                        "embedding": best_cluster_face["embedding"],
                        "x_coord": best_cluster_face["x_coord"]
                    })

            # Build results list
            for emp_id, emp_data in unique_employees.items():
                results.append({
                    "face_crop": emp_data["crop_b64"],
                    "is_stranger": False,
                    "employee_id": emp_id,
                    "employee_name": emp_data["employee_name"],
                    "department": emp_data["department"],
                    "confidence": emp_data["confidence"],
                    "first_seen_seconds": emp_data["first_seen"]
                })

            for st in unique_strangers:
                results.append({
                    "face_crop": st["crop_b64"],
                    "is_stranger": True,
                    "employee_id": "—",
                    "employee_name": "Stranger",
                    "department": "—",
                    "confidence": 0.0,
                    "first_seen_seconds": st["first_seen"],
                    "embedding": st["embedding"].tolist() if st.get("embedding") is not None else None
                })

    except Exception as e:
        print(f"[Error] Video detection failed: {e}")
        return jsonify({"error": f"Internal processing error: {str(e)}"}), 500
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass

    return jsonify({
        "success": True,
        "results": results
    })


# ──────────────────────────────────────────────────────────────
# Live IP Cameras — view feeds from camera_config.json in browser
# ──────────────────────────────────────────────────────────────
CAMERA_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_config.json")


def load_camera_list():
    """Read camera entries from camera_config.json."""
    try:
        with open(CAMERA_CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("cameras", [])
    except (OSError, json.JSONDecodeError) as e:
        print(f"[Live Cameras] Failed to load camera config: {e}")
        return []


def resolve_camera_source(url):
    """
    Resolve a camera config URL to an OpenCV source.

    Returns (source, is_file). Network URLs pass through unchanged;
    local video paths (demo/virtual cameras) resolve relative to the
    app directory so they work regardless of the process working dir.
    """
    if url.lower().startswith(("http://", "https://", "rtsp://", "rtmp://")):
        return url, False
    if url.isdigit():
        return int(url), False
    path = url if os.path.isabs(url) else os.path.join(os.path.dirname(os.path.abspath(__file__)), url)
    return path, True


def mjpeg_stream(url):
    """
    Relay frames from a camera source as an MJPEG multipart stream.
    Each browser viewer gets its own capture connection.
    Local video files loop forever (virtual demo camera) and are
    paced at their native FPS.
    """
    source, is_file = resolve_camera_source(url)
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame_delay = 0.05  # ~20 fps cap for network streams
    if is_file:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_delay = 1.0 / max(fps, 1)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    # Loop the video file like a continuous camera
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            # Downscale large frames to keep bandwidth reasonable
            h, w = frame.shape[:2]
            if w > 960:
                scale = 960 / w
                frame = cv2.resize(frame, (960, int(h * scale)))
            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
            time.sleep(frame_delay)
    finally:
        cap.release()


@app.route("/live_cameras", methods=["GET"])
def live_cameras_page():
    return render_template("live_cameras.html")


@app.route("/api/cameras", methods=["GET"])
def api_cameras():
    cameras = [
        {"id": c["id"], "name": c.get("name", c["id"]), "enabled": c.get("enabled", True)}
        for c in load_camera_list()
    ]
    return jsonify({"cameras": cameras})


@app.route("/camera_feed/<camera_id>", methods=["GET"])
def camera_feed(camera_id):
    camera = next((c for c in load_camera_list() if c["id"] == camera_id), None)
    if camera is None:
        return jsonify({"error": f"Camera not found: {camera_id}"}), 404
    return Response(
        mjpeg_stream(camera["url"]),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def _grab_camera_frames(url, count=1, skip=5, random_start=False):
    """
    Grab `count` frames from a camera source, skipping `skip` frames
    between samples. For local video files, optionally start at a
    random position so repeated calls see different moments.
    """
    import random

    source, is_file = resolve_camera_source(url)
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    frames = []
    try:
        if not cap.isOpened():
            return frames
        if is_file and random_start:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total > 30:
                cap.set(cv2.CAP_PROP_POS_FRAMES, random.randint(0, total - 20))
        while len(frames) < count:
            ret, frame = cap.read()
            if not ret:
                if is_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        break
                else:
                    break
            frames.append(frame)
            for _ in range(skip):
                if not cap.grab():
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    else:
                        break
    finally:
        cap.release()
    return frames


def _detect_face_crops(frame):
    """
    Detect faces in a frame and return their crops. If nothing is found
    at native resolution, retry on a 2x upscale (helps distant/small
    faces from CCTV-style footage).
    """
    for scale in (1.0, 2.0):
        work = frame if scale == 1.0 else cv2.resize(
            frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        with model_lock:
            try:
                detections = detector.detect(work)
            except Exception as e:
                print(f"[Camera Detect] Detection failed: {e}")
                detections = None
        crops = []
        for det in (detections or []):
            crop = crop_face(work, det["bounding_box"])
            if crop is not None:
                crops.append(crop)
        if crops:
            return crops
    return []


@app.route("/api/camera_detect/<camera_id>", methods=["GET"])
def api_camera_detect(camera_id):
    """
    Sample several frames from the camera, detect + recognize every face,
    and merge duplicates of the same person across frames (keeping the
    largest crop). Returns each unique person with a crop image,
    recognition result, and embedding (used to register a stranger).
    """
    camera = next((c for c in load_camera_list() if c["id"] == camera_id), None)
    if camera is None:
        return jsonify({"error": f"Camera not found: {camera_id}"}), 404

    fast_mode = request.args.get("fast") == "true"
    sample_count = 2 if fast_mode else 10
    sample_skip = 1 if fast_mode else 12
    frames = _grab_camera_frames(camera["url"], count=sample_count, skip=sample_skip, random_start=not fast_mode)
    if not frames:
        return jsonify({"error": "Could not read a frame from the camera."}), 503

    refresh_employee_cache()

    # Collect faces from ALL sampled frames, merging the same person
    # (embedding similarity) so each person appears once. Kept fairly
    # loose because the same face at different distances/angles can
    # drop to ~0.4 similarity; different people are typically < 0.3.
    SAME_PERSON = 0.40
    persons = []  # each: {"crop", "embedding", "area"}
    for frame in frames:
        for crop in _detect_face_crops(frame):
            embedding = get_embedding(crop)
            if embedding is None:
                continue
            area = crop.shape[0] * crop.shape[1]
            for p in persons:
                if cosine_similarity(embedding, p["embedding"]) >= SAME_PERSON:
                    if area > p["area"]:  # keep the clearest (largest) crop
                        p.update(crop=crop, embedding=embedding, area=area)
                    break
            else:
                persons.append({"crop": crop, "embedding": embedding, "area": area})

    faces = []
    for p in persons:
        with _cache_lock:
            match = compare_with_employees(p["embedding"], _employee_cache)
            possible = None

        ok, buffer = cv2.imencode(".jpg", p["crop"], [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue
        crop_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("utf-8")

        faces.append({
            "recognized": match is not None,
            "employee_id": match["employee_id"] if match else None,
            "full_name": match["full_name"] if match else None,
            "confidence": round(match["confidence"], 3) if match else None,
            "possible_match": possible,
            "crop_b64": crop_b64,
            "embedding": p["embedding"].tolist(),
        })

    return jsonify({"camera_id": camera_id, "faces": faces})


@app.route("/api/set_phone_camera", methods=["POST"])
def api_set_phone_camera():
    """
    Connect the user's mobile phone (IP Webcam app) as a live camera.

    Expects JSON with either:
      - ip: phone's WiFi IP (e.g. "192.168.1.5" or "192.168.1.5:8080")
      - url: full stream URL (overrides ip)

    Tests that the stream responds, then saves it as the 'cam-phone'
    entry in camera_config.json with enabled=true.
    """
    import urllib.request
    from urllib.parse import urlparse

    data = request.get_json() or {}
    url = (data.get("url") or "").strip()
    ip = (data.get("ip") or "").strip()

    if url:
        # Auto-append /video if user passed a bare host URL (e.g., http://192.168.18.92:8080)
        try:
            parsed = urlparse(url)
            if parsed.scheme in ("http", "https") and parsed.path in ("", "/"):
                if url.endswith("/"):
                    url = url + "video"
                else:
                    url = url + "/video"
        except Exception:
            pass

    if not url:
        if not ip:
            return jsonify({"error": "Phone IP required (shown in the IP Webcam app, e.g. 192.168.1.5)."}), 400
        if ":" not in ip:
            ip = f"{ip}:8080"
        url = f"http://{ip}/video"

    # Quick reachability test (IP Webcam responds with a multipart stream)
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=6) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if resp.status != 200:
                return jsonify({"error": f"Camera responded with HTTP {resp.status}."}), 502
            if "multipart" not in content_type and "video" not in content_type and "image" not in content_type:
                return jsonify({"error": f"URL responded but is not a video stream (Content-Type: {content_type})."}), 502
    except Exception as e:
        return jsonify({
            "error": f"Could not connect to the phone camera ({e.__class__.__name__}). "
                     "Check that: phone and PC are on the same WiFi, 'Start Server' is pressed in the IP Webcam app, and the IP is correct."
        }), 502

    # Save into camera_config.json
    try:
        with open(CAMERA_CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        cameras = config.setdefault("cameras", [])
        phone = next((c for c in cameras if c["id"] == "cam-phone"), None)
        if phone is None:
            phone = {"id": "cam-phone"}
            cameras.insert(0, phone)
        phone["name"] = "Mobile Camera (IP Webcam)"
        phone["url"] = url
        phone["enabled"] = True
        with open(CAMERA_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except (OSError, json.JSONDecodeError) as e:
        return jsonify({"error": f"Config save failed: {e}"}), 500

    return jsonify({
        "success": True,
        "message": "Mobile camera connected!",
        "camera_id": "cam-phone",
        "url": url,
    })


# ──────────────────────────────────────────────────────────────
# Employee Photo API
# ──────────────────────────────────────────────────────────────
@app.route("/api/employee_photo/<employee_id>")
def api_employee_photo(employee_id):
    """Serve the first face image from the employee's dataset folder."""
    from flask import send_file
    import io
    dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", employee_id)
    if os.path.isdir(dataset_dir):
        for fname in sorted(os.listdir(dataset_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                return send_file(os.path.join(dataset_dir, fname), mimetype='image/jpeg')
    # Fallback: 1x1 transparent PNG
    pixel = (b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
             b'\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
             b'\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01'
             b'\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82')
    return send_file(io.BytesIO(pixel), mimetype='image/png')


# ──────────────────────────────────────────────────────────────
# Logs Page & API
# ──────────────────────────────────────────────────────────────
@app.route("/logs", methods=["GET"])
def logs_page():
    return render_template("logs.html")


@app.route("/api/logs", methods=["GET"])
def api_logs():
    """Return unified system logs for a given date and optional event type."""
    date_str = request.args.get("date", "")
    event_type = request.args.get("type", "all")

    if not date_str:
        date_str = date.today().isoformat()

    try:
        target_date = date.fromisoformat(date_str)
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD."}), 400

    logs = get_all_logs(target_date, event_type if event_type != "all" else "all")
    return jsonify({"success": True, "logs": logs, "date": date_str, "total": len(logs)})


# ──────────────────────────────────────────────────────────────
# API — Video Register (register stranger from video detection)
# ──────────────────────────────────────────────────────────────
@app.route("/api/video_register", methods=["POST"])
def api_video_register():
    """
    Register a stranger detected in a video/image upload.
    Expects JSON with:
      - employee_id: str
      - full_name: str
      - department: str (optional)
      - embedding: list of floats (512-dim face embedding)
      - face_crop: base64 data URI of the face crop image
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request data."}), 400

    emp_id = data.get("employee_id", "").strip()
    full_name = data.get("full_name", "").strip()
    department = data.get("department", "").strip()
    embedding_list = data.get("embedding")
    face_crop_b64 = data.get("face_crop", "")

    if not emp_id:
        return jsonify({"error": "Employee ID is required."}), 400
    if not full_name:
        return jsonify({"error": "Full Name is required."}), 400
    if not embedding_list:
        return jsonify({"error": "Missing face embedding. Please detect faces first."}), 400
    if employee_exists(emp_id):
        return jsonify({"error": f"Employee ID '{emp_id}' is already registered."}), 409

    embedding = np.array(embedding_list, dtype=np.float32)
    if embedding.shape[0] != 512:
        return jsonify({"error": f"Invalid embedding dimension: {embedding.shape[0]} (expected 512)."}), 400

    # Decode face crop from base64 and save to dataset folder
    face_images = []
    if face_crop_b64:
        img = decode_base64_image(face_crop_b64)
        if img is not None:
            face_images.append(img)

    image_count = max(len(face_images), 1)
    if face_images:
        saved_paths = save_face_images(emp_id, face_images)
        print(f"[Video Register] Saved {len(saved_paths)} images to dataset/{emp_id}/")

    # Save using the provided embedding directly
    embedding_bytes = embedding.tobytes()
    success = save_employee(emp_id, full_name, department, embedding_bytes, image_count)

    if success:
        global _cache_last_refresh
        _cache_last_refresh = 0
        return jsonify({
            "success": True,
            "message": f"'{full_name}' registered successfully from video detection.",
        })
    return jsonify({"error": "Database error while saving employee."}), 500


@app.route("/api/camera_register", methods=["POST"])
def api_camera_register():
    """
    Register a stranger seen on a live camera as an employee.

    Expects JSON:
      - camera_id: which camera the person was seen on
      - employee_id, full_name, department: new employee details
      - target_embedding: embedding of the clicked face (from /api/camera_detect)

    Collects multiple face crops of that same person from the camera
    feed, then saves them exactly like normal registration.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request data."}), 400

    camera_id = data.get("camera_id", "")
    emp_id = data.get("employee_id", "").strip()
    full_name = data.get("full_name", "").strip()
    department = data.get("department", "").strip()
    target = data.get("target_embedding")

    if not emp_id:
        return jsonify({"error": "Employee ID is required."}), 400
    if not full_name:
        return jsonify({"error": "Full Name is required."}), 400
    if not target:
        return jsonify({"error": "Missing target face embedding. Scan faces first."}), 400
    if employee_exists(emp_id):
        return jsonify({"error": f"Employee ID '{emp_id}' is already registered."}), 409

    camera = next((c for c in load_camera_list() if c["id"] == camera_id), None)
    if camera is None:
        return jsonify({"error": f"Camera not found: {camera_id}"}), 404

    target_emb = np.array(target, dtype=np.float32)

    # Sample frames across the feed and keep crops of the SAME person
    # (embedding close to the clicked face).
    SAME_PERSON_THRESHOLD = 0.45
    frames = _grab_camera_frames(camera["url"], count=40, skip=3)
    face_images = []
    for frame in frames:
        if len(face_images) >= 15:
            break
        for crop in _detect_face_crops(frame):
            embedding = get_embedding(crop)
            if embedding is None:
                continue
            if cosine_similarity(embedding, target_emb) >= SAME_PERSON_THRESHOLD:
                face_images.append(crop)
                break  # one crop of this person per frame

    if len(face_images) < 5:
        return jsonify({
            "error": f"Could only capture {len(face_images)} clear face images of this person "
                     f"from the camera (need 5+). Try again when the face is clearly visible."
        }), 400

    print(f"[Camera Register] Collected {len(face_images)} face crops for '{full_name}' ({emp_id}) from {camera_id}.")

    saved_paths = save_face_images(emp_id, face_images)
    print(f"[Camera Register] Saved {len(saved_paths)} images to dataset/{emp_id}/")

    avg_embedding = compute_average_embedding(face_images)
    if avg_embedding is None:
        return jsonify({"error": "Failed to extract face embeddings from the captured images."}), 500

    success = save_employee(emp_id, full_name, department, avg_embedding.tobytes(), len(face_images))
    if success:
        global _cache_last_refresh
        _cache_last_refresh = 0
        return jsonify({
            "success": True,
            "message": f"'{full_name}' registered from camera with {len(face_images)} face images.",
        })
    return jsonify({"error": "Database error while saving employee."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True,
    )
