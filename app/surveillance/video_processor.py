"""
Video Processor — processes frames from a single video source (RTSP stream
or recorded file), performing face detection, recognition, expression analysis,
texture-based liveness, wave checkout, and attendance recording.

Each camera thread creates its own VideoProcessor instance.
"""

import os
import sys
import time
import logging
from datetime import datetime

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import (
    get_all_employees,
    check_in_employee,
    update_check_out,
    log_recognition,
    get_employee_attendance_today,
)
from tracker import MultiFaceTracker

from surveillance.recognition_engine import RecognitionEngine
from surveillance.expression_engine import ExpressionEngine
from surveillance.duplicate_filter import DuplicateFilter
from surveillance.detection_log import log_cctv_detection
from surveillance.config import ProcessingConfig

logger = logging.getLogger("surveillance")


class VideoProcessor:
    """
    Processes frames from a single video source (one camera or one file).

    Handles the complete pipeline:
        1. Read frames from the source
        2. Skip frames for performance
        3. Detect faces (via shared RecognitionEngine)
        4. Track faces across frames (per-camera MultiFaceTracker)
        5. Recognize employees (throttled via tracker's frame counter)
        6. Analyze expressions
        7. Perform texture-based liveness check
        8. Detect hand waves for checkout
        9. Record attendance and log detections

    Dependencies are injected for thread safety and resource sharing.
    """

    # How often to run full recognition (every N processed frames)
    RECOGNITION_INTERVAL = 3

    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        recognition_engine: RecognitionEngine,
        expression_engine: ExpressionEngine,
        duplicate_filter: DuplicateFilter,
        config: ProcessingConfig,
    ):
        """
        Args:
            camera_id: Unique identifier for this camera.
            camera_name: Human-readable camera name.
            recognition_engine: Shared face/hand detection + recognition engine.
            expression_engine: Shared expression classifier.
            duplicate_filter: Shared deduplication filter.
            config: Processing settings.
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.engine = recognition_engine
        self.expression = expression_engine
        self.dedup = duplicate_filter
        self.config = config

        # Per-camera face tracker (maintains face tracks between frames)
        self._tracker = MultiFaceTracker()

        # Employee cache (refreshed periodically)
        self._employee_cache: list[dict] = []
        self._cache_last_refresh: float = 0.0

        # Stats
        self.frames_read: int = 0
        self.frames_processed: int = 0
        self.last_detection_time: float = 0.0

    def _refresh_employee_cache(self):
        """Reload employee embeddings from DB if stale."""
        now = time.time()
        if now - self._cache_last_refresh < self.config.employee_cache_refresh_seconds:
            return  # Cache is still fresh

        employees = get_all_employees()
        if employees:
            self._employee_cache = employees
            self._cache_last_refresh = now
            logger.debug(
                "[%s] Refreshed employee cache: %d employees loaded.",
                self.camera_id,
                len(employees),
            )

    def _determine_attendance_status(self) -> str:
        """Returns 'present' or 'late' based on current time vs office hours."""
        now = datetime.now()
        start_hour = self.config.office_start_hour
        start_min = self.config.office_start_minute
        if now.hour > start_hour or (now.hour == start_hour and now.minute > start_min):
            return "late"
        return "present"

    def _process_single_frame(self, frame: np.ndarray):
        """
        Run the full detection/recognition/attendance pipeline on one frame.

        This is called every Nth frame (based on frame_skip setting).
        """
        self.frames_processed += 1

        # 1. Refresh employee cache if needed
        self._refresh_employee_cache()

        # 2. Detect faces
        detected_faces = self.engine.detect_faces(frame)
        if not detected_faces:
            return

        # 3. Update face tracker
        tracks = self._tracker.update_tracks(detected_faces)

        # 4. Detect hands (for wave checkout)
        if self.config.wave_checkout_enabled:
            hand_landmarks = self.engine.detect_hands(frame)
            h, w = frame.shape[:2]
            self._tracker.associate_hands(hand_landmarks, w, h)

        # 5. Process each detected face + track
        for face_data, track in zip(detected_faces, tracks):
            self._process_face(frame, face_data, track)

    def _process_face(self, frame: np.ndarray, face_data: dict, track):
        """Process a single detected face: recognize, classify, attend."""
        bbox = face_data["bounding_box"]
        blendshapes = face_data.get("blendshapes", {})

        # Crop the face for recognition
        face_crop = self.engine.crop_face(frame, bbox)

        # --- Throttled recognition ---
        track.recognition_frame_counter += 1
        run_recognition = (
            track.recognition_frame_counter >= self.RECOGNITION_INTERVAL
            or not track.last_recognized_person.get("is_recognized", False)
        )

        if run_recognition and face_crop is not None:
            track.recognition_frame_counter = 0
            match = self.engine.recognize(face_crop, self._employee_cache)

            if match:
                track.last_recognized_person = {
                    "employee_id": match["employee_id"],
                    "employee_name": match["full_name"],
                    "department": match.get("department"),
                    "recognition_confidence": match["confidence"],
                    "is_recognized": True,
                }

        # If person is not recognized, nothing more to do
        person = track.last_recognized_person
        if not person.get("is_recognized", False):
            return

        emp_id = person["employee_id"]
        emp_name = person["employee_name"]
        department = person.get("department")
        confidence = person.get("recognition_confidence", 0.0)

        self.last_detection_time = time.time()

        # --- Expression analysis ---
        expression_label = "neutral"
        expr_result = self.expression.analyze(blendshapes)
        if expr_result:
            expression_label = expr_result["dominant_label"]

        # --- Attendance: auto check-in ---
        if self.dedup.should_record_attendance(emp_id):
            # Check if already checked in today
            today_record = get_employee_attendance_today(emp_id)
            if today_record is None:
                # Not checked in — auto check-in
                status = self._determine_attendance_status()
                success = check_in_employee(emp_id, emp_name, department, status)
                if success:
                    self.dedup.mark_attendance_recorded(emp_id)
                    logger.info(
                        "📥 CHECK-IN [%s] %s (%s) — %s — confidence: %.2f",
                        self.camera_name, emp_name, emp_id, status, confidence,
                    )

        # --- Wave checkout ---
        if self.config.wave_checkout_enabled:
            wave_detected = getattr(track, "wave_detected", False)
            if wave_detected and self.dedup.should_record_checkout(emp_id):
                today_record = get_employee_attendance_today(emp_id)
                if today_record and today_record.get("check_in") and not today_record.get("check_out"):
                    success = update_check_out(emp_id)
                    if success:
                        self.dedup.mark_checkout_recorded(emp_id)
                        logger.info(
                            "📤 CHECK-OUT [%s] %s (%s) — wave detected — confidence: %.2f",
                            self.camera_name, emp_name, emp_id, confidence,
                        )

        # --- Recognition log ---
        if self.dedup.should_log_recognition(emp_id):
            log_recognition(emp_id, emp_name, expression_label, confidence)
            log_cctv_detection(
                emp_id, emp_name, self.camera_id, self.camera_name,
                expression_label, confidence,
            )
            self.dedup.mark_recognition_logged(emp_id)
            logger.debug(
                "📋 LOGGED [%s] %s — %s (%.2f)",
                self.camera_name, emp_name, expression_label, confidence,
            )

    def process_stream(self, url: str, stop_event) -> str:
        """
        Main processing loop for an RTSP/IP camera stream.

        Reads frames continuously, processes every Nth frame,
        and returns a status string when the stream disconnects
        or stop_event is set.

        Args:
            url: RTSP URL or camera index (e.g., 0 for webcam).
            stop_event: threading.Event — set to signal shutdown.

        Returns:
            "stopped" if stop_event was set, "disconnected" if stream broke.
        """
        logger.info("[%s] Connecting to: %s", self.camera_id, url)

        # Open video stream with RTSP transport hints
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering

        if not cap.isOpened():
            logger.error("[%s] Failed to open stream: %s", self.camera_id, url)
            return "disconnected"

        logger.info("[%s] Connected to %s", self.camera_id, self.camera_name)
        self.frames_read = 0
        self.frames_processed = 0
        self._tracker.reset()

        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("[%s] Stream read failed — disconnected.", self.camera_id)
                    return "disconnected"

                self.frames_read += 1

                # Skip frames for performance
                if self.frames_read % self.config.frame_skip != 0:
                    continue

                try:
                    self._process_single_frame(frame)
                except Exception as e:
                    logger.error(
                        "[%s] Error processing frame %d: %s",
                        self.camera_id, self.frames_read, e,
                    )
        finally:
            cap.release()
            logger.info(
                "[%s] Stream closed — %d frames read, %d processed.",
                self.camera_id, self.frames_read, self.frames_processed,
            )

        return "stopped"

    def process_file(self, file_path: str, stop_event, frame_skip: int = None) -> str:
        """
        Process a recorded video file (offline mode).

        Same pipeline as process_stream but reads from a file
        and processes to EOF.

        Args:
            file_path: Path to the video file (MP4, AVI, etc.).
            stop_event: threading.Event — set to signal early shutdown.
            frame_skip: Override frame_skip for offline processing.

        Returns:
            "completed" on EOF, "stopped" if stop_event was set.
        """
        if not os.path.exists(file_path):
            logger.error("[%s] Video file not found: %s", self.camera_id, file_path)
            return "error"

        skip = frame_skip or self.config.frame_skip
        logger.info("[%s] Processing offline file: %s (skip=%d)", self.camera_id, file_path, skip)

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            logger.error("[%s] Failed to open file: %s", self.camera_id, file_path)
            return "error"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        logger.info(
            "[%s] File info: %d frames, %.1f FPS, ~%.1f seconds",
            self.camera_id, total_frames, fps, total_frames / fps,
        )

        self.frames_read = 0
        self.frames_processed = 0
        self._tracker.reset()

        try:
            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.info("[%s] Reached end of file.", self.camera_id)
                    return "completed"

                self.frames_read += 1

                if self.frames_read % skip != 0:
                    continue

                try:
                    self._process_single_frame(frame)
                except Exception as e:
                    logger.error(
                        "[%s] Error processing frame %d: %s",
                        self.camera_id, self.frames_read, e,
                    )

                # Log progress every 500 processed frames
                if self.frames_processed % 500 == 0 and self.frames_processed > 0:
                    pct = (self.frames_read / total_frames * 100) if total_frames else 0
                    logger.info(
                        "[%s] Progress: %d/%d frames (%.1f%%)",
                        self.camera_id, self.frames_read, total_frames, pct,
                    )
        finally:
            cap.release()
            logger.info(
                "[%s] File processing complete — %d frames read, %d processed.",
                self.camera_id, self.frames_read, self.frames_processed,
            )

        return "stopped"
