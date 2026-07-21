"""
Camera Manager — manages multiple camera threads with auto-reconnection.

Each enabled camera gets its own thread running a VideoProcessor.
Offline video files also get their own threads.
"""

import logging
import threading
import time

from surveillance.config import SurveillanceConfig, CameraConfig, OfflineFile
from surveillance.recognition_engine import RecognitionEngine
from surveillance.expression_engine import ExpressionEngine
from surveillance.duplicate_filter import DuplicateFilter
from surveillance.video_processor import VideoProcessor

logger = logging.getLogger("surveillance")


class CameraManager:
    """
    Orchestrates multiple camera threads, each running its own VideoProcessor.

    Shared resources (RecognitionEngine, ExpressionEngine, DuplicateFilter)
    are created once and injected into all VideoProcessors.

    Auto-reconnection: if a camera stream disconnects, the worker thread
    waits and retries automatically.
    """

    def __init__(self, config: SurveillanceConfig):
        """
        Args:
            config: Loaded surveillance configuration.
        """
        self.config = config
        self._threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._processors: dict[str, VideoProcessor] = {}
        self._running = False

        # Shared engines (created during start)
        self._recognition_engine: RecognitionEngine | None = None
        self._expression_engine: ExpressionEngine | None = None
        self._duplicate_filter: DuplicateFilter | None = None

    def start(self):
        """
        Initialize shared engines and start a thread per enabled camera
        and per offline video file.
        """
        if self._running:
            logger.warning("CameraManager already running.")
            return

        self._running = True

        # Create shared engines
        logger.info("Initializing shared recognition engine...")
        self._recognition_engine = RecognitionEngine(
            max_faces=self.config.processing.max_faces_per_frame,
            confidence_threshold=self.config.processing.confidence_threshold,
            wave_checkout_enabled=self.config.processing.wave_checkout_enabled,
        )
        self._expression_engine = ExpressionEngine()
        self._duplicate_filter = DuplicateFilter(
            attendance_cooldown=self.config.processing.attendance_cooldown_seconds,
            log_cooldown=self.config.processing.log_cooldown_seconds,
        )

        # Start live camera threads
        enabled = self.config.enabled_cameras
        if enabled:
            logger.info("Starting %d camera thread(s)...", len(enabled))
            for cam in enabled:
                self._start_camera_thread(cam)
        else:
            logger.warning("No cameras enabled in configuration.")

        # Start offline processing threads
        if self.config.offline.enabled and self.config.offline.video_files:
            logger.info(
                "Starting %d offline processing thread(s)...",
                len(self.config.offline.video_files),
            )
            for vf in self.config.offline.video_files:
                self._start_offline_thread(vf)

        total = len(self._threads)
        logger.info("CameraManager started — %d active thread(s).", total)

    def _start_camera_thread(self, cam: CameraConfig):
        """Start a worker thread for a single camera."""
        stop_event = threading.Event()
        self._stop_events[cam.id] = stop_event

        thread = threading.Thread(
            target=self._camera_worker,
            args=(cam, stop_event),
            name=f"cam-{cam.id}",
            daemon=True,
        )
        self._threads[cam.id] = thread
        thread.start()
        logger.info("Started thread for camera: %s (%s)", cam.name, cam.id)

    def _start_offline_thread(self, vf: OfflineFile):
        """Start a worker thread for an offline video file."""
        thread_id = f"offline-{vf.camera_id}"
        stop_event = threading.Event()
        self._stop_events[thread_id] = stop_event

        thread = threading.Thread(
            target=self._offline_worker,
            args=(vf, stop_event),
            name=thread_id,
            daemon=True,
        )
        self._threads[thread_id] = thread
        thread.start()
        logger.info("Started offline thread: %s (%s)", vf.camera_name, vf.path)

    def _camera_worker(self, cam: CameraConfig, stop_event: threading.Event):
        """
        Thread target for a live camera stream.

        Runs in a reconnection loop:
            1. Create VideoProcessor
            2. Process stream until disconnect or stop
            3. If disconnected, wait and retry
        """
        processor = VideoProcessor(
            camera_id=cam.id,
            camera_name=cam.name,
            recognition_engine=self._recognition_engine,
            expression_engine=self._expression_engine,
            duplicate_filter=self._duplicate_filter,
            config=self.config.processing,
        )
        self._processors[cam.id] = processor

        reconnect_delay = self.config.processing.reconnect_delay_seconds
        max_attempts = self.config.processing.max_reconnect_attempts
        attempt = 0

        while not stop_event.is_set():
            attempt += 1

            # Check max reconnect attempts (0 = unlimited)
            if max_attempts > 0 and attempt > max_attempts:
                logger.error(
                    "[%s] Max reconnect attempts (%d) reached. Stopping.",
                    cam.id, max_attempts,
                )
                break

            if attempt > 1:
                logger.info(
                    "[%s] Reconnection attempt %d (waiting %ds)...",
                    cam.id, attempt, reconnect_delay,
                )
                # Wait with stop check
                for _ in range(reconnect_delay):
                    if stop_event.is_set():
                        break
                    time.sleep(1)
                if stop_event.is_set():
                    break

            result = processor.process_stream(cam.url, stop_event)

            if result == "stopped":
                break
            elif result == "disconnected":
                logger.warning("[%s] Stream disconnected. Will retry...", cam.id)
                continue

        logger.info("[%s] Camera worker exited.", cam.id)

    def _offline_worker(self, vf: OfflineFile, stop_event: threading.Event):
        """
        Thread target for offline video file processing.
        Processes the file once and exits.
        """
        processor = VideoProcessor(
            camera_id=vf.camera_id,
            camera_name=vf.camera_name,
            recognition_engine=self._recognition_engine,
            expression_engine=self._expression_engine,
            duplicate_filter=self._duplicate_filter,
            config=self.config.processing,
        )
        thread_id = f"offline-{vf.camera_id}"
        self._processors[thread_id] = processor

        result = processor.process_file(
            vf.path,
            stop_event,
            frame_skip=self.config.offline.frame_skip,
        )

        logger.info(
            "[%s] Offline processing finished: %s — result: %s",
            vf.camera_id, vf.path, result,
        )

    def stop(self):
        """Signal all threads to stop and wait for them to finish."""
        if not self._running:
            return

        logger.info("Stopping CameraManager...")
        self._running = False

        # Signal all threads
        for stop_event in self._stop_events.values():
            stop_event.set()

        # Wait for all threads to finish (timeout per thread)
        for thread_id, thread in self._threads.items():
            logger.debug("Waiting for thread: %s", thread_id)
            thread.join(timeout=10)
            if thread.is_alive():
                logger.warning("Thread %s did not exit cleanly.", thread_id)

        # Clean up shared engines
        if self._recognition_engine:
            self._recognition_engine.close()
            self._recognition_engine = None

        self._threads.clear()
        self._stop_events.clear()
        self._processors.clear()
        logger.info("CameraManager stopped.")

    def status(self) -> dict:
        """
        Get status of all managed cameras/processors.

        Returns:
            dict keyed by thread_id, each with:
                - running: bool
                - frames_read: int
                - frames_processed: int
                - last_detection: float (timestamp) or None
        """
        result = {}
        for thread_id, thread in self._threads.items():
            proc = self._processors.get(thread_id)
            result[thread_id] = {
                "running": thread.is_alive(),
                "frames_read": proc.frames_read if proc else 0,
                "frames_processed": proc.frames_processed if proc else 0,
                "last_detection": proc.last_detection_time if proc else None,
            }
        return result
