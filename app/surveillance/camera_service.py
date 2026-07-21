"""
Camera Service — top-level orchestrator for the background CCTV surveillance system.

This is the main entry point that initializes everything and runs the service.
It can be started independently of the Flask web server.
"""

import os
import sys
import signal
import logging
import threading

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import init_tables
from surveillance.config import load_config, SurveillanceConfig
from surveillance.detection_log import init_cctv_tables
from surveillance.camera_manager import CameraManager


class CameraService:
    """
    Top-level orchestrator for the background CCTV surveillance system.

    Usage:
        service = CameraService("camera_config.json")
        service.run_forever()  # Blocks until Ctrl+C
    """

    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: Path to camera_config.json.
                         Defaults to camera_config.json in the project root.
        """
        if config_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(project_root, "camera_config.json")

        self.config = load_config(config_path)
        self._camera_manager: CameraManager | None = None
        self._shutdown_event = threading.Event()

        # Set up logging
        self._setup_logging()

        self.logger = logging.getLogger("surveillance")
        self.logger.info("Configuration loaded: %s", self.config)

    def _setup_logging(self):
        """Configure Python logging with file + console handlers."""
        log_cfg = self.config.logging

        # Map string level to logging constant
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        level = level_map.get(log_cfg.level.upper(), logging.INFO)

        # Create logger
        logger = logging.getLogger("surveillance")
        logger.setLevel(level)

        # Clear existing handlers (in case of re-initialization)
        logger.handlers.clear()

        # Formatter
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-7s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # File handler
        if log_cfg.log_file:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_path = os.path.join(project_root, log_cfg.log_file)
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

        # Console handler
        if log_cfg.console_output:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            ch.setFormatter(fmt)
            logger.addHandler(ch)

    def start(self):
        """
        Initialize database tables, create CameraManager, and start
        all camera threads.
        """
        self.logger.info("=" * 60)
        self.logger.info("   Infigo FaceIntel — CCTV Surveillance Service")
        self.logger.info("=" * 60)

        # Initialize database tables (existing + CCTV-specific)
        self.logger.info("Initializing database tables...")
        init_tables()
        init_cctv_tables()

        # Create and start camera manager
        self._camera_manager = CameraManager(self.config)
        self._camera_manager.start()

        self.logger.info("Surveillance service is now running.")
        self.logger.info("Press Ctrl+C to stop.")

    def stop(self):
        """Gracefully shut down the service."""
        self.logger.info("Shutting down surveillance service...")
        self._shutdown_event.set()

        if self._camera_manager:
            self._camera_manager.stop()

        self.logger.info("Surveillance service stopped.")

    def run_forever(self):
        """
        Start the service and block until Ctrl+C or SIGTERM.

        This is the main entry point for running the service
        as a standalone process.
        """
        # Register signal handlers for graceful shutdown
        def _signal_handler(signum, frame):
            self.logger.info("Received signal %d — shutting down...", signum)
            self.stop()

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        try:
            self.start()

            # Block until shutdown event is set
            self._shutdown_event.wait()
        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received.")
            self.stop()
        except Exception as e:
            self.logger.error("Fatal error: %s", e, exc_info=True)
            self.stop()
            raise

    def status(self) -> dict:
        """Get the status of all managed cameras."""
        if self._camera_manager:
            return self._camera_manager.status()
        return {}


def main():
    """CLI entry point for the surveillance service."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Infigo FaceIntel — Background CCTV Surveillance Service",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to camera_config.json (default: ./camera_config.json)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Run only offline video processing (ignore live cameras)",
    )
    args = parser.parse_args()

    # If --offline flag is set, modify config to disable live cameras
    service = CameraService(config_path=args.config)

    if args.offline:
        # Disable all live cameras, enable offline
        for cam in service.config.cameras:
            cam.enabled = False
        service.config.offline.enabled = True
        service.logger.info("Offline mode: live cameras disabled.")

    service.run_forever()


if __name__ == "__main__":
    main()
