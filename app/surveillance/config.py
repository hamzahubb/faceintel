"""
Configuration Loader — reads and validates camera_config.json,
providing typed access to all surveillance settings.
"""

import json
import os


class CameraConfig:
    """Configuration for a single camera source."""

    def __init__(self, data: dict):
        self.id: str = data["id"]
        self.name: str = data.get("name", self.id)
        self.url: str = data["url"]
        self.enabled: bool = data.get("enabled", True)

    def __repr__(self):
        state = "ON" if self.enabled else "OFF"
        return f"CameraConfig({self.id!r}, {self.name!r}, [{state}])"


class OfflineFile:
    """Configuration for a single offline video file."""

    def __init__(self, data: dict):
        self.path: str = data["path"]
        self.camera_id: str = data.get("camera_id", "offline")
        self.camera_name: str = data.get("camera_name", "Offline Video")


class ProcessingConfig:
    """Processing parameters for frame analysis."""

    def __init__(self, data: dict):
        self.frame_skip: int = data.get("frame_skip", 5)
        self.confidence_threshold: float = data.get("confidence_threshold", 0.62)
        self.max_faces_per_frame: int = data.get("max_faces_per_frame", 4)
        self.reconnect_delay_seconds: int = data.get("reconnect_delay_seconds", 5)
        self.max_reconnect_attempts: int = data.get("max_reconnect_attempts", 0)
        self.employee_cache_refresh_seconds: int = data.get("employee_cache_refresh_seconds", 15)
        self.attendance_cooldown_seconds: int = data.get("attendance_cooldown_seconds", 300)
        self.log_cooldown_seconds: int = data.get("log_cooldown_seconds", 10)
        self.texture_liveness_enabled: bool = data.get("texture_liveness_enabled", True)
        self.wave_checkout_enabled: bool = data.get("wave_checkout_enabled", True)
        self.office_start_hour: int = data.get("office_start_hour", 9)
        self.office_start_minute: int = data.get("office_start_minute", 0)


class OfflineConfig:
    """Offline video processing configuration."""

    def __init__(self, data: dict):
        self.enabled: bool = data.get("enabled", False)
        self.frame_skip: int = data.get("frame_skip", 15)
        self.video_files: list[OfflineFile] = []
        for vf in data.get("video_files", []):
            self.video_files.append(OfflineFile(vf))


class LoggingConfig:
    """Logging configuration."""

    def __init__(self, data: dict):
        self.level: str = data.get("level", "INFO")
        self.log_file: str = data.get("log_file", "surveillance.log")
        self.console_output: bool = data.get("console_output", True)


class SurveillanceConfig:
    """
    Top-level surveillance configuration.
    Loads all settings from a JSON file and provides typed access.
    """

    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Create a camera_config.json file — see README for format."
            )

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Parse cameras
        self.cameras: list[CameraConfig] = []
        for cam_data in data.get("cameras", []):
            self.cameras.append(CameraConfig(cam_data))

        # Parse processing settings
        self.processing = ProcessingConfig(data.get("processing", {}))

        # Parse offline settings
        self.offline = OfflineConfig(data.get("offline", {}))

        # Parse logging settings
        self.logging = LoggingConfig(data.get("logging", {}))

    @property
    def enabled_cameras(self) -> list[CameraConfig]:
        """Return only cameras that are enabled."""
        return [c for c in self.cameras if c.enabled]

    def __repr__(self):
        return (
            f"SurveillanceConfig("
            f"{len(self.cameras)} cameras, "
            f"{len(self.enabled_cameras)} enabled, "
            f"offline={'ON' if self.offline.enabled else 'OFF'})"
        )


def load_config(config_path: str = None) -> SurveillanceConfig:
    """
    Load surveillance configuration from JSON file.

    Args:
        config_path: Path to camera_config.json.
                     Defaults to camera_config.json in the project root.

    Returns:
        SurveillanceConfig instance.
    """
    if config_path is None:
        # Default: look in same directory as app.py
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "camera_config.json")

    return SurveillanceConfig(config_path)
