"""
Multi-Face and Multi-Hand Tracker — associates detected faces and hands across
frames to maintain persistent session states for liveness tracking, face
recognition caching, and wave gesturing.
"""

import time
import numpy as np
from liveness import BlinkTracker
from gesture import WaveTracker

# Max distance in pixels (640x480 frame) to consider it the same face
MAX_TRACKING_DISTANCE_PIXELS = 150

# Track expiration in seconds
TRACK_EXPIRATION_SECONDS = 1.5


class FaceTrack:
    """Represents a persistent tracking session for a single person's face."""

    def __init__(self, track_id: int, bbox: dict):
        self.track_id = track_id
        self.bbox = bbox
        self.center = self._get_center(bbox)
        self.last_seen = time.time()

        # Custom blink and wave trackers for this specific face track
        self.blink_tracker = BlinkTracker()
        self.wave_tracker = WaveTracker()

        # Throttled face recognition cache for this track
        self.last_recognized_person = {
            "employee_id": None,
            "employee_name": "Unknown Person",
            "department": None,
            "recognition_confidence": 0.0,
            "is_recognized": False,
        }
        self.recognition_frame_counter = 0

    def update(self, bbox: dict):
        """Update bounding box, center, and timestamp when seen in a new frame."""
        self.bbox = bbox
        self.center = self._get_center(bbox)
        self.last_seen = time.time()

    @staticmethod
    def _get_center(bbox: dict) -> tuple[float, float]:
        """Compute the (x, y) center of the bounding box."""
        cx = bbox["originX"] + bbox["width"] / 2.0
        cy = bbox["originY"] + bbox["height"] / 2.0
        return (cx, cy)


class MultiFaceTracker:
    """Manages active face tracks, matches detections to tracks, and clean up expired tracks."""

    def __init__(self):
        self.active_tracks: dict[int, FaceTrack] = {}
        self._next_id = 1

    def update_tracks(self, detected_faces: list[dict]) -> list[FaceTrack]:
        """
        Match detected faces in the current frame to existing face tracks.
        Creates new tracks for unmatched faces and prunes stale tracks.

        Args:
            detected_faces: List of dicts returned by FaceDetector.detect

        Returns:
            List of FaceTrack objects corresponding to the detected faces in order.
        """
        now = time.time()

        # 1. Clean up stale tracks
        expired = [tid for tid, track in self.active_tracks.items()
                   if now - track.last_seen > TRACK_EXPIRATION_SECONDS]
        for tid in expired:
            del self.active_tracks[tid]
        # 2. Match detections to existing tracks, maintaining 1:1 order alignment with detected_faces
        matched_tracks = []

        # If no active tracks, all detections immediately create new tracks
        if not self.active_tracks:
            for face in detected_faces:
                track = self._create_new_track(face["bounding_box"])
                matched_tracks.append(track)
        else:
            # Prepare candidates
            track_ids = list(self.active_tracks.keys())
            tracks = [self.active_tracks[tid] for tid in track_ids]

            for face in detected_faces:
                bbox = face["bounding_box"]
                cx = bbox["originX"] + bbox["width"] / 2.0
                cy = bbox["originY"] + bbox["height"] / 2.0

                # Find closest track
                min_dist = float("inf")
                best_track = None

                for track in tracks:
                    dist = np.hypot(cx - track.center[0], cy - track.center[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_track = track

                # If closest track is within threshold, match it
                if best_track is not None and min_dist < MAX_TRACKING_DISTANCE_PIXELS:
                    best_track.update(bbox)
                    matched_tracks.append(best_track)
                    # Remove from candidates for subsequent faces in this frame
                    tracks.remove(best_track)
                else:
                    # Create a new track immediately to preserve the index order of detected_faces
                    new_track = self._create_new_track(bbox)
                    matched_tracks.append(new_track)

        return matched_tracks

    def _create_new_track(self, bbox: dict) -> FaceTrack:
        """Helper to instantiate and register a new track."""
        track = FaceTrack(self._next_id, bbox)
        self.active_tracks[self._next_id] = track
        self._next_id += 1
        return track

    def associate_hands(self, hand_landmarks_list: list[list] | None, w: int, h: int):
        """
        Associate each detected hand with the closest face track.
        Passes hand landmarks to the matched track's wave tracker, and None to unmatched.
        """
        if not self.active_tracks:
            return

        # Map each active track ID to its associated hand landmarks (or None)
        associations = {tid: None for tid in self.active_tracks.keys()}

        if hand_landmarks_list:
            for hand in hand_landmarks_list:
                # Wrist in pixel coords
                wx = hand[0][0] * w
                wy = hand[0][1] * h

                min_dist = float("inf")
                best_tid = None

                for tid, track in self.active_tracks.items():
                    dist = np.hypot(wx - track.center[0], wy - track.center[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_tid = tid

                # Associate hand with closest track
                if best_tid is not None:
                    associations[best_tid] = hand

        # Feed the (possibly None) landmarks to each track's wave tracker
        for tid, track in self.active_tracks.items():
            track.wave_detected = track.wave_tracker.update(associations[tid])
            track.hand_visible = associations[tid] is not None

    def reset(self):
        """Reset all tracking state."""
        self.active_tracks.clear()
        self._next_id = 1
