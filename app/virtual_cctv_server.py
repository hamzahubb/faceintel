"""
Virtual CCTV Camera Server — simulates real IP cameras over HTTP.

Exposes the same camera API a real Axis/Panasonic IP camera provides,
so the surveillance system consumes a genuine CCTV Camera API for demos.
When a real office camera is installed, only the URL in
camera_config.json changes — nothing else in the system.

API endpoints (Axis-style):
    GET /                          -> JSON: server info + camera list
    GET /api/cameras               -> JSON: camera list
    GET /mjpg/<cam_id>/video.mjpg  -> live MJPEG stream (like a real IP cam)
    GET /jpg/<cam_id>/image.jpg    -> current snapshot (single JPEG)

Usage:
    python virtual_cctv_server.py            # serves on port 8081

Cameras are defined in VIRTUAL_CAMERAS below — each loops a video file
at its native FPS so every viewer sees the same "live" moment, exactly
like a real camera.
"""

import json
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 8081

VIRTUAL_CAMERAS = {
    "office": {
        "name": "Office CCTV (Full HD)",
        "file": os.path.join(BASE_DIR, "demo_videos", "office_cam.mp4"),
    },
    "lobby": {
        "name": "Entrance CCTV (people walking in)",
        "file": os.path.join(BASE_DIR, "demo_videos", "lobby_cam.mp4"),
    },
    "reception": {
        "name": "Reception CCTV (person close-up)",
        "file": os.path.join(BASE_DIR, "demo_videos", "reception_cam.mp4"),
    },
    "cabin": {
        "name": "Cabin CCTV (person at desk)",
        "file": os.path.join(BASE_DIR, "demo_videos", "cabin_cam.mp4"),
    },
}


class CameraWorker(threading.Thread):
    """
    Plays a video file in a loop at native FPS, holding the latest
    encoded JPEG frame in memory — one shared 'live' feed per camera,
    just like a real CCTV camera.
    """

    def __init__(self, cam_id: str, video_path: str):
        super().__init__(name=f"vcam-{cam_id}", daemon=True)
        self.cam_id = cam_id
        self.video_path = video_path
        self.latest_jpeg: bytes | None = None
        self.frame_event = threading.Event()
        self.fps = 12.0

    def _apply_cctv_overlay(self, frame):
        """Stamp a realistic CCTV OSD: camera id, timestamp, REC dot."""
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = f"CAM-{self.cam_id.upper()}"
        stamp = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

        # Top bar: camera name (left) + REC (right)
        cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(frame, label, (8, 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, (w - 70, 14), 6, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 58, 20), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom-left: timestamp
        cv2.rectangle(frame, (0, h - 26), (235, h), (0, 0, 0), -1)
        cv2.putText(frame, stamp, (8, h - 8), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def run(self):
        while True:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"[vcam-{self.cam_id}] Cannot open {self.video_path}, retrying in 5s...")
                time.sleep(5)
                continue
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
            delay = 1.0 / max(self.fps, 1)
            while True:
                start = time.time()
                ret, frame = cap.read()
                if not ret:
                    break  # end of file -> reopen (loop)
                frame = self._apply_cctv_overlay(frame)
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self.latest_jpeg = buf.tobytes()
                    self.frame_event.set()
                    self.frame_event.clear()
                elapsed = time.time() - start
                if elapsed < delay:
                    time.sleep(delay - elapsed)
            cap.release()


WORKERS: dict[str, CameraWorker] = {}


class CameraAPIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass  # keep console quiet

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _camera_list(self):
        return [
            {
                "id": cam_id,
                "name": cfg["name"],
                "stream_url": f"http://127.0.0.1:{PORT}/mjpg/{cam_id}/video.mjpg",
                "snapshot_url": f"http://127.0.0.1:{PORT}/jpg/{cam_id}/image.jpg",
                "online": WORKERS[cam_id].latest_jpeg is not None,
            }
            for cam_id, cfg in VIRTUAL_CAMERAS.items()
        ]

    def do_GET(self):
        parts = [p for p in self.path.split("?")[0].split("/") if p]

        # GET /  or  /api/cameras -> camera list API
        if not parts or parts == ["api", "cameras"]:
            self._send_json({
                "server": "Virtual CCTV Camera Server",
                "api_version": "1.0",
                "cameras": self._camera_list(),
            })
            return

        # GET /jpg/<cam_id>/image.jpg -> snapshot API
        if len(parts) == 3 and parts[0] == "jpg" and parts[2] == "image.jpg" and parts[1] in WORKERS:
            frame = WORKERS[parts[1]].latest_jpeg
            if frame is None:
                self._send_json({"error": "Camera warming up."}, 503)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)
            return

        # GET /mjpg/<cam_id>/video.mjpg -> live MJPEG stream API
        if len(parts) == 3 and parts[0] == "mjpg" and parts[2] == "video.mjpg" and parts[1] in WORKERS:
            worker = WORKERS[parts[1]]
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=vcamframe")
            self.end_headers()
            try:
                while True:
                    worker.frame_event.wait(timeout=2)
                    frame = worker.latest_jpeg
                    if frame is None:
                        continue
                    self.wfile.write(b"--vcamframe\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return  # viewer disconnected
            return

        self._send_json({"error": "Not found. See / for the camera API."}, 404)


def main():
    for cam_id, cfg in VIRTUAL_CAMERAS.items():
        if not os.path.exists(cfg["file"]):
            print(f"[WARN] Video missing for '{cam_id}': {cfg['file']}")
        worker = CameraWorker(cam_id, cfg["file"])
        WORKERS[cam_id] = worker
        worker.start()
        print(f"[vcam] Started camera '{cam_id}' -> http://127.0.0.1:{PORT}/mjpg/{cam_id}/video.mjpg")

    server = ThreadingHTTPServer(("0.0.0.0", PORT), CameraAPIHandler)
    print(f"[vcam] Virtual CCTV Camera Server running on http://127.0.0.1:{PORT}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[vcam] Stopped.")


if __name__ == "__main__":
    main()
