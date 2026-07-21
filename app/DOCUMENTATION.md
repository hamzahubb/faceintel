# Infigo FaceIntel — Complete Project Documentation

**AI-Powered Face Recognition Attendance & CCTV Surveillance System**

Version 2.0 · Python + Flask + MediaPipe + ArcFace + MySQL

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Features](#3-features)
4. [Requirements & Installation](#4-requirements--installation)
5. [Quick Start](#5-quick-start)
6. [Service 1 — Virtual CCTV Camera Server](#6-service-1--virtual-cctv-camera-server)
7. [Service 2 — Web Dashboard (Flask)](#7-service-2--web-dashboard-flask)
8. [Service 3 — CCTV Surveillance Service](#8-service-3--cctv-surveillance-service)
9. [Camera Configuration](#9-camera-configuration)
10. [Connecting a Mobile Phone as a Camera](#10-connecting-a-mobile-phone-as-a-camera)
11. [Live Cameras — Scan & Register Workflow](#11-live-cameras--scan--register-workflow)
12. [Attendance Logic](#12-attendance-logic)
13. [Database Schema](#13-database-schema)
14. [REST API Reference](#14-rest-api-reference)
15. [Project Structure](#15-project-structure)
16. [Moving to Real CCTV Cameras](#16-moving-to-real-cctv-cameras)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. Project Overview

Infigo FaceIntel is a complete employee attendance and surveillance system based on face
recognition. It connects to IP/CCTV cameras over standard camera APIs (MJPEG-over-HTTP or
RTSP), detects and recognizes faces in the live feeds, and automatically records attendance
(check-in, late arrival, and wave-gesture checkout) into a MySQL database.

Everything runs **locally** — no cloud services. AI models used:

| Task | Model |
|---|---|
| Face detection + landmarks | Google MediaPipe Face Landmarker (468 points) |
| Face recognition (512-d embeddings) | ArcFace (ONNX Runtime) |
| Emotion / expression analysis | Rule-based classifier on MediaPipe blendshapes |
| Hand detection (wave checkout) | MediaPipe Hand Landmarker |
| Liveness (anti-spoofing) | Texture analysis |

For demos, a bundled **Virtual CCTV Camera Server** simulates real IP cameras (same API a
real Axis-style camera exposes), streaming looping HD footage of people so the full
pipeline can be shown without physical cameras.

---

## 2. System Architecture

The system runs as **three independent services** that share one MySQL database:

```
┌──────────────────────┐   MJPEG / RTSP    ┌──────────────────────┐
│ CAMERA SOURCES        │ ────────────────▶ │ 2. WEB DASHBOARD      │
│                       │                   │    (app.py :5000)     │
│ • Virtual CCTV Server │                   │  live view, scan,     │
│   (port 8081)         │                   │  register, attendance │
│ • Mobile phone        │                   └──────────┬───────────┘
│   (IP Webcam app)     │   MJPEG / RTSP               │
│ • PC webcam           │ ────────────────▶ ┌──────────▼───────────┐
│ • Real CCTV (RTSP)    │                   │ MySQL (XAMPP)         │
└──────────────────────┘                    │ database:             │
           ▲                                │ "facial detector"     │
           │ MJPEG / RTSP                   └──────────▲───────────┘
┌──────────┴───────────┐                               │
│ 3. SURVEILLANCE       │  attendance, logs            │
│    SERVICE            │ ──────────────────────────────┘
│ (run_surveillance.py) │
│ 24/7 background       │
│ monitoring            │
└──────────────────────┘
```

- **Virtual CCTV Camera Server** (`virtual_cctv_server.py`, port 8081) — simulates IP
  cameras for demos. Replaced by real cameras in production (config change only).
- **Web Dashboard** (`app.py`, port 5000) — browser UI: live camera view, face scan,
  one-click registration from camera, employee management, attendance reports, video
  upload analysis.
- **Surveillance Service** (`run_surveillance.py`) — headless background process; one
  thread per enabled camera with auto-reconnect; performs recognition, liveness, wave
  checkout, and writes attendance to the DB.

---

## 3. Features

- **Live IP camera feeds** in the browser (any MJPEG/RTSP source)
- **Face scan on live feed** — samples multiple frames, merges duplicate detections of
  the same person, and labels each unique person:
  - 🟢 **Registered** — matched employee (name + confidence)
  - 🟡 **Possibly \<name\>** — probably a registered person seen far away / blurry
    (prevents accidental duplicate registration)
  - 🔴 **Stranger** — unknown person, with a **Register** button
- **One-click registration from camera** — the system captures 5–15 face images of the
  selected person directly from the live feed and builds a robust average embedding
- **Mobile phone as camera** — connect from the UI by typing the phone's IP (IP Webcam app)
- **Automatic attendance** — check-in on first recognition, late detection after office
  start time, checkout by waving at the camera
- **Emotion detection** — 7 expressions from facial blendshapes
- **Liveness check** — texture-based anti-spoofing
- **Video/image upload analysis** — extract and identify every unique person in a file
- **Multi-camera** — unlimited cameras, one processing thread each, auto-reconnect
- **Virtual CCTV server** for demos, with realistic OSD overlay (camera ID, timestamp, REC)

---

## 4. Requirements & Installation

### Software

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10+ (tested on 3.14) | with pip |
| XAMPP (MySQL/MariaDB) | any recent | MySQL must be running |
| OS | Windows / Linux / macOS | project developed on Windows |

### Python packages

```
pip install -r requirements.txt
```

`requirements.txt` contains: `opencv-python`, `mediapipe`, `numpy`, `Flask`,
`onnxruntime`, `pymysql`.

### Database setup

1. Start **XAMPP → MySQL** (default: user `root`, empty password, host `localhost`).
2. The database **`facial detector`** and all tables are **created automatically** on
   first run. To create the DB manually:
   ```sql
   CREATE DATABASE IF NOT EXISTS `facial detector`;
   ```
3. Connection settings live at the top of `database.py` (`DB_CONFIG`).

### AI models

Model files (face landmarker, hand landmarker, ArcFace ONNX) **download automatically**
on first run and are cached in the `models/` folder.

---

## 5. Quick Start

### One click (recommended for demos)

Double-click **`start_demo.bat`** — it launches all three services in separate windows
and opens the Live Cameras page in your browser.

### Manual (three terminals)

```cmd
cd c:\Users\NLP\Desktop\app\app

:: Terminal 1 — virtual demo cameras (skip if using only real cameras)
python virtual_cctv_server.py

:: Terminal 2 — web dashboard
python app.py

:: Terminal 3 — background surveillance / attendance
python run_surveillance.py
```

Then open:

| URL | Page |
|---|---|
| http://127.0.0.1:5000/ | Real-time detection (browser webcam) |
| http://127.0.0.1:5000/live_cameras | **Live IP cameras + scan + register** |
| http://127.0.0.1:5000/register | Manual employee registration |
| http://127.0.0.1:5000/attendance | Attendance dashboard |
| http://127.0.0.1:5000/video_detection | Video/image upload analysis |
| http://127.0.0.1:8081/ | Virtual CCTV camera API (JSON) |

---

## 6. Service 1 — Virtual CCTV Camera Server

`virtual_cctv_server.py` simulates real IP cameras so the whole system can be
demonstrated without hardware. Each virtual camera loops an HD video file at native FPS
(all viewers see the same "live" moment) and stamps a realistic CCTV overlay: camera ID,
date/time, and a red REC dot.

### Camera API endpoints (Axis-style, port 8081)

| Endpoint | Returns |
|---|---|
| `GET /` or `GET /api/cameras` | JSON list of cameras with stream/snapshot URLs |
| `GET /mjpg/<cam_id>/video.mjpg` | Live MJPEG stream (`multipart/x-mixed-replace`) |
| `GET /jpg/<cam_id>/image.jpg` | Current snapshot (single JPEG) |

### Bundled demo cameras

| ID | Name | Footage |
|---|---|---|
| `office` | Office CCTV (Full HD) | Man walks toward the camera, face becomes large and clear |
| `lobby` | Entrance CCTV | People walking in through a hallway |
| `reception` | Reception CCTV | Person close-up, face visible in every frame |
| `cabin` | Cabin CCTV | Person at a desk facing the camera |

Demo videos live in `demo_videos/` (sourced from Intel's open sample-videos and Pexels
free stock, trimmed to face-dense segments). To add a virtual camera, drop a video in
`demo_videos/`, add an entry to `VIRTUAL_CAMERAS` in `virtual_cctv_server.py`, and add a
matching entry in `camera_config.json`.

---

## 7. Service 2 — Web Dashboard (Flask)

`app.py` serves the browser UI on port 5000.

### Pages

- **Detection (`/`)** — real-time emotion + recognition using the PC's browser webcam
- **Register (`/register`)** — manual employee registration (guided face capture)
- **Attendance (`/attendance`)** — daily attendance table, summary, and logs
- **Video Detection (`/video_detection`)** — upload a video/image; extracts every unique
  person, matches against employees
- **Live Cameras (`/live_cameras`)** — all configured IP cameras: live view, mobile
  camera connect panel, face scan, and register-from-camera

### How the live view works

The dashboard relays each camera's stream as browser-friendly MJPEG via
`/camera_feed/<camera_id>` (down-scaled to ≤960 px width for bandwidth). Local video
file sources loop forever and are paced at native FPS.

---

## 8. Service 3 — CCTV Surveillance Service

`run_surveillance.py` is the 24/7 background worker used for automatic attendance.

```
python run_surveillance.py                  # default camera_config.json
python run_surveillance.py --config x.json  # custom config
python run_surveillance.py --offline        # process recorded files only
```

Per enabled camera it runs a thread that:

1. Connects to the stream (`cv2.VideoCapture`, FFMPEG backend)
2. Reads frames, processing every Nth frame (`frame_skip`)
3. Detects and tracks faces across frames (per-camera `MultiFaceTracker`)
4. Recognizes employees (ArcFace embedding vs DB, threshold `confidence_threshold`)
5. Runs expression analysis and texture liveness
6. Detects hand-wave gesture → records checkout
7. Writes attendance + detection logs (with cooldowns to avoid duplicates)
8. **Auto-reconnects** if the stream drops (`reconnect_delay_seconds`)

Logs go to console and `surveillance.log`.

> **Note:** the surveillance service reads `camera_config.json` at startup — restart it
> after adding/enabling a camera. The dashboard reads the config per request, so the
> Live Cameras page picks up changes immediately.

---

## 9. Camera Configuration

All camera sources are defined in **`camera_config.json`**:

```json
{
  "cameras": [
    {
      "id": "cam-demo-office",                              // unique id
      "name": "Office CCTV (Camera API)",                   // display name
      "url": "http://127.0.0.1:8081/mjpg/office/video.mjpg",// stream URL
      "enabled": true                                        // thread starts if true
    }
  ],
  "processing": {
    "frame_skip": 5,                  // process every 5th frame
    "confidence_threshold": 0.62,     // min cosine similarity for a match
    "max_faces_per_frame": 4,
    "reconnect_delay_seconds": 5,
    "max_reconnect_attempts": 0,      // 0 = retry forever
    "employee_cache_refresh_seconds": 15,
    "attendance_cooldown_seconds": 300,
    "log_cooldown_seconds": 10,
    "texture_liveness_enabled": true,
    "wave_checkout_enabled": true,
    "office_start_hour": 9,           // arrivals after 09:00 are "late"
    "office_start_minute": 0
  },
  "offline": { "enabled": false, "video_files": [], "frame_skip": 15 },
  "logging": { "level": "INFO", "log_file": "surveillance.log", "console_output": true }
}
```

### Supported URL formats

| Type | Example |
|---|---|
| MJPEG over HTTP (IP cams, IP Webcam app, virtual server) | `http://192.168.18.37:8080/video` |
| RTSP (real CCTV/NVR) | `rtsp://admin:pass@192.168.1.100:554/stream1` |
| Local webcam index | `"0"` |
| Local video file (looped like a live camera) | `demo_videos/office_cam.mp4` |

---

## 10. Connecting a Mobile Phone as a Camera

The phone becomes a real Full-HD IP camera.

1. Install **IP Webcam** (Android, by Pavel Khlebovich) from the Play Store
2. Make sure phone and PC are on the **same WiFi**
3. In the app: *Video preferences → Video resolution* → `1920x1080`
4. Tap **Start Server** — the app shows an address like `http://192.168.18.37:8080`
5. Open **http://127.0.0.1:5000/live_cameras** → in the 📱 *Connect Your Mobile Camera*
   panel type `192.168.18.37:8080` → **Connect**

The system validates the stream, saves it to `camera_config.json` (`cam-phone` entry),
and the phone feed appears as a camera card. Restart the surveillance service if you
also want automatic attendance from the phone camera.

*(iPhone: use "IP Camera Lite" — enter its full stream URL in the same panel.)*

---

## 11. Live Cameras — Scan & Register Workflow

1. Open **Live Cameras** → each enabled camera shows its live feed
2. Click **🔍 Scan Faces** — the server samples ~10 frames, detects every face, computes
   ArcFace embeddings, merges duplicates of the same person, and compares against the
   employee database
3. Results appear as chips under the camera:
   - 🟢 `✔ Name · EMP001 · 89%` — recognized employee
   - 🟡 `Possibly Name` — likely a registered person, seen unclear/far (no register
     button, prevents duplicates)
   - 🔴 `Stranger` + **➕ Register** button
4. Click **Register** on a stranger → enter Employee ID, Full Name, Department →
   the system watches the feed for a few seconds, captures 5–15 clear face images of
   *that same person* (matched by embedding), saves them to `dataset/<EMP_ID>/`, stores
   the average embedding in the DB
5. Scan again — the person now shows as **Registered**, and the surveillance service
   starts marking their attendance automatically

---

## 12. Attendance Logic

| Event | Rule |
|---|---|
| **Check-in** | First confident recognition of the day (`confidence ≥ 0.62`) |
| **Present / Late** | Compared to `office_start_hour:office_start_minute` (default 09:00) |
| **Checkout** | Employee waves a hand at the camera (`wave_checkout_enabled`) |
| **Duplicate protection** | `attendance_cooldown_seconds` (default 5 min) per person |
| **Log throttling** | `log_cooldown_seconds` (default 10 s) per person per camera |
| **Anti-spoofing** | Texture liveness check (`texture_liveness_enabled`) |

---

## 13. Database Schema

Database: **`facial detector`** (MySQL/MariaDB via XAMPP). All tables auto-created.

| Table | Purpose |
|---|---|
| `employees` | Employee ID, name, department, 512-d face embedding (BLOB), image count |
| `attendance` | Daily check-in/check-out times and status (present/late) |
| `recognition_logs` | Every recognition event (person, confidence, emotion, timestamp) |
| `cctv_detection_log` | Per-camera CCTV detection events from the surveillance service |

Face images are stored on disk in `dataset/<EMPLOYEE_ID>/img_XXX.jpg`.

---

## 14. REST API Reference

### Web Dashboard (port 5000)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/cameras` | List configured cameras (id, name, enabled) |
| GET | `/camera_feed/<camera_id>` | Relay camera stream as MJPEG |
| GET | `/api/camera_detect/<camera_id>` | Scan feed → faces with recognition results + embeddings |
| POST | `/api/camera_register` | Register a scanned stranger from the live feed |
| POST | `/api/set_phone_camera` | Validate & save a mobile phone camera (`{"ip": "..."} `or `{"url": "..."}`) |
| POST | `/api/register_employee` | Manual registration (base64 face images) |
| GET | `/api/employees` | List registered employees |
| POST | `/api/delete_employee` | Remove an employee |
| GET | `/api/attendance?date=YYYY-MM-DD` | Attendance records for a date |
| POST | `/process_frame` | Real-time browser-webcam pipeline (detection page) |
| POST | `/api/video_detect` | Analyze an uploaded video/image |

`POST /api/camera_register` body:

```json
{
  "camera_id": "cam-demo-office",
  "employee_id": "EMP005",
  "full_name": "Ali Khan",
  "department": "Engineering",
  "target_embedding": [ /* 512 floats from /api/camera_detect */ ]
}
```

### Virtual CCTV Camera Server (port 8081)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` , `/api/cameras` | Camera list (JSON) |
| GET | `/mjpg/<cam_id>/video.mjpg` | Live MJPEG stream |
| GET | `/jpg/<cam_id>/image.jpg` | Snapshot JPEG |

---

## 15. Project Structure

```
app/
├── app.py                     # Web dashboard (Flask, port 5000)
├── run_surveillance.py        # Surveillance service entry point
├── virtual_cctv_server.py     # Virtual CCTV camera API server (port 8081)
├── start_demo.bat             # One-click demo startup
├── camera_config.json         # All camera sources + processing settings
├── database.py                # MySQL connection, tables, attendance queries
├── detector.py                # MediaPipe face & hand landmarker wrappers
├── recognizer.py              # ArcFace embeddings, matching, registration helpers
├── classifier.py              # Emotion classification from blendshapes
├── liveness.py                # Anti-spoofing checks
├── gesture.py                 # Wave-gesture (checkout) tracker
├── tracker.py                 # Multi-face tracking across frames
├── requirements.txt
├── DOCUMENTATION.md           # This file
├── surveillance/              # Background surveillance package
│   ├── camera_service.py      #   service entry (config load, lifecycle)
│   ├── camera_manager.py      #   thread-per-camera orchestration + reconnect
│   ├── video_processor.py     #   per-camera frame pipeline
│   ├── recognition_engine.py  #   shared detection/recognition engine
│   ├── expression_engine.py   #   shared emotion engine
│   ├── duplicate_filter.py    #   attendance/log cooldowns
│   ├── detection_log.py       #   cctv_detection_log writes
│   └── config.py              #   typed config loader
├── templates/                 # Web UI (Flask/Jinja)
│   ├── index.html             #   real-time detection page
│   ├── register.html          #   manual registration
│   ├── attendance.html        #   attendance dashboard
│   ├── video_detection.html   #   upload analysis
│   └── live_cameras.html      #   live cameras + scan + register + phone connect
├── demo_videos/               # Looping footage for virtual cameras
├── dataset/                   # Saved face images per employee
└── models/                    # Auto-downloaded AI models
```

---

## 16. Moving to Real CCTV Cameras

Nothing in the code changes — only `camera_config.json`:

1. Get the camera's RTSP URL (from its manual or NVR), e.g.
   `rtsp://admin:password@192.168.1.100:554/stream1`
2. Add/enable an entry:
   ```json
   { "id": "cam-entrance", "name": "Main Entrance", "url": "rtsp://admin:password@192.168.1.100:554/stream1", "enabled": true }
   ```
3. Restart the surveillance service. Done — live view, scan, register, and attendance
   all work identically.

**Camera placement tip:** for reliable face recognition, mount the camera at face
height (or slightly above) where people walk toward it — e.g. at the entrance door —
within 1–4 meters. High ceiling-mounted cameras see faces too small/angled to recognize.

---

## 17. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| `[DB Error] Unknown database` | Start XAMPP MySQL; DB is auto-created on next run |
| Camera card shows "Stream unavailable" | Check the URL in `camera_config.json`; is the virtual server / phone app running? |
| Phone connect fails | Phone & PC on same WiFi; "Start Server" pressed; IP typed exactly as shown in app |
| Scan finds no faces | Person must be near/facing the camera; scan again (frames are sampled) |
| Same person shows as Stranger far away | Expected — shows as 🟡 "Possibly \<name\>" when unsure; recognition confirms when they come closer |
| Attendance not marked from a new camera | Restart `run_surveillance.py` (config is read at startup) |
| Port 5000/8081 already in use | Stop the old process or change the port in `app.py` / `virtual_cctv_server.py` |
| Console shows `UnicodeEncodeError` in logs | Cosmetic (Windows console encoding); log file is unaffected |

---

*Generated 16 July 2026 — Infigo FaceIntel v2.0*
