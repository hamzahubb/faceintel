# 🧠 Infigo FaceIntel

**Enterprise AI-Powered Face Recognition, Attendance Management & Real-Time CCTV Surveillance System**

Infigo FaceIntel is a high-performance, 100% local computer vision web application for automated attendance tracking, multi-camera live CCTV surveillance, emotion analysis, and stranger registration. Built with Flask, OpenCV, MediaPipe, and ONNX Runtime ArcFace embeddings.

---

## 🚀 Key Capabilities & Recent Updates

### 🎥 1. Real-Time Live Camera Auto-Scanning
* **Automatic Background Detection**: Cameras automatically detect and identify faces as soon as a person steps into the feed without requiring manual scan clicks.
* **Low-Latency Fast Mode**: `/api/camera_detect/<id>?fast=true` samples frames with minimal skip intervals for instantaneous detection.
* **Visual Auto-Scan Pulse**: Displays a green pulsing indicator (`● Auto Scanning`) next to each camera stream.
* **Auto-Scan UI Toggle**: Toggle between continuous auto-scanning and manual scan mode per camera.
* **Modal Safety Lock**: Auto-scan updates pause automatically while registering a stranger to prevent UI form resets.

### 📹 2. Video Detection & Stranger Registration
* **Multi-Face Video Upload Analysis**: Upload video or image files to extract and cluster unique individuals using cosine similarity.
* **Stranger Registration**: One-click **➕ Register** modal on video detection stranger cards, generating 512-dim ArcFace embeddings directly.

### 📊 3. Unified Activity Logs Dashboard
* **Comprehensive Audit Trail**: Access `/logs` to view all system events chronologically.
* **Event Type Badging**: Color-coded badges for **Registration**, **Check-in**, **Check-out**, **Recognition**, and **CCTV Detection**.
* **Date & Event Filters**: Filter logs dynamically by specific dates and event categories.

### 🖼️ 4. Dedicated Employee Photo Column
* **Dataset Photo Serving**: Serves registered employee face crops via `/api/employee_photo/<employee_id>`.
* **Gradient Fallback Avatars**: Renders stylized initial-based avatar circles when a photo is missing or unlinked.

### ⚡ 5. Performance Optimizations
* **Batched Canvas Rendering (10x Faster)**: Batches 468 MediaPipe landmark points into a single HTML5 Canvas path, eliminating frame lag and stuttering.
* **In-Memory Attendance Caching**: Caches daily attendance status with a 3-second TTL to eliminate SQL query overhead on every frame.
* **Conditional Hand Landmark Detection**: Runs MediaPipe Hand Detection only when a recognized user requiring wave-to-checkout is on screen.
* **Failproof Database Auto-Connection**: Auto-detects ports `3306` & `3307` with passwords `""` & `"root"`, automatically creating the database schema if missing.

---

## 🛠️ Architecture & Port Map

| Component / Service | Script / Route | Default Port | Description |
|:---|:---|:---:|:---|
| **Web Dashboard** | `app.py` | `5000` | Flask application serving web pages and API endpoints. |
| **Virtual CCTV Server** | `virtual_cctv_server.py` | `8081` | Serves simulated MJPEG CCTV video streams for testing. |
| **Surveillance AI Engine** | `run_surveillance.py` | Background | Continuously monitors feeds and records attendance into MySQL. |
| **Database** | MySQL / MariaDB | `3306` / `3307` | Stores `employees`, `attendance`, `recognition_logs`, and `cctv_detection_log`. |

---

## 💻 Quick Start & One-Click Launchers

### Option A: One-Click Desktop Launcher (Recommended)
Double-click **`Infigo_FaceIntel.bat`** on your Desktop. It automatically:
1. Starts the Virtual CCTV Camera Server.
2. Starts the Surveillance AI Engine.
3. Starts the Web Application.
4. Opens **http://127.0.0.1:5000** in your browser.

### Option B: Batch File Launcher
Inside the `app` folder, double-click:
```cmd
run_app.bat
```

### Option C: Manual Command Prompt (CMD) Execution
```cmd
cd /d "c:\Users\hp\Desktop\appL\app"
python app.py
```

---

## 📡 API Reference Overview

| Endpoint | Method | Purpose |
|:---|:---:|:---|
| `/` | `GET` | Main webcam detection dashboard. |
| `/register` | `GET` | Employee webcam registration page. |
| `/attendance` | `GET` | Daily attendance statistics & logs. |
| `/video_detection` | `GET` | Video file upload analysis page. |
| `/live_cameras` | `GET` | Live CCTV cameras auto-scanning dashboard. |
| `/logs` | `GET` | Unified system activity logs page. |
| `/process_frame` | `POST` | Processes real-time webcam frames for tracking, emotions, and wave checkout. |
| `/api/camera_detect/<id>`| `GET` | Detects/recognizes faces from a camera feed (`?fast=true` for auto-scan). |
| `/api/camera_register` | `POST` | Registers a stranger from a live camera feed. |
| `/api/video_detect` | `POST` | Uploads and clusters faces from a video file. |
| `/api/video_register` | `POST` | Registers a stranger from video detection results. |
| `/api/employee_photo/<id>`| `GET` | Serves employee dataset face crop image. |
| `/api/logs` | `GET` | Retrieves system logs filtered by date and event type. |
| `/api/set_phone_camera` | `POST` | Connects an Android IP Webcam feed to the system. |

---

## ⚙️ Requirements

* **Python**: 3.10+ (`opencv-python`, `mediapipe`, `Flask`, `onnxruntime`, `numpy`, `pymysql`, `pillow`, `werkzeug`)
* **Database**: XAMPP (MariaDB / MySQL) local server
* **Models**: MediaPipe FaceLandmarker, HandLandmarker & ArcFace ONNX (Downloaded automatically on first launch)

---

## 🔒 Privacy & Local Processing
All AI models (MediaPipe + ArcFace ONNX) run 100% locally on your machine. No facial data or video streams are transmitted to external cloud servers.
