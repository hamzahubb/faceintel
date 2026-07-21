"""
Face Recognizer — uses ONNX Runtime with an ArcFace MobileFaceNet model (w600k_mbf)
to extract 512-d face embeddings and compare them via cosine similarity.

No TensorFlow required — runs entirely on ONNX Runtime (CPU).
Model auto-downloads on first run (~16 MB) and is cached locally.
"""

import os
import urllib.request
import numpy as np
import cv2
from numpy.linalg import norm

# ──────────────────────────────────────────────────────────────
# Model Configuration
# ──────────────────────────────────────────────────────────────
# ArcFace MobileFaceNet trained on WebFace600K — 512-d embeddings
MODEL_URL = (
    "https://huggingface.co/deepghs/insightface/resolve/main/"
    "buffalo_s/w600k_mbf.onnx"
)
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
RECOGNITION_MODEL_PATH = os.path.join(MODEL_DIR, "w600k_mbf.onnx")

# Recognition threshold: cosine similarity above this = match
MATCH_THRESHOLD = 0.60

# ArcFace input size
INPUT_SIZE = (112, 112)

# Embedding dimension
EMBEDDING_DIM = 512

# Dataset folder for storing captured face images
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# ──────────────────────────────────────────────────────────────
# ONNX Session (lazy-loaded)
# ──────────────────────────────────────────────────────────────
_session = None
_input_name = None


def _download_recognition_model(progress_callback=None):
    """Download the ArcFace ONNX model if not cached locally."""
    if os.path.exists(RECOGNITION_MODEL_PATH):
        if progress_callback:
            progress_callback("Recognition model already cached locally.")
        return

    os.makedirs(MODEL_DIR, exist_ok=True)

    if progress_callback:
        progress_callback("Downloading ArcFace recognition model (~16 MB)...")

    try:
        urllib.request.urlretrieve(MODEL_URL, RECOGNITION_MODEL_PATH)
        if progress_callback:
            progress_callback("Recognition model download complete!")
    except Exception as e:
        if progress_callback:
            progress_callback(f"Download failed: {e}")
        raise


def _ensure_model_loaded():
    """Lazily load the ONNX model on first call."""
    global _session, _input_name

    if _session is not None:
        return

    import onnxruntime as ort

    print("[Recognition] Loading ArcFace MobileFaceNet ONNX model...")
    _download_recognition_model(progress_callback=print)

    _session = ort.InferenceSession(
        RECOGNITION_MODEL_PATH,
        providers=["CPUExecutionProvider"],
    )
    _input_name = _session.get_inputs()[0].name

    # Warm up with a dummy inference
    dummy = np.zeros((1, 3, 112, 112), dtype=np.float32)
    _session.run(None, {_input_name: dummy})

    print("[Recognition] ArcFace model ready!")


def _preprocess_face(face_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocess a BGR face crop for ArcFace inference.
    Steps: resize to 112x112, BGR→RGB, normalize to [-1, 1], transpose to NCHW.
    """
    # Resize to 112x112
    face = cv2.resize(face_bgr, INPUT_SIZE)

    # BGR → RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Normalize to [-1, 1] (standard ArcFace preprocessing)
    face = face.astype(np.float32)
    face = (face - 127.5) / 127.5

    # HWC → CHW → NCHW
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)

    return face


def get_embedding(face_image: np.ndarray) -> np.ndarray | None:
    """
    Extract a 512-d face embedding from a BGR face crop using ArcFace.

    Args:
        face_image: BGR OpenCV image (cropped to face region)

    Returns:
        numpy array of shape (512,) or None if extraction fails.
    """
    _ensure_model_loaded()

    try:
        input_tensor = _preprocess_face(face_image)
        outputs = _session.run(None, {_input_name: input_tensor})
        embedding = outputs[0].flatten().astype(np.float32)

        # L2-normalize the embedding
        emb_norm = norm(embedding)
        if emb_norm > 0:
            embedding = embedding / emb_norm

        return embedding
    except Exception as e:
        print(f"[Recognition Error] Embedding extraction failed: {e}")
        return None


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors (0 = dissimilar, 1 = identical)."""
    dot_product = np.dot(vec_a, vec_b)
    norm_a = norm(vec_a)
    norm_b = norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def compare_with_employees(embedding: np.ndarray, employees: list[dict]) -> dict | None:
    """
    Compare a face embedding against all stored employee embeddings.

    Args:
        embedding: 512-d numpy vector of the detected face.
        employees: list of employee dicts from database (each has 'embedding' as bytes).

    Returns:
        Best match dict with keys: employee_id, full_name, department, confidence.
        Or None if no match meets the threshold.
    """
    best_match = None
    best_score = -1.0

    for emp in employees:
        # Deserialize stored embedding from bytes
        stored_emb = np.frombuffer(emp["embedding"], dtype=np.float32)

        if stored_emb.shape[0] != EMBEDDING_DIM:
            continue

        score = cosine_similarity(embedding, stored_emb)

        if score > best_score:
            best_score = score
            best_match = emp

    if best_match and best_score >= MATCH_THRESHOLD:
        return {
            "employee_id": best_match["employee_id"],
            "full_name": best_match["full_name"],
            "department": best_match.get("department", ""),
            "confidence": round(best_score, 4),
        }

    return None


def compute_average_embedding(face_images: list[np.ndarray]) -> np.ndarray | None:
    """
    Extract embeddings from multiple face images and return their average.
    Used during employee registration for a robust, noise-resistant template.

    Args:
        face_images: list of BGR OpenCV images (cropped to face).

    Returns:
        Averaged 512-d embedding or None if no valid embeddings found.
    """
    embeddings = []

    for img in face_images:
        emb = get_embedding(img)
        if emb is not None:
            embeddings.append(emb)

    if len(embeddings) == 0:
        return None

    # Average all embeddings for a robust representation
    avg_embedding = np.mean(embeddings, axis=0).astype(np.float32)

    # L2-normalize the averaged embedding
    avg_norm = norm(avg_embedding)
    if avg_norm > 0:
        avg_embedding = avg_embedding / avg_norm

    return avg_embedding


def save_face_images(employee_id: str, face_images: list[np.ndarray]) -> list[str]:
    """
    Save face images to the local dataset folder.

    Args:
        employee_id: The employee's ID (used as subfolder name).
        face_images: list of BGR OpenCV images.

    Returns:
        List of saved file paths.
    """
    emp_dir = os.path.join(DATASET_DIR, employee_id)
    os.makedirs(emp_dir, exist_ok=True)

    saved_paths = []
    for i, img in enumerate(face_images):
        filename = f"img_{i+1:03d}.jpg"
        filepath = os.path.join(emp_dir, filename)
        cv2.imwrite(filepath, img)
        saved_paths.append(filepath)

    return saved_paths
