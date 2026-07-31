"""
Anti-Spoofing Calibration Tool
==============================

Records labelled samples from your webcam, then measures how well every
anti-spoof signal actually separates real faces from replay attacks, and
recommends thresholds based on YOUR camera, lighting and phone.

This exists because thresholds cannot be guessed. The shipped defaults were
calibrated on synthetic data and several never fired at all.

USAGE
-----
Step 1 — record yourself, live in front of the camera:

    python scripts/calibrate_antispoof.py capture --label real

Step 2 — display a still PHOTO of your face on your phone, hold it to the camera:

    python scripts/calibrate_antispoof.py capture --label phone_photo

Step 3 — play a VIDEO of your face on your phone, hold it to the camera:

    python scripts/calibrate_antispoof.py capture --label phone_video

Step 4 — analyse and get recommended thresholds:

    python scripts/calibrate_antispoof.py calibrate

Capture 2-3 rounds of each label under different lighting for a solid result.
"""

import argparse
import os
import sys
import time
import glob
import json

import cv2
import numpy as np

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, APP_DIR)

SAMPLE_DIR = os.path.join(APP_DIR, "calibration_samples")
LABELS = ("real", "phone_photo", "phone_video", "printed_photo")
SPOOF_LABELS = ("phone_photo", "phone_video", "printed_photo")
CASCADE = os.path.join(APP_DIR, "models", "haarcascade_frontalface_default.xml")


# ──────────────────────────────────────────────────────────────
# Capture
# ──────────────────────────────────────────────────────────────

def capture(label: str, n_frames: int, cam_index: int):
    if label not in LABELS:
        print(f"Unknown label '{label}'. Choose from: {', '.join(LABELS)}")
        return 1

    out_dir = os.path.join(SAMPLE_DIR, label)
    os.makedirs(out_dir, exist_ok=True)
    existing = len(glob.glob(os.path.join(out_dir, "*.jpg")))

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Could not open camera {cam_index}.")
        return 1

    det = cv2.CascadeClassifier(CASCADE)
    print(f"\n=== Capturing '{label}' ({n_frames} frames) ===")
    if label == "real":
        print("Sit normally in front of the camera. Move a little, blink naturally.")
    else:
        print("Hold the phone/print steady in the camera view, filling a good part of the frame.")
    print("A 3 second countdown starts once a face is visible. Press Q to abort.\n")

    saved = 0
    countdown_from = None
    while saved < n_frames:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = det.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        view = frame.copy()

        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 220, 0), 2)
            if countdown_from is None:
                countdown_from = time.time()
            remaining = 3.0 - (time.time() - countdown_from)
            if remaining > 0:
                cv2.putText(view, f"Starting in {remaining:.1f}s", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 255), 2)
            else:
                cv2.imwrite(os.path.join(out_dir, f"{label}_{existing + saved:04d}.jpg"), frame)
                saved += 1
                cv2.putText(view, f"REC {saved}/{n_frames}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            countdown_from = None
            cv2.putText(view, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow(f"Capture: {label}", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Aborted.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} frames to {out_dir}")
    total = {l: len(glob.glob(os.path.join(SAMPLE_DIR, l, '*.jpg'))) for l in LABELS}
    print("Sample counts so far:", {k: v for k, v in total.items() if v})
    return 0


# ──────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────

def extract_features(paths):
    """Compute every anti-spoof signal for each sample frame."""
    from utils.anti_spoofing import run_minifas
    from liveness import (_compute_fft_moire_score, _compute_glare_ratio,
                          _compute_skin_chroma_score, _compute_laplacian_variance,
                          _compute_lbp_variance)

    det = cv2.CascadeClassifier(CASCADE)
    rows = []
    for p in paths:
        frame = cv2.imread(p)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = det.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if not len(faces):
            continue
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = frame[y:y + h, x:x + w]
        if face.size == 0:
            continue
        bbox = {"originX": int(x), "originY": int(y), "width": int(w), "height": int(h)}
        _, chroma = _compute_skin_chroma_score(face)
        rows.append({
            "p_real": run_minifas(face, bbox, frame)["p_real"],
            "fft_moire": _compute_fft_moire_score(face),
            "glare": _compute_glare_ratio(face),
            "chroma": chroma,
            "laplacian": _compute_laplacian_variance(face),
            "lbp": _compute_lbp_variance(face),
        })
    return rows


def _best_threshold(real_vals, spoof_vals, spoof_is_lower, min_real_accept=0.98):
    """Find the threshold maximising spoof rejection while keeping real acceptance."""
    cand = np.unique(np.concatenate([real_vals, spoof_vals]))
    if len(cand) < 2:
        return None
    best = None
    for t in cand:
        if spoof_is_lower:
            racc = (real_vals >= t).mean()
            sblk = (spoof_vals < t).mean()
        else:
            racc = (real_vals <= t).mean()
            sblk = (spoof_vals > t).mean()
        if racc >= min_real_accept and (best is None or sblk > best[1]):
            best = (float(t), float(sblk), float(racc))
    return best


def calibrate(min_real_accept: float):
    data = {}
    for label in LABELS:
        paths = sorted(glob.glob(os.path.join(SAMPLE_DIR, label, "*.jpg")))
        if paths:
            print(f"Extracting features: {label} ({len(paths)} frames)...")
            rows = extract_features(paths)
            if rows:
                data[label] = rows

    if "real" not in data:
        print(f"\nNo 'real' samples found in {SAMPLE_DIR}.")
        print("Run:  python scripts/calibrate_antispoof.py capture --label real")
        return 1
    if not any(l in data for l in SPOOF_LABELS):
        print(f"\nNo spoof samples found. Capture at least one of: {', '.join(SPOOF_LABELS)}")
        return 1

    signals = ["p_real", "fft_moire", "glare", "chroma", "laplacian", "lbp"]
    # For these signals a LOWER value indicates a spoof.
    lower_is_spoof = {"p_real": True, "fft_moire": False, "glare": False,
                      "chroma": True, "laplacian": True, "lbp": True}

    def col(label, sig):
        return np.array([r[sig] for r in data[label]])

    print("\n" + "=" * 78)
    print("MEASURED DISTRIBUTIONS")
    print("=" * 78)
    hdr = f"{'signal':>10} |" + "".join(f" {l:>15} |" for l in data)
    print(hdr)
    print("-" * len(hdr))
    for sig in signals:
        line = f"{sig:>10} |"
        for l in data:
            v = col(l, sig)
            line += f" {v.mean():7.3f}±{v.std():6.3f} |"
        print(line)

    spoof_all = np.concatenate([[r for r in data[l]] for l in SPOOF_LABELS if l in data])
    print("\n" + "=" * 78)
    print(f"RECOMMENDED THRESHOLDS  (keeping real-face acceptance >= {min_real_accept:.0%})")
    print("=" * 78)

    recs = {}
    for sig in signals:
        rv = col("real", sig)
        sv = np.array([r[sig] for r in spoof_all])
        res = _best_threshold(rv, sv, lower_is_spoof[sig], min_real_accept)
        d = abs(rv.mean() - sv.mean()) / np.sqrt((rv.var() + sv.var()) / 2 + 1e-9)
        if res is None:
            print(f"  {sig:>10}: UNUSABLE — cannot separate at this acceptance rate  (d'={d:.2f})")
            continue
        t, sblk, racc = res
        op = ">=" if lower_is_spoof[sig] else "<="
        verdict = "STRONG" if sblk >= .8 else ("USABLE" if sblk >= .4 else "WEAK")
        print(f"  {sig:>10}: accept when value {op} {t:8.4f}  -> blocks {sblk:5.1%} of spoofs, "
              f"accepts {racc:5.1%} of real  (d'={d:.2f})  [{verdict}]")
        recs[sig] = {"threshold": t, "spoof_block": sblk, "real_accept": racc,
                     "direction": op, "dprime": float(d)}

    # Per attack type, using the strongest signal found
    if recs:
        best_sig = max(recs, key=lambda s: recs[s]["spoof_block"])
        print(f"\nStrongest single signal: {best_sig} "
              f"(blocks {recs[best_sig]['spoof_block']:.1%} of spoofs)")
        print("\nPer attack type with that signal:")
        t = recs[best_sig]["threshold"]
        for l in SPOOF_LABELS:
            if l not in data:
                continue
            v = col(l, best_sig)
            blk = (v < t).mean() if lower_is_spoof[best_sig] else (v > t).mean()
            print(f"   {l:>14}: {blk:6.1%} blocked  (n={len(v)})")

    out = os.path.join(SAMPLE_DIR, "calibration_results.json")
    with open(out, "w") as f:
        json.dump({"min_real_accept": min_real_accept,
                   "counts": {l: len(data[l]) for l in data},
                   "recommendations": recs}, f, indent=2)
    print(f"\nSaved -> {out}")
    print("\nApply the p_real threshold to MINIFAS_REAL_THRESHOLD in utils/anti_spoofing.py,")
    print("and the rest to the corresponding constants in liveness.py:check_screen_spoof().")
    print("Re-enable Model A blocking in auth.py only if p_real shows STRONG separation.")
    return 0


WIZARD_STEPS = [
    ("real", "Sit in front of the camera as you normally would to log in.\n"
             "  Move a little and blink naturally."),
    ("phone_video", "Play the VIDEO of your face on your phone and hold the phone\n"
                    "  up to the webcam, filling a good part of the frame.\n"
                    "  Move the phone slightly, try a few angles and distances."),
    ("phone_photo", "Show a still PHOTO of your face on your phone and hold it up\n"
                    "  to the webcam the same way."),
]


def wizard(n_frames: int, cam_index: int):
    """Guided capture of every label in one run."""
    print("\n" + "=" * 66)
    print("ANTI-SPOOF CALIBRATION WIZARD")
    print("=" * 66)
    print("Three short recordings. Have your phone ready with both a video")
    print("and a photo of your own face.\n")

    for i, (label, instruction) in enumerate(WIZARD_STEPS, 1):
        print("-" * 66)
        print(f"STEP {i} of {len(WIZARD_STEPS)}  —  '{label}'")
        print(f"  {instruction}")
        print("-" * 66)
        try:
            input("Press ENTER when ready (or Ctrl+C to stop)... ")
        except (EOFError, KeyboardInterrupt):
            print("\nStopped.")
            return 1
        if capture(label, n_frames, cam_index) != 0:
            return 1
        print()

    print("=" * 66)
    print("Capture complete — analysing...")
    print("=" * 66)
    return calibrate(0.98)


def main():
    ap = argparse.ArgumentParser(description="Anti-spoofing calibration tool")
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("wizard", help="guided capture of all labels, then calibrate")
    w.add_argument("--frames", type=int, default=60)
    w.add_argument("--camera", type=int, default=0)

    c = sub.add_parser("capture", help="record labelled samples from the webcam")
    c.add_argument("--label", required=True, choices=LABELS)
    c.add_argument("--frames", type=int, default=60)
    c.add_argument("--camera", type=int, default=0)

    k = sub.add_parser("calibrate", help="analyse samples and recommend thresholds")
    k.add_argument("--min-real-accept", type=float, default=0.98,
                   help="minimum fraction of real faces that must still be accepted")

    a = ap.parse_args()
    if a.cmd == "wizard":
        return wizard(a.frames, a.camera)
    if a.cmd == "capture":
        return capture(a.label, a.frames, a.camera)
    return calibrate(a.min_real_accept)


if __name__ == "__main__":
    sys.exit(main())
