"""
Diagnostic Script — Tests liveness detection parameters on live webcam feed.
Captures real-time metrics to calibrate anti-spoofing thresholds.

Usage: Run this script, point webcam at:
  1. Your real face → observe values
  2. A phone screen photo → observe values  
  3. A phone screen video → observe values

Press 'q' to quit.
"""

import cv2
import numpy as np
import time
import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from liveness import (
    _compute_laplacian_variance, _compute_lbp_variance,
    _compute_glare_ratio, _compute_fft_moire_score,
    _compute_skin_chroma_score, check_3d_depth_liveness
)

def main():
    try:
        import mediapipe as mp
    except ImportError:
        print("mediapipe not installed. Run: pip install mediapipe")
        return

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return

    print("=" * 70)
    print("LIVENESS DIAGNOSTIC — Point camera at:")
    print("  1. Your REAL face")
    print("  2. A PHOTO on phone screen")
    print("  3. A VIDEO on phone screen")
    print("Press 'q' to quit")
    print("=" * 70)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:  # Process every 5th frame
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks_mp = results.multi_face_landmarks[0]
            h, w, _ = frame.shape

            # Extract landmarks
            landmarks = [(lm.x, lm.y, lm.z) for lm in landmarks_mp.landmark]
            
            # Compute bounding box
            xs = [lm.x * w for lm in landmarks_mp.landmark]
            ys = [lm.y * h for lm in landmarks_mp.landmark]
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            
            # Pad
            pad = 20
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                # Compute all metrics
                lap_var = _compute_laplacian_variance(face_crop)
                lbp_var = _compute_lbp_variance(face_crop)
                glare_ratio = _compute_glare_ratio(face_crop)
                fft_score = _compute_fft_moire_score(face_crop)
                skin_pass, skin_ratio = _compute_skin_chroma_score(face_crop)
                
                # 3D depth check
                zs = [pt[2] for pt in landmarks]
                z_std = float(np.std(zs))
                z_range = max(zs) - min(zs)
                depth_pass = z_std >= 0.005

                # Blendshapes for blink detection
                blink_left = 0.0
                blink_right = 0.0

                # Edge density (real faces have smoother edges, screens have sharp pixel boundaries)
                gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_crop, 50, 150)
                edge_density = float(np.count_nonzero(edges) / edges.size)

                # Color channel variance (screens emit uniform RGB, real skin has varied color)
                b_var = float(np.var(face_crop[:,:,0]))
                g_var = float(np.var(face_crop[:,:,1]))
                r_var = float(np.var(face_crop[:,:,2]))
                color_var_ratio = min(b_var, g_var, r_var) / (max(b_var, g_var, r_var) + 1e-8)
                
                # Gradient magnitude analysis
                gray_resized = cv2.resize(gray_crop, (128, 128))
                sobelx = cv2.Sobel(gray_resized, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray_resized, cv2.CV_64F, 0, 1, ksize=3)
                gradient_mag = np.sqrt(sobelx**2 + sobely**2)
                gradient_mean = float(np.mean(gradient_mag))
                gradient_std = float(np.std(gradient_mag))

                # HSV saturation analysis (screens often have lower/different saturation)
                hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
                sat_mean = float(np.mean(hsv[:,:,1]))
                sat_std = float(np.std(hsv[:,:,1]))

                print(f"\n{'='*70}")
                print(f"Frame {frame_count}:")
                print(f"  3D Depth:     z_std={z_std:.6f}  z_range={z_range:.6f}  pass={'✅' if depth_pass else '❌'}")
                print(f"  Laplacian:    var={lap_var:.2f}  (thresh >= 15.0)")
                print(f"  LBP Texture:  var={lbp_var:.4f}  (thresh >= 0.5)")
                print(f"  Glare:        ratio={glare_ratio:.4f}  (thresh < 0.15)")
                print(f"  FFT Moiré:    score={fft_score:.4f}  (thresh < 0.080)")
                print(f"  Skin Chroma:  ratio={skin_ratio:.4f}  pass={'✅' if skin_pass else '❌'}")
                print(f"  Edge Density: {edge_density:.4f}")
                print(f"  Color Var:    B={b_var:.1f} G={g_var:.1f} R={r_var:.1f}  ratio={color_var_ratio:.4f}")
                print(f"  Gradient:     mean={gradient_mean:.2f}  std={gradient_std:.2f}")
                print(f"  HSV Sat:      mean={sat_mean:.2f}  std={sat_std:.2f}")
                print(f"{'='*70}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()
