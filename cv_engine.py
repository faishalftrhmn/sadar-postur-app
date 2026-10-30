# cv_engine.py
import cv2

def draw_posture_annotations(frame, head_pt, shoulder, hip, hip_visible, pred):
    """
    Menggambar titik anatomis dan garis kerangka pada citra.
    - Garis Merah jika Bad Posture (pred == 1)
    - Garis Hijau jika Good Posture (pred == 0)
    """
    annotated_frame = frame.copy()
    h, w, _ = annotated_frame.shape

    # Tentukan warna (Format BGR bawaan OpenCV)
    # Merah: (40, 40, 230) | Hijau: (40, 180, 40)
    color = (40, 40, 230) if pred == 1 else (40, 180, 40)

    # Konversi koordinat normalisasi MediaPipe (0.0 - 1.0) ke koordinat piksel
    pt_head = (int(head_pt.x * w), int(head_pt.y * h))
    pt_shoulder = (int(shoulder.x * w), int(shoulder.y * h))

    # 1. Gambar titik dan garis leher (Kepala/Telinga ke Bahu)
    cv2.circle(annotated_frame, pt_head, 8, (255, 255, 255), -1)       # Titik pusat putih
    cv2.circle(annotated_frame, pt_head, 10, color, 2)                 # Lingkaran luar berwarna
    cv2.circle(annotated_frame, pt_shoulder, 8, (255, 255, 255), -1)
    cv2.circle(annotated_frame, pt_shoulder, 10, color, 2)
    cv2.line(annotated_frame, pt_head, pt_shoulder, color, 4)          # Garis leher

    # 2. Gambar garis torso jika pinggul terdeteksi
    if hip_visible == 1.0:
        pt_hip = (int(hip.x * w), int(hip.y * h))
        cv2.circle(annotated_frame, pt_hip, 8, (255, 255, 255), -1)
        cv2.circle(annotated_frame, pt_hip, 10, color, 2)
        cv2.line(annotated_frame, pt_shoulder, pt_hip, color, 4)       # Garis torso

    # Konversi dari BGR ke RGB agar warna di Streamlit tampil natural
    return cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
