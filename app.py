import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA
# ==========================================
st.set_page_config(
    page_title="SadarPostur - Cek Kebiasaan Dudukmu",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan elegan, hangat, dan ramah
st.markdown("""
<style>
    /* Tipografi & Warna Latar */
    .main {
        background-color: #fbfbfd;
    }
    .hero-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 32px;
        border-radius: 18px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.08);
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 10px;
        color: #f8fafc;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #cbd5e1;
    }
    .info-card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.04);
    }
    .metric-badge {
        font-weight: 700;
        font-size: 1.25rem;
        padding: 6px 14px;
        border-radius: 8px;
        display: inline-block;
        margin-bottom: 8px;
    }
    .badge-good {
        background-color: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }
    .badge-bad {
        background-color: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
    }
    .alert-danger-box {
        background-color: #fff1f2;
        border-left: 5px solid #e11d48;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 14px;
        color: #881337;
    }
    .alert-success-box {
        background-color: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 14px;
        color: #14532d;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PEMUATAN MODEL (CACHED)
# ==========================================
@st.cache_resource
def load_models():
    rf = joblib.load("model_postur_rf.pkl")
    base_options = python.BaseOptions(model_asset_path="pose_landmarker.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        min_pose_detection_confidence=0.1,
        min_tracking_confidence=0.1,
        output_segmentation_masks=False
    )
    det = vision.PoseLandmarker.create_from_options(options)
    return rf, det

try:
    rf_model, detector = load_models()
except Exception as e:
    st.error(f"Gagal memuat aset model: {e}. Pastikan file `model_postur_rf.pkl` dan `pose_landmarker.task` berada dalam folder yang sama.")
    st.stop()

# Helper perhitungan geometri
def calculate_angle(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    radians = np.arctan2(abs(dx), abs(dy) + 1e-6)
    return float(np.degrees(radians))

def get_landmark_score(point):
    p = getattr(point, 'presence', 0.0)
    v = getattr(point, 'visibility', 0.0)
    return max(p, v)

# ==========================================
# 3. SIDEBAR: PUSAT EDUKASI & FAKTA MEDIS
# ==========================================
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?auto=format&fit=crop&w=600&q=80", use_container_width=True)
    st.markdown("### ⚠️ Sadarkah Anda?")
    st.write(
        "Kebiasaan duduk bungkuk berjam-jam bukan sekadar masalah penampilan, "
        "melainkan **ancaman kesehatan jangka panjang** yang sering diabaikan."
    )
    
    st.markdown("---")
    st.markdown("#### 🚨 Risiko Fatal Postur Buruk:")
    st.markdown("""
    * **Herniasi Diskus (Saraf Kejepit):** Tekanan pada bantalan tulang belakang meningkat hingga **200%** saat membungkuk.
    * **Text-Neck & Forward Head:** Setiap 2,5 cm kepala condong ke depan, beban leher bertambah **±4,5 kg**.
    * **Kapasitas Paru Turun:** Rongga dada terhimpit membuat suplai oksigen ke otak berkurang hingga **30%**, memicu rasa kantuk dan kelelahan kronis.
    * **Ketegangan Otot Tegang Trapezius:** Pemicu utama sakit kepala tegang (*tension headache*).
    """)
    
    st.markdown("---")
    st.markdown("#### 💡 Prinsip Ergonomi 90-90-90:")
    st.markdown("""
    1. Sudut siku membentuk 90°.
    2. Sudut panggul ke punggung 90°-100°.
    3. Sudut lutut membentuk 90° dengan telapak kaki menapak rata di lantai.
    4. Ketinggian monitor sejajar dengan garis mata.
    """)

# ==========================================
# 4. AREA UTAMA (HERO BANNER)
# ==========================================
st.markdown("""
<div class="hero-box">
    <div class="hero-title">✨ SadarPostur: Cek & Kenali Kebiasaan Dudukmu</div>
    <div class="hero-subtitle">
        Banyak dari kita menghabiskan waktu berjam-jam di depan meja tanpa menyadari bahwa tubuh perlahan mengalami deformasi postur. 
        Gunakan alat evaluasi interaktif ini untuk memeriksa apakah posisi duduk Anda saat ini ergonomis dan aman bagi tulang belakang.
    </div>
</div>
""", unsafe_allow_html=True)

# Petunjuk penggunaan singkat
col_tutor1, col_tutor2, col_tutor3 = st.columns(3)
with col_tutor1:
    st.markdown("##### 1. Atur Posisi Kamera")
    st.caption("Posisikan laptop atau webcam agak dari samping atau sudut serong (±45°–90°) agar bahu dan telinga terlihat jelas.")
with col_tutor2:
    st.markdown("##### 2. Duduk Alami")
    st.caption("Duduklah seperti kebiasaan kerja harian Anda tanpa dibuat-buat agar hasil evaluasi akurat dan objektif.")
with col_tutor3:
    st.markdown("##### 3. Ambil Gambar")
    st.caption("Klik tombol foto di bawah. Algoritma kecerdasan buatan akan menganalisis sudut tulang leher dan torso Anda.")

st.markdown("---")

# ==========================================
# 5. INPUT KAMERA & INTERAKSI
# ==========================================
col_cam, col_res = st.columns([1.1, 1], gap="medium")

with col_cam:
    st.markdown("#### 📷 Pemeriksaan Kamera")
    img_file_buffer = st.camera_input("Ambil foto posisi duduk Anda saat ini:")

with col_res:
    st.markdown("#### 📋 Hasil Analisis & Rekomendasi Medis")

    if img_file_buffer is None:
        st.info("👋 **Kamera siap.** Silakan posisikan diri Anda di depan kamera lalu klik tombol ambil foto untuk memulai pengecekan postur.")
    else:
        # Proses Citra
        image_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        frame = cv2.imdecode(image_bytes, 1)
        h, w, _ = frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detection_result = detector.detect(mp_image)

        if not detection_result.pose_landmarks:
            st.warning("⚠️ **Tubuh Belum Terdeteksi.** Pastikan pencahayaan ruangan cukup terang dan tubuh bagian atas (kepala hingga dada/pinggul) tidak terpotong tepi layar.")
        else:
            lm = detection_result.pose_landmarks[0]
            vis_left = get_landmark_score(lm[7]) + get_landmark_score(lm[11])
            vis_right = get_landmark_score(lm[8]) + get_landmark_score(lm[12])

            if vis_left >= vis_right:
                ear, shoulder, hip = lm[7], lm[11], lm[23]
            else:
                ear, shoulder, hip = lm[8], lm[12], lm[24]

            nose = lm[0]
            head_pt = ear if get_landmark_score(ear) > 0.1 else nose

            if get_landmark_score(shoulder) < 0.1 or get_landmark_score(head_pt) < 0.1:
                st.warning("⚠️ **Bagian Kepala atau Bahu Terhalang.** Singkirkan barang atau pakaian tebal yang menutupi siluet leher dan bahu Anda.")
            else:
                # Perhitungan Fitur
                ref_dist = np.sqrt((head_pt.x - shoulder.x)**2 + (head_pt.y - shoulder.y)**2) + 1e-6
                neck_angle = calculate_angle([head_pt.x, head_pt.y], [shoulder.x, shoulder.y])

                if get_landmark_score(hip) > 0.1:
                    torso_angle = calculate_angle([shoulder.x, shoulder.y], [hip.x, hip.y])
                    hip_visible = 1.0
                else:
                    torso_angle = neck_angle
                    hip_visible = 0.0

                head_forward_norm = abs(head_pt.x - shoulder.x) / ref_dist
                head_drop_norm = abs(head_pt.y - shoulder.y) / ref_dist

                # Prediksi Model
                features = np.array([[neck_angle, torso_angle, head_forward_norm, head_drop_norm, hip_visible]])
                pred = rf_model.predict(features)[0]
                prob = rf_model.predict_proba(features)[0][pred] * 100

                # Render Visualisasi pada Gambar
                color = (40, 40, 230) if pred == 1 else (40, 180, 40) # BGR
                pt_head = (int(head_pt.x * w), int(head_pt.y * h))
                pt_shoulder = (int(shoulder.x * w), int(shoulder.y * h))
                cv2.circle(frame, pt_head, 9, color, -1)
                cv2.circle(frame, pt_shoulder, 9, color, -1)
                cv2.line(frame, pt_head, pt_shoulder, color, 4)

                # Tampilkan Kartu Metrik Ergonomis
                if pred == 1:
                    st.markdown('<span class="metric-badge badge-bad">⚠️ PERINGATAN: POSTUR BUNGKUK (POOR POSTURE)</span>', unsafe_allow_html=True)
                    st.markdown(f"**Tingkat Kepastian Analisis:** `{prob:.1f}%`")
                    
                    st.markdown(f"""
                    <div class="alert-danger-box">
                        <strong>Perhatian!</strong> Kepala Anda condong ke depan sebesar <b>{neck_angle:.1f}°</b> melebihi batas ergonomi netral (&lt; 20°). 
                        Jika dibiarkan terus-menerus, posisi ini memberi tekanan berlebih pada bantalan servikal leher dan mempercepat keausan sendi tulang belakang.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("##### 🛠️ Langkah Perbaikan Segera:")
                    st.markdown("""
                    * **Tarik Dagu ke Belakang (*Chin Tuck*):** Posisikan telinga sejajar lurus dengan puncak bahu Anda.
                    * **Tegakkan Punggung Bawah:** Gunakan bantalan kecil (*lumbar roll*) untuk menyangga lekukan punggung bawah Anda.
                    * **Naikkan Layar Monitor:** Pastikan bagian atas layar sejajar tepat dengan garis pandang horizontal mata Anda.
                    """)
                else:
                    st.markdown('<span class="metric-badge badge-good">✅ LUAR BIASA: POSTUR ERGONOMIS & SEHAT</span>', unsafe_allow_html=True)
                    st.markdown(f"**Tingkat Kepastian Analisis:** `{prob:.1f}%`")
                    
                    st.markdown(f"""
                    <div class="alert-success-box">
                        <strong>Bagus Sekali!</strong> Penjajaran leher Anda berada pada sudut netral (<b>{neck_angle:.1f}°</b>). 
                        Beban berat kepala terdistribusi seimbang pada seluruh ruas tulang belakang tanpa membebani otot leher secara berlebih.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("##### 💡 Tips Menjaga Konsistensi:")
                    st.markdown("""
                    * **Terapkan Aturan 20-20-20:** Setiap 20 menit bekerja, alihkan pandangan sejauh 6 meter selama 20 detik.
                    * **Berdiri Sejenak:** Sempatkan untuk berdiri atau melakukan peregangan ringan setiap 45–60 menit.
                    """)

                # Parameter Detail
                with st.expander("🔍 Lihat Rincian Parameter Biomekanik"):
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric(label="Kemiringan Kepala-Leher", value=f"{neck_angle:.1f}°", delta="Target: < 20°", delta_color="inverse")
                    col_m2.metric(label="Kemiringan Torso", value=f"{torso_angle:.1f}°", delta="Target: < 15°", delta_color="inverse")
                    st.caption("Titik deteksi warna pada leher dan bahu memvisualisasikan sumbu gravitasi tubuh terhadap pusat tumpuan duduk.")

# ==========================================
# 6. FOOTER EDUKASI
# ==========================================
st.markdown("---")
st.markdown(
    "<center><small style='color: #64748b;'>"
    "✨ SadarPostur • Inisiatif Edukasi Kesehatan Tulang Belakang & Ergonomi Sehari-hari • "
    "Dikembangkan menggunakan Analisis Spesial MediaPipe Pose & Klasifikasi Random Forest."
    "</small></center>",
    unsafe_allow_html=True
)