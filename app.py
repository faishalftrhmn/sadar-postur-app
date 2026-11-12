import cv2
import numpy as np
from PIL import Image
import streamlit as st
from cv_engine import SadarPosturHybridEngine

# ==========================================
# 1. KONFIGURASI HALAMAN & TEMA
# ==========================================
st.set_page_config(
    page_title="SadarPostur - Cek Kebiasaan Dudukmu",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
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
    .metric-badge {
        font-weight: 700;
        font-size: 1.2rem;
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
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. PEMUATAN ENGINE HIBRIDA (CACHED)
# ==========================================
@st.cache_resource
def load_engine():
  return SadarPosturHybridEngine(model_path="models/best.pt")


try:
  engine = load_engine()
except Exception as e:
  st.error(
      f"Gagal memuat model: {e}. Pastikan `best.pt` berada di folder"
      " `models/`."
  )
  st.stop()

# ==========================================
# 3. SIDEBAR: PENGATURAN & EDUKASI MEDIS
# ==========================================
with st.sidebar:
  st.image(
      "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b?auto=format&fit=crop&w=600&q=80",
      use_container_width=True,
  )

  st.markdown("### ⚙️ Pengaturan Sensitivitas")
  conf_threshold = st.slider(
      "Ambang Batas Deteksi (Confidence)",
      min_value=0.10,
      max_value=0.90,
      value=0.35,
      step=0.05,
      help="Turunkan nilai ini jika kamera belum mendeteksi tubuh Anda.",
  )

  st.markdown("---")
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
    * **Kapasitas Paru Turun:** Rongga dada terhimpit membuat suplai oksigen ke otak berkurang hingga **30%**.
    * **Ketegangan Otot Trapezius:** Pemicu utama sakit kepala tegang (*tension headache*).
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
st.markdown(
    """
<div class="hero-box">
    <div class="hero-title">✨ SadarPostur: Cek & Kenali Kebiasaan Dudukmu</div>
    <div class="hero-subtitle">
        Evaluasi postur duduk Anda menggunakan arsitektur <b>YOLO11n</b> yang dipadukan dengan pemetaan garis biomekanik 
        telinga–bahu–punggung untuk mendeteksi risiko pembungkukan secara presisi.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

col_tutor1, col_tutor2, col_tutor3 = st.columns(3)
with col_tutor1:
  st.markdown("##### 1. Atur Posisi Kamera")
  st.caption(
      "Posisikan kamera dari sudut samping atau serong (±45°–90°) agar telinga,"
      " bahu, dan punggung terlihat jelas."
  )
with col_tutor2:
  st.markdown("##### 2. Duduk Alami")
  st.caption(
      "Duduklah seperti kebiasaan kerja harian Anda tanpa dibuat-buat agar"
      " hasil evaluasi objektif."
  )
with col_tutor3:
  st.markdown("##### 3. Ambil / Unggah Gambar")
  st.caption(
      "Klik ambil foto atau unggah gambar. Sistem akan menampilkan kotak"
      " klasifikasi YOLO11n beserta garis sumbu postur."
  )

st.markdown("---")

# ==========================================
# 5. INPUT KAMERA / UPLOAD & HASIL ANALISIS
# ==========================================
col_cam, col_res = st.columns([1.1, 1], gap="medium")

with col_cam:
  st.markdown("#### 📷 Sumber Citra Pemeriksaan")
  tab_cam, tab_upload = st.tabs(["🎥 Kamera Langsung", "📁 Unggah Foto"])

  img_source = None
  with tab_cam:
    cam_buffer = st.camera_input("Ambil foto posisi duduk Anda saat ini:")
    if cam_buffer is not None:
      img_source = cam_buffer

  with tab_upload:
    upload_buffer = st.file_uploader(
        "Pilih berkas gambar postur duduk (.jpg, .jpeg, .png):",
        type=["jpg", "jpeg", "png"],
    )
    if upload_buffer is not None:
      img_source = upload_buffer

with col_res:
  st.markdown("#### 📋 Hasil Analisis & Rekomendasi Medis")

  if img_source is None:
    st.info(
        "👋 **Sistem siap.** Ambil foto melalui kamera atau unggah gambar pada"
        " panel sebelah kiri untuk memulai pengecekan."
    )
  else:
    pil_img = Image.open(img_source).convert("RGB")
    frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Jalankan Analisis Hibrida (YOLO11n + Garis Keypoints)
    annotated_rgb, detections, biomechanics, inference_ms = (
        engine.analyze_frame(frame_bgr, conf_threshold=conf_threshold)
    )

    st.image(
        annotated_rgb,
        caption="Deteksi YOLO11n & Garis Penjajaran Telinga–Bahu–Punggung",
        use_container_width=True,
    )

    if not detections:
      st.warning(
          "⚠️ **Postur Belum Terdeteksi oleh YOLO11n.** Coba geser slider"
          " *Ambang Batas Deteksi* di sidebar ke angka lebih rendah (misal"
          " `0.20`) atau pastikan tubuh bagian atas masuk ke layar."
      )
    else:
      utama = detections[0]
      pred = utama["class_id"]  # 0: Good Posture, 1: Bad Posture
      prob = utama["confidence"]

      neck_info = (
          f" (sudut leher **{biomechanics['neck_angle']:.1f}°**)"
          if biomechanics
          else ""
      )

      if pred == 1:
        st.markdown(
            '<span class="metric-badge badge-bad">⚠️ PERINGATAN: POSTUR BUNGKUK'
            " (BAD POSTURE)</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Tingkat Keyakinan YOLO11n:** `{prob:.2f}%` | **Waktu Proses:**"
            f" `{inference_ms:.1f} ms`"
        )

        st.markdown(
            f"""
                <div class="alert-danger-box">
                    <strong>Perhatian!</strong> Sistem mendeteksi postur duduk tidak ergonomis{neck_info}. 
                    Posisi kepala dan punggung yang condong ke depan memberi tekanan berlebih pada bantalan tulang belakang.
                </div>
                """,
            unsafe_allow_html=True,
        )

        st.markdown("##### 🛠️ Langkah Perbaikan Segera:")
        st.markdown("""
                * **Tarik Dagu ke Belakang (*Chin Tuck*):** Posisikan telinga sejajar lurus dengan puncak bahu.
                * **Tegakkan Punggung Bawah:** Sandarkan punggung tegak mengikuti garis vertikal kursi.
                * **Naikkan Layar Monitor:** Sejajarkan bagian atas layar dengan garis pandang mata.
                """)
      else:
        st.markdown(
            '<span class="metric-badge badge-good">✅ LUAR BIASA: POSTUR'
            " ERGONOMIS (GOOD POSTURE)</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Tingkat Keyakinan YOLO11n:** `{prob:.2f}%` | **Waktu Proses:**"
            f" `{inference_ms:.1f} ms`"
        )

        st.markdown(
            f"""
                <div class="alert-success-box">
                    <strong>Bagus Sekali!</strong> Postur duduk Anda berada pada penjajaran ergonomis{neck_info}. 
                    Beban kepala terdistribusi seimbang pada ruas tulang belakang.
                </div>
                """,
            unsafe_allow_html=True,
        )

        st.markdown("##### 💡 Tips Menjaga Konsistensi:")
        st.markdown("""
                * **Terapkan Aturan 20-20-20:** Setiap 20 menit bekerja, alihkan pandangan sejauh 6 meter selama 20 detik.
                * **Berdiri Sejenak:** Lakukan peregangan ringan setiap 45–60 menit.
                """)

      # Rincian Parameter Biomekanik & Deteksi
      with st.expander("🔍 Lihat Rincian Parameter Biomekanik & Deteksi"):
        if biomechanics:
          col_m1, col_m2, col_m3 = st.columns(3)
          col_m1.metric(
              label="Sudut Kepala–Leher",
              value=f"{biomechanics['neck_angle']:.1f}°",
              delta="Target: < 20°",
              delta_color="inverse",
          )
          col_m2.metric(
              label="Sudut Torso–Punggung",
              value=f"{biomechanics['torso_angle']:.1f}°",
              delta="Target: < 15°",
              delta_color="inverse",
          )
          col_m3.metric(label="Keyakinan YOLO11n", value=f"{prob:.2f}%")
        else:
          st.caption(
              "Titik sendi anatomis sebagian terhalang, namun klasifikasi tetap"
              " berhasil dilakukan oleh YOLO11n melalui fitur visual"
              " menyeluruh."
          )

# ==========================================
# 6. FOOTER EDUKASI
# ==========================================
st.markdown("---")
st.markdown(
    "<center><small style='color: #64748b;'>✨ SadarPostur • Sistem Deteksi"
    " Ergonomi Berbasis YOLO11n (mAP@50: 96,41%) & Pemetaan Garis"
    " Biomekanik.</small></center>",
    unsafe_allow_html=True,
)