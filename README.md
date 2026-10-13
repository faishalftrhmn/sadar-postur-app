# ✨ SadarPostur: AI-Powered Sitting Posture Detection

SadarPostur adalah platform evaluasi postur duduk interaktif berbasis Computer Vision dan Machine Learning untuk mengedukasi masyarakat mengenai bahaya kebiasaan duduk bungkuk serta pentingnya ergonomi kerja harian.

---

## 📌 Fitur Utama
* **Deteksi Landmark Real-Time:** Memanfaatkan MediaPipe Pose untuk melacak titik anatomis telinga, bahu, dan pinggul.
* **Klasifikasi Machine Learning:** Menggunakan algoritma Random Forest untuk mengklasifikasikan postur tubuh (*Good Posture* vs *Bad Posture*).
* **Rekomendasi Ergonomis:** Memberikan analisis sudut kemiringan kepala-leher serta saran perbaikan postur langsung setelah pengambilan citra.
* **Berbasis Web:** Dibangun dengan Streamlit dan dapat diakses publik melalui browser tanpa instalasi tambahan.

---

## 🛠️ Struktur File
* `app.py` : Antarmuka web utama berbasis Streamlit.
* `model_postur_rf.pkl` : Model Random Forest yang telah dilatih.
* `pose_landmarker.task` : Model MediaPipe Pose Landmarker.
* `requirements.txt` : Daftar pustaka Python yang dibutuhkan aplikasi.

---

## 💻 Cara Menjalankan Secara Lokal

1. **Clone repositori:**
   ```bash
   git clone [https://github.com/faishalftrhmn/sadar-postur-app.git](https://github.com/faishalftrhmn/sadar-postur-app.git)
   cd sadar-postur-app
