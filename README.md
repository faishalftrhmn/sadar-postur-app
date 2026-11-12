# 🪑 SadarPostur: Sistem Deteksi Postur Duduk Ergonomis Berbasis YOLO11n & Pemetaan Biomekanik

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11n-00FFFF)
![MediaPipe](https://img.shields.io/badge/MediaPipe-PoseLandmarker-0097A7)

**SadarPostur** adalah aplikasi visi komputer interaktif berbasis web untuk mendeteksi dan mengevaluasi kelayakan postur duduk secara *real-time*. Sistem ini dirancang untuk membantu pengguna mengenali kebiasaan duduk membungkuk (*bad posture*) sejak dini guna mencegah risiko gangguan muskuloskeletal seperti nyeri punggung bawah (*low back pain*) dan *forward head posture*.

---

## 🧠 Arsitektur Sistem (*Hybrid Decision Fusion*)

Sistem ini mengintegrasikan dua pendekatan visi komputer secara komplementer:
1. **Klasifikasi Visual Utama (`YOLO11n`):** Mendeteksi subjek dan mengklasifikasikan kondisi postur ke dalam dua kelas (`Good Posture` dan `Bad Posture`) berdasarkan fitur visual menyeluruh dengan kecepatan inferensi tinggi.
2. **Pemetaan & Verifikasi Biomekanik (`MediaPipe PoseLandmarker`):** Mengekstraksi titik koordinat anatomis (**telinga, bahu, dan pinggul**) untuk memvisualisasikan garis sumbu tulang belakang, menghitung sudut kemiringan leher (*neck angle*) serta torso (*torso angle*), dan memverifikasi hasil deteksi pada kondisi pencahayaan rendah (*low-light*).

---

## 📊 Performa & Evaluasi Model

Model dilatih menggunakan dataset citra postur duduk yang telah dibagi secara proporsional dengan rasio **80% Train : 10% Validation : 10% Test**.

### 1. Hasil Evaluasi YOLO11n
| Tahap Evaluasi | Jumlah Citra | mAP@50 (Akurasi) | mAP@50-95 | Precision | Recall | Kecepatan Inferensi |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Set (10%)** | 847 | **98,30%** | 87,40% | 96,90% | 94,10% | 3,8 ms/citra |
| **Test Set (10% - Independen)** | 847 | **96,41%** | 84,90% | 92,88% | 87,82% | 3,9 ms/citra |

### 2. Rincian Performa per Kelas (Test Set)
* **Good Posture (Kelas 0):** mAP@50 **97,30%** | Precision **94,60%** | Recall **90,00%**
* **Bad Posture (Kelas 1):** mAP@50 **95,50%** | Precision **91,20%** | Recall **85,60%**

### 3. Peningkatan dari Metode Sebelumnya
Peralihan arsitektur dari pendekatan klasik (*MediaPipe Keypoints + Random Forest*) ke **YOLO11n** meningkatkan akurasi deteksi sebesar **+19,02%** (dari **77,39%** menjadi **96,41%**).

---

## 📁 Struktur Direktori Proyek

```text
sadar-postur-app/
│
├── models/
│   ├── best.pt                     # Bobot model YOLO11n hasil fine-tuning
│   └── pose_landmarker_lite.task   # Aset model MediaPipe PoseLandmarker
│
├── app.py                          # Antarmuka web utama (Streamlit UI & edukasi ergonomi)
├── cv_engine.py                    # Mesin deteksi hibrida YOLO11n + garis biomekanik
├── packages.txt                    # Dependensi sistem Linux untuk OpenCV (Streamlit Cloud)
├── requirements.txt                # Daftar pustaka Python yang dibutuhkan
├── .gitignore                      # Konfigurasi pengecualian berkas sementara & dataset
└── README.md                       # Dokumentasi proyek
```

---

## 🚀 Cara Instalasi & Menjalankan di Lokal

1. **Kloning repositori ini:**
   ```bash
   git clone [https://github.com/faishalftrhmn/sadar-postur-app.git](https://github.com/faishalftrhmn/sadar-postur-app.git)
   cd sadar-postur-app
   ```

2. **Instal seluruh dependensi Python:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi Streamlit:**
   ```bash
   streamlit run app.py
   ```

4. Buka peramban (*browser*) dan akses alamat lokal yang muncul (umumnya `http://localhost:8501`).

---

## 💡 Panduan Penggunaan Singkat

1. **Posisi Kamera:** Arahkan kamera dari sudut samping atau serong (±45° – 90°) dengan pencahayaan ruangan yang memadai agar siluet telinga, bahu, dan punggung terlihat jelas.
2. **Pilih Metode Input:** Gunakan tab **Kamera Langsung** untuk pengecekan *real-time* melalui webcam, atau tab **Unggah Foto** untuk menguji berkas gambar (`.jpg`, `.jpeg`, `.png`).
3. **Evaluasi Hasil:** Sistem akan menampilkan *bounding box* klasifikasi, garis penjajaran biomekanik, sudut kemiringan leher, serta rekomendasi koreksi postur secara otomatis.

---

## 👨‍💻 Pengembang

Dikembangkan oleh **Faishal Fathurrahman** ([@faishalftrhmn](https://github.com/faishalftrhmn)) sebagai bagian dari penelitian dan pengembangan sistem pemantauan ergonomi berbasis *Deep Learning*.
