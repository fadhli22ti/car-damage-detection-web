# 🚗 Car Damage Detection Web Application using YOLOv8

Proyek Tugas Akhir ini adalah aplikasi berbasis web yang mampu mendeteksi kerusakan eksterior pada bodi mobil secara otomatis menggunakan algoritma **YOLOv8** (You Only Look Once). Pengguna cukup mengunggah foto mobil, dan sistem akan menandai lokasi kerusakan beserta jenisnya secara otomatis.

## 🛠️ Fitur Utama

- **Deteksi Otomatis**: Mengenali jenis kerusakan dari foto yang diunggah hanya dengan satu klik, tanpa pengaturan manual.
- **Multi-Class Detection**: Mendeteksi 3 jenis kerusakan bodi: `Dent` (penyok), `Scratch` (lecet), dan `Cracked Glass` (kaca retak).
- **Adaptive Confidence**: Sistem secara otomatis mencoba beberapa tingkat sensitivitas (dari ketat ke longgar) untuk menemukan kerusakan terbaik tanpa perlu diatur oleh pengguna. Mode manual juga tersedia di panel "Mode Lanjutan" bagi yang ingin bereksperimen.
- **Unduh Hasil**: Gambar hasil deteksi (lengkap dengan bounding box) dapat langsung diunduh.
- **Antarmuka Sederhana**: Dibangun dengan Streamlit agar mudah digunakan oleh siapa saja tanpa pengetahuan teknis.

## 📂 Struktur Proyek

```
car-damage-detection-web/
├── app.py              # Aplikasi utama Streamlit
├── best.pt              # Model YOLOv8 hasil training
├── requirements.txt      # Daftar dependency Python
└── README.md             # Dokumentasi proyek
```

## 📊 Dataset & Training

- Dataset dasar: [Car Damage Dataset](https://www.kaggle.com/datasets/vinayjose/car-damage-dataset) dari Kaggle.
- Karena dataset asli hanya berupa label klasifikasi (bukan bounding box), seluruh anotasi objek dilakukan secara manual menggunakan [Roboflow](https://roboflow.com) untuk menghasilkan format YOLO (`.txt` per gambar).
- Proses training model dilakukan menggunakan YOLOv8 di Google Colab dengan notebook `Car_Damage_Detection_YOLOv8.ipynb`.

## 🚀 Cara Menjalankan Secara Lokal

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/fadhli22ti/car-damage-detection-web.git
   cd car-damage-detection-web
   ```

2. **Buat virtual environment (opsional tapi disarankan):**
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependency:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pastikan file model `best.pt` ada di folder yang sama dengan `app.py`.**

5. **Jalankan aplikasi:**
   ```bash
   streamlit run app.py
   ```

6. Aplikasi akan otomatis terbuka di browser pada alamat `http://localhost:8501`.

## 📈 Evaluasi Model

Model dilatih menggunakan arsitektur **YOLOv8n (nano)** selama 50 epoch dengan ukuran gambar 640x640. Berikut adalah hasil evaluasi performa model pada data validasi (60 gambar, 69 instances kerusakan):

| Kelas | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Kaca Retak | 0.725 | 0.765 | 0.741 | 0.312 |
| Penyok | 0.318 | 0.261 | 0.211 | 0.064 |
| Lecet | 0.234 | 0.127 | 0.168 | 0.064 |
| **Rata-rata (all)** | **0.426** | **0.384** | **0.373** | **0.147** |

### Analisis Hasil

Dari tabel di atas terlihat bahwa performa model sangat bervariasi antar kelas. Kelas **kaca_retak** memiliki performa jauh lebih baik (mAP50 = 0.741) dibanding **penyok** (mAP50 = 0.211) dan **lecet** (mAP50 = 0.168). Hal ini cukup menarik karena tidak sepenuhnya berkorelasi dengan jumlah data training — kelas *lecet* memiliki jumlah gambar anotasi lebih banyak (138 gambar) dibanding *penyok* (96 gambar), namun justru menunjukkan recall yang lebih rendah (0.127 vs 0.261).

Beberapa faktor yang diduga menjadi penyebab perbedaan performa ini:

1. **Karakteristik visual antar kelas berbeda.** Kerusakan kaca retak memiliki pola visual yang sangat khas dan kontras (garis retakan tajam pada permukaan kaca), sehingga lebih mudah dipelajari oleh model dibanding penyok dan lecet yang berupa perubahan tekstur halus pada bodi mobil dan dapat menyerupai bayangan, pantulan cahaya, atau lekukan desain bodi yang memang sudah ada.
2. **Kemiripan visual antara kelas penyok dan lecet** dapat menyebabkan model kesulitan membedakan keduanya, terutama pada kerusakan yang menggabungkan unsur penyok dan goresan cat secara bersamaan.
3. **Ukuran model yang digunakan (YOLOv8n)** merupakan varian paling ringan dalam keluarga YOLOv8 dengan kapasitas representasi yang lebih terbatas dibanding varian yang lebih besar (YOLOv8s/m/l), sehingga lebih rentan kesulitan pada kelas dengan variasi visual tinggi.
4. **Ukuran data validasi yang kecil** (69 instances) membuat metrik evaluasi ini memiliki margin kesalahan yang relatif besar dan belum tentu sepenuhnya merepresentasikan performa model pada data yang lebih luas.

### Rencana Pengembangan Selanjutnya

- Menambah jumlah dan variasi data anotasi, khususnya untuk kelas *lecet* dan *penyok*.
- Melakukan augmentasi data (rotasi, perubahan pencahayaan, flipping) untuk memperkaya variasi visual tanpa perlu anotasi baru.
- Mencoba varian model yang lebih besar (YOLOv8s atau YOLOv8m) untuk meningkatkan kapasitas representasi.
- Menambah jumlah epoch training dan melakukan validasi pada dataset yang lebih besar untuk hasil evaluasi yang lebih representatif.

## ⚠️ Batasan (Limitations)

- Akurasi deteksi masih bergantung pada jumlah dan variasi data training yang digunakan. Karena dataset anotasi masih terbatas, model terkadang memerlukan confidence threshold rendah (±0.05–0.15) agar dapat mendeteksi kerusakan dengan baik.
- Model dapat menghasilkan false positive (menandai area yang sebenarnya bukan kerusakan) pada kondisi pencahayaan atau sudut foto tertentu.
- Pengembangan lebih lanjut dapat dilakukan dengan menambah jumlah data, augmentasi gambar, atau menggunakan teknik *model-assisted labeling* di Roboflow untuk mempercepat dan memperluas anotasi.

## 🧰 Teknologi yang Digunakan

- [YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics) — model deteksi objek
- [Streamlit](https://streamlit.io) — framework web app
- [Roboflow](https://roboflow.com) — anotasi dan manajemen dataset
- [OpenCV](https://opencv.org) & [Pillow](https://python-pillow.org) — pemrosesan gambar
- Google Colab — training model

## 👤 Muhammad Fadhli Hamdani

Proyek ini dikembangkan sebagai bagian dari tugas akhir mata kuliah di **Politeknik Citra Riau**.
