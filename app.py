import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import torch
import io

# 🛠️ SOLUSI ERROR KOTAK MERAH: Mengizinkan PyTorch membaca model YOLO
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception:
    pass

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Car Damage Detection AI",
    page_icon="🚗",
    layout="wide"
)

# =========================================================
# CUSTOM CSS — tampilan lebih modern
# =========================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 1rem 1rem 1rem;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FF512F, #DD2476);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #6b7280;
        font-size: 1.05rem;
    }
    .upload-card, .result-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
    }
    .damage-badge {
        display: inline-block;
        background: linear-gradient(90deg, #FF512F, #DD2476);
        color: white;
        padding: 6px 14px;
        border-radius: 999px;
        margin: 4px 6px 4px 0;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #FF512F, #DD2476);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.4rem;
        font-weight: 700;
        font-size: 1rem;
        width: 100%;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🚗 AI Deteksi Kerusakan Bodi Mobil</h1>
    <p>Unggah foto mobil, AI akan otomatis mendeteksi lecet, penyok, dan kaca retak — tanpa pengaturan rumit.</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

with st.spinner("Memuat model AI..."):
    try:
        model = load_model()
        model_loaded = True
    except Exception as e:
        model_loaded = False
        st.error(f"❌ Gagal memuat model. Error: {e}")

# Status singkat di sidebar (tanpa pengaturan teknis untuk user)
with st.sidebar:
    st.header("ℹ️ Status Sistem")
    if model_loaded:
        st.success("Otak AI (best.pt) aktif dan siap digunakan ✅")
    st.caption(
        "Tingkat sensitivitas dideteksi secara **otomatis** oleh sistem "
        "berdasarkan kondisi tiap gambar, jadi kamu tidak perlu mengatur apa pun."
    )
    with st.expander("⚙️ Mode Lanjutan (opsional)"):
        manual_override = st.checkbox("Atur sensitivitas secara manual")
        manual_conf = st.slider("Confidence manual", 0.05, 1.00, 0.25, 0.05) if manual_override else None

# =========================================================
# FUNGSI DETEKSI OTOMATIS (ADAPTIVE CONFIDENCE)
# =========================================================
def auto_detect(model, image_array, thresholds=(0.55, 0.45, 0.35, 0.25, 0.15, 0.08)):
    """
    Mencoba beberapa ambang confidence dari yang paling ketat ke paling longgar.
    Berhenti begitu kerusakan pertama berhasil terdeteksi, sehingga user
    tidak perlu mengatur sensitivitas secara manual.
    """
    last_result = None
    used_conf = thresholds[-1]
    for conf in thresholds:
        results = model.predict(source=image_array, conf=conf, verbose=False)
        last_result = results[0]
        used_conf = conf
        if len(last_result.boxes) > 0:
            break
    return last_result, used_conf

# =========================================================
# HELPER: Batasi tinggi gambar agar tidak memenuhi halaman
# =========================================================
def cap_image_height(img, max_height=450):
    """Resize gambar secara proporsional jika tingginya melebihi max_height."""
    w, h = img.size
    if h > max_height:
        ratio = max_height / h
        new_w = int(w * ratio)
        img = img.resize((new_w, max_height), Image.LANCZOS)
    return img

# =========================================================
# UPLOAD & DETEKSI
# =========================================================
uploaded_file = st.file_uploader(
    "📤 Pilih dan unggah foto mobil rusak...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model_loaded:
    source_image = Image.open(uploaded_file).convert("RGB")
    display_image = cap_image_height(source_image)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="upload-card">', unsafe_allow_html=True)
        st.image(display_image, caption="Foto Mobil Asli", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    detect_clicked = st.button("🚀 Deteksi Kerusakan Sekarang")

    if detect_clicked:
        with st.spinner("AI sedang menganalisis gambar..."):
            input_image = np.array(source_image)

            if manual_override and manual_conf is not None:
                results = model.predict(source=input_image, conf=manual_conf, verbose=False)
                result = results[0]
                used_conf = manual_conf
            else:
                result, used_conf = auto_detect(model, input_image)

            annotated_img_bgr = result.plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)

            with col2:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                annotated_pil = cap_image_height(Image.fromarray(annotated_img_rgb))
                st.image(annotated_pil, caption="Hasil Deteksi AI", use_column_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("---")
            boxes = result.boxes

            if len(boxes) > 0:
                st.success(f"🎯 Berhasil mendeteksi **{len(boxes)}** titik kerusakan pada mobil!")
                st.caption(f"Sensitivitas otomatis yang digunakan sistem: {used_conf:.2f}")

                class_names = model.names
                detected_classes = [class_names[int(box.cls)] for box in boxes]

                st.write("**Rincian Kerusakan Terdeteksi:**")
                badges_html = ""
                for cls in set(detected_classes):
                    count = detected_classes.count(cls)
                    badges_html += f'<span class="damage-badge">🔴 {cls}: {count} titik</span>'
                st.markdown(badges_html, unsafe_allow_html=True)

                # Tombol unduh hasil
                result_pil = Image.fromarray(annotated_img_rgb)
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                st.download_button(
                    "⬇️ Unduh Gambar Hasil Deteksi",
                    data=buf.getvalue(),
                    file_name="hasil_deteksi_kerusakan.png",
                    mime="image/png"
                )
            else:
                st.warning(
                    "ℹ️ AI tidak menemukan kerusakan yang jelas pada gambar ini, "
                    "bahkan setelah mencoba beberapa tingkat sensitivitas secara otomatis. "
                    "Coba unggah foto dengan kerusakan yang lebih terlihat jelas."
                )

elif uploaded_file is not None and not model_loaded:
    st.error("Model belum berhasil dimuat, deteksi tidak dapat dilakukan.")
else:
    st.info("👆 Silakan unggah foto mobil untuk memulai analisis.")