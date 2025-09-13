import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import datetime
import re
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Aplikasi Sistem Prediksi Penyakit Jantung", layout="wide")

st.markdown("""
<style>
/* CSS Umum */
.box-wrapper {
    background-color: rgba(255, 255, 255, 0.05);
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    margin-bottom: 1.5rem;
}

.stButton>button {
    padding: 0.5rem 1.5rem;
    border-radius: 8px;
    font-weight: 600;
    background-color: #0066cc;
    color: white;
    border: none;
}

.stButton>button:hover {
    background-color: #004999;
}

h1, h2, h3, h4 {
    font-size: 1.6rem;
    color: #333333; /* Warna teks gelap untuk judul utama di halaman konten */
}

/* --- CSS Khusus untuk Tabel HTML Manual (Tabel Probabilitas) --- */
/* Pastikan warna background dan border sesuai tema gelap dan polos */
.centered-table {
    width: 100%;
    border-collapse: collapse;
    margin-left: auto;
    margin-right: auto;
    color: #F8F9FA !important; /* Warna teks default untuk isi tabel, pakai !important */
    background-color: transparent !important; /* Pastikan latar belakang tabel transparan */
}

.centered-table th,
.centered-table td {
    border: 1px solid #444444 !important; /* Border yang lebih gelap agar sesuai tema, pakai !important */
    padding: 8px !important;
    text-align: center !important;
    vertical-align: middle !important;
    background-color: transparent !important; /* Pastikan semua sel transparan */
    color: #F8F9FA !important; /* Pastikan teks di sel berwarna terang */
}

.centered-table th {
    /* Mengubah warna abu-abu agar lebih pudar (transparan) dan tidak solid */
    background-color: rgba(85, 85, 85, 0.5) !important; /* Abu-abu #555555 dengan transparansi 50% */
    color: #F8F9FA !important; /* Warna teks putih terang untuk header */
    font-weight: normal !important; /* Menghilangkan bold */
}

/* Hapus warna latar belakang bergantian untuk baris (jika ingin benar-benar polos) */
.centered-table tr:nth-child(even) {
    background-color: transparent !important;
}
.centered-table tr:nth-child(odd) {
    background-color: transparent !important;
}

/* Jika border ingin dihilangkan sepenuhnya, ubah border menjadi none */
/* .centered-table th, .centered-table td {
    border: none !important;
}
.centered-table {
    border: 1px solid #444444 !important; /* Jika ingin hanya border luar tabel */
/* } */


/* CSS untuk Sidebar */
section[data-testid="stSidebar"] > div:first-child {
    padding: 1.5rem;
    background-color: #2C353F;
    border-right: 1px solid #44546A;
    border-radius: 0 12px 12px 0;
    box-shadow: 2px 0 15px rgba(0, 0, 0, 0.3);
    display: flex;
    flex-direction: column;
    align-items: center;
}

section[data-testid="stSidebar"] .sidebar-title-container {
    display: flex;
    flex-direction: row; /* Membuat item di dalamnya sejajar horizontal */
    align-items: center; /* Pusatkan secara vertikal */
    justify-content: center; /* Pusatkan secara horizontal */
    margin-bottom: 25px;
    padding-bottom: 5px;
    border-bottom: 2px solid rgba(255, 255, 255, 0.15);
    width: 100%;
}

section[data-testid="stSidebar"] h3 {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: #F8F9FA;
    text-shadow: 0.5px 0.5px 1px rgba(0,0,0,0.2);
    letter-spacing: 0.5px;
    margin: 0;
    padding-right: 8px; /* Memberi sedikit ruang antara teks dan ikon */
}

section[data-testid="stSidebar"] .hint-icon-svg {
    width: 18px;
    height: 18px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23FFFFFF' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='3' y='3' width='18' height='18' rx='2' ry='2'%3E%3C/rect%3E%3Cline x1='12' y1='16' x2='12' y2='12'%3E%3C/line%3E%3Cline x1='12' y1='8' x2='12.01' y2='8'%3E%3C/line%3E%3C/svg%3E");
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    cursor: help;
    position: relative;
    top: 2px; /* Sesuaikan posisi vertikal ikon jika perlu */
}

section[data-testid="stSidebar"] .hint-icon-svg::before {
    content: 'Pilih halaman aplikasi';
    position: absolute;
    bottom: 120%;
    left: 50%;
    transform: translateX(-50%);
    background-color: #555;
    color: white;
    padding: 5px 8px;
    border-radius: 5px;
    font-size: 12px;
    white-space: nowrap;
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.2s ease, visibility 0.2s ease;
    z-index: 1000;
}

section[data-testid="stSidebar"] .hint-icon-svg:hover::before {
    opacity: 1;
    visibility: visible;
}

/* Menyembunyikan label "Pilih Halaman" dari st.radio */
section[data-testid="stSidebar"] div.stRadio > label:first-child {
    display: none !important;
}

/* Mengatur tata letak radio button group */
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    gap: 8px;
    padding: 0;
    width: 100%;
    margin-top: 15px;
}

/* Styling untuk setiap label radio button */
section[data-testid="stSidebar"] .stRadio > label {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    padding: 0.7rem 0.5rem;
    border-radius: 8px;
    cursor: pointer;
    background-color: transparent;
    color: #F8F9FA; /* Warna teks default untuk pilihan navigasi */
    font-weight: 500;
    transition: background-color 0.3s ease, color 0.3s ease;
    position: relative;
    min-height: 42px;
    width: 100%;
}

section[data-testid="stSidebar"] .stRadio > label:hover {
    background-color: rgba(255, 255, 255, 0.08);
    color: #FFFFFF;
}

/* Styling untuk pilihan radio button yang sedang aktif (selected) */
section[data-testid="stSidebar"] .stRadio > label[data-baseweb="radio"] {
    font-weight: 600;
    color: #FFFFFF; /* Teks putih solid untuk pilihan aktif */
    background-color: transparent; /* Pastikan latar belakang tetap transparan untuk pilihan aktif */
}

/* Menyembunyikan lingkaran radio button Streamlit default */
section[data-testid="stSidebar"] .stRadio > label > div[data-testid="stFlex"] > div:first-child {
    display: none !important;
}

/* Mengatur teks radio button agar rata kiri */
section[data-testid="stSidebar"] .stRadio > label > div[data-testid="stFlex"] > div:nth-child(2) {
    flex-grow: 1;
    text-align: left;
    margin-left: 0 !important;
    padding-left: 10px; /* Jarak antara lingkaran kustom dan teks */
    font-size: 16px;
}

/* Lingkaran kustom untuk pilihan radio button aktif */
section[data-testid="stSidebar"] .stRadio > label[data-baseweb="radio"]::before {
    content: '';
    display: block;
    width: 12px;
    height: 12px;
    background-color: #FF4B4B; /* Warna merah untuk aktif */
    border-radius: 50%;
    margin-right: 15px;
    border: 2px solid white; /* Border putih */
    box-shadow: 0 0 5px rgba(255, 75, 75, 0.7); /* Efek bayangan merah */
    flex-shrink: 0;
}

/* Lingkaran kustom untuk pilihan radio button tidak aktif */
section[data-testid="stSidebar"] .stRadio > label:not([data-baseweb="radio"])::before {
    content: '';
    display: block;
    width: 12px;
    height: 12px;
    background-color: #6C757D; /* Warna abu-abu untuk tidak aktif */
    border-radius: 50%;
    margin-right: 15px;
    border: 2px solid #555; /* Border abu-abu gelap */
    flex-shrink: 0;
}

/* Mengatur padding group radio button Streamlit */
section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > div {
    padding-left: 0px !important;
    padding-right: 0px !important;
}

</style>
""", unsafe_allow_html=True)

pages = ["Beranda", "Upload & Prediksi", "Visualisasi Sinyal", "Unduh Hasil"]

# Inisialisasi session_state untuk semua variabel yang diperlukan
if "page_index" not in st.session_state:
    st.session_state.page_index = 0
if "metadata_valid" not in st.session_state:
    st.session_state.metadata_valid = False
if "signal" not in st.session_state:
    st.session_state.signal = None
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "pred_label" not in st.session_state:
    st.session_state.pred_label = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "nama" not in st.session_state:
    st.session_state.nama = ""
if "usia" not in st.session_state:
    st.session_state.usia = 10 
if "jenis_kelamin" not in st.session_state:
    st.session_state.jenis_kelamin = ""


def sidebar_navigasi():
    with st.sidebar:
        st.markdown(f"""
            <div class="sidebar-title-container">
                <h3>Navigasi</h3>
                <div class="hint-icon-svg" title="Pilih halaman aplikasi"></div> 
            </div>
        """, unsafe_allow_html=True)
        
        current_selection_index = st.session_state.page_index
        
        selected_page_name = st.radio(
            "Pilih Halaman",
            pages,
            index=current_selection_index, 
            key="sidebar_radio_nav", 
            label_visibility="collapsed"
        )

        new_index = pages.index(selected_page_name)
        if new_index != current_selection_index:
            if selected_page_name == "Upload & Prediksi" and not st.session_state.metadata_valid:
                st.warning("Silakan isi dan simpan informasi pengguna terlebih dahulu di halaman Beranda.")
                st.session_state.page_index = current_selection_index 
            else:
                st.session_state.page_index = new_index
                st.rerun() 

sidebar_navigasi()
page = pages[st.session_state.page_index] 

def nav_buttons():
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.session_state.page_index > 0:
            if st.button("⬅ Sebelumnya"):
                st.session_state.page_index -= 1
                st.rerun()
    with col3:
        if st.session_state.page_index < len(pages) - 1:
            if st.button("Selanjutnya ➡"):
                if page == "Beranda" and not st.session_state.metadata_valid:
                    st.warning("Silakan isi dan simpan informasi pengguna terlebih dahulu.")
                else:
                    st.session_state.page_index += 1
                    st.rerun()

@st.cache_resource
def load_artifacts():
    model = load_model("model_fold_3.keras")
    scaler = joblib.load("standard_scaler.joblib")
    class_names = np.load("class_names.npy", allow_pickle=True)
    return model, scaler, class_names

model, scaler, class_names = load_artifacts()
warna_kotak = {"NORM": "#1E90FF", "STTC": "#DAA520", "MI": "#FF6347"}
warna_teks = {"NORM": "white", "STTC": "white", "MI": "white"}
deskripsi_kelas = {
    "NORM": "Hasil ini menunjukkan bahwa tidak ada kelainan signifikan pada sinyal EKG. Namun, hasil normal tidak sepenuhnya mengesampingkan kondisi jantung lain, dan sebaiknya tetap dikonsultasikan ke dokter untuk evaluasi lebih lanjut.",
    "STTC": "Perubahan pada segmen ST atau gelombang T dapat mengindikasikan iskemia (kurangnya suplai darah ke jantung), stres jantung, atau pengaruh obat-obatan tertentu. Perlu pemeriksaan lanjutan seperti EKG 12-lead atau tes laboratorium.",
    "MI": "Merupakan kondisi serius yang terjadi ketika aliran aliran darah ke bagian otot jantung tersumbat, biasanya akibat pembekuan darah. Gejala bisa berupa nyeri dada, sesak napas, dan kelelahan ekstrem. Deteksi dini sangat penting untuk mencegah mencegah kerusakan permanen pada jantung."
}

if page == "Beranda":
    st.markdown("""<h1 style='font-size: 46px;'>Aplikasi Sistem Prediksi Penyakit Jantung</h1>""", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size: 18px; margin-top: -15px; margin-bottom: 25px;'>
        Aplikasi ini dirancang untuk membantu analisis awal sinyal EKG mendeteksi potensi penyakit jantung secara cepat dan efisien menggunakan model 1D CNN BiLSTM.
    </p>
    <div class='box-wrapper'>
        <h4 style='font-size: 22px;'>Panduan Penggunaan Aplikasi</h4>
        <ol style='font-size: 18px;'>
            <li>Isi informasi pengguna terlebih dahulu di formulir di bawah.</li>
            <li>Masuk ke menu <strong>Upload & Prediksi</strong>.</li>
            <li>Unggah file sinyal EKG yang berdimensi (1000, 12) dalam format .npy</li>
            <li>Lihat hasil prediksi serta visualisasi sinyal EKG.</li>
            <li>Unduh hasil prediksi jika diperlukan.</li>
        </ol>
        <p style='font-size: 16px; color: gray;'>
            <em>Catatan: Aplikasi ini hanya berfungsi sebagai alat bantu prediksi dan tidak menggantikan diagnosis medis profesional.</em>
        </p>
    </div>
    <div class='box-wrapper'>
        <h4>Formulir Identitas Pengguna</h4>
    """, unsafe_allow_html=True)

    with st.form("form_metadata"):
        nama = st.text_input("Nama Pasien", value=st.session_state.get("nama", ""))
        usia = st.number_input("Usia", value=st.session_state.get("usia", 10)) 
        jk = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        simpan = st.form_submit_button("Simpan Informasi")

        if simpan:
            is_valid = True
            
            if not nama.strip(): 
                st.warning("Nama pengguna tidak boleh kosong.")
                is_valid = False
            elif not re.fullmatch(r"^[a-zA-Z\s'-]+$", nama):
                st.warning("Nama hanya boleh terdiri dari huruf dan spasi.")
                is_valid = False
            
            if usia < 6 or usia > 120:
                st.warning("Usia harus antara 6 hingga 120 tahun.")
                is_valid = False
            elif usia <= 0:
                st.warning("Usia harus merupakan angka positif.")
                is_valid = False

            if is_valid:
                st.session_state.nama = nama.strip()
                st.session_state.usia = usia
                st.session_state.jenis_kelamin = jk
                st.session_state.metadata_valid = True
                st.success("Data Anda telah disimpan.")
            else:
                st.session_state.metadata_valid = False 

    st.markdown("</div>", unsafe_allow_html=True)
    nav_buttons()

elif page == "Upload & Prediksi":
    st.title("Upload Sinyal EKG dan Prediksi")

    if not st.session_state.metadata_valid:
        st.warning("Silakan isi informasi pasien terlebih dahulu di halaman Beranda sebelum mengunggah file.")
        nav_buttons()
        st.stop()
    
    uploaded_file = st.file_uploader(
        "Unggah file sinyal EKG berdimensi (1000,12) dalam format (.npy)",
        type="npy"
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Memproses sinyal dan melakukan prediksi..."):
                signal = np.load(uploaded_file)
                if signal.shape != (1000, 12):
                    st.error(f"Dimensi sinyal tidak sesuai. Diharapkan (1000, 12), namun terdeteksi {signal.shape}. Harap unggah file .npy yang valid.")
                    st.session_state.signal = None 
                    st.session_state.prediction = None
                    st.session_state.pred_label = None
                    st.session_state.confidence = None
                else:
                    signal_scaled = scaler.transform(signal)
                    signal_ready = signal_scaled.reshape(1, 1000, 12)
                    prediction = model.predict(signal_ready)
                    pred_label = class_names[np.argmax(prediction)]
                    confidence = np.max(prediction)

                    st.session_state.signal = signal
                    st.session_state.prediction = prediction
                    st.session_state.pred_label = pred_label
                    st.session_state.confidence = confidence

                    st.markdown("---")
                    st.subheader("Informasi Pengguna dan Hasil Prediksi")
                    with st.container(border=True): 
                        st.markdown(f"<p style='font-size: 1.15rem; margin-bottom: 0.5rem;'><b>Nama Pasien:</b> {st.session_state.nama if st.session_state.nama else '-'}</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size: 1.15rem; margin-bottom: 0.5rem;'><b>Usia:</b> {st.session_state.usia if st.session_state.usia else '-'} tahun</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size: 1.15rem; margin-bottom: 0.5rem;'><b>Jenis Kelamin:</b> {st.session_state.jenis_kelamin if st.session_state.jenis_kelamin else '-'}</p>", unsafe_allow_html=True)
                    
                        st.markdown(f"""
                        <div style='background-color: {warna_kotak[pred_label]}; padding: 10px 18px; border-radius: 10px; margin: 20px 0;'>
                            <h4 style='color: {warna_teks[pred_label]}; margin: 0; font-size: 1.4rem;'>Model memprediksi: {pred_label}</h4>
                        </div>
                        """, unsafe_allow_html=True)

                        st.metric("Tingkat Keyakinan", f"{confidence:.2%}")
                        
                        with st.expander("Penjelasan Prediksi Model"):
                            st.markdown(f"<p style='font-size: 1.1rem;'>{deskripsi_kelas.get(pred_label, 'Tidak ada penjelasan untuk hasil ini.')}</p>", unsafe_allow_html=True)

                        st.subheader("Distribusi Probabilitas")
                        df_probs = pd.DataFrame({"Kelas": class_names, "Probabilitas": prediction[0]})
                        st.bar_chart(df_probs.set_index("Kelas"))

                        st.subheader("Tabel Probabilitas")
                        df_probs_display = pd.DataFrame({
                            "Kelas": class_names,
                            "Probabilitas": [f"{p:.2%}" for p in prediction[0]]
                        })
                        
                        df_probs_display = df_probs_display.sort_values("Probabilitas", ascending=False)
                        
                        html_table = df_probs_display.to_html(index=False, classes='centered-table')
                        
                        st.markdown(html_table, unsafe_allow_html=True)

        except ValueError as ve:
            st.error(f"Gagal memuat file. File .npy mungkin rusak atau kosong: {ve}. Harap unggah file .npy yang valid.")
            st.session_state.signal = None 
            st.session_state.prediction = None
            st.session_state.pred_label = None
            st.session_state.confidence = None
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}. Pastikan file berformat .npy yang valid dan dimensinya sesuai.")
            st.session_state.signal = None 
            st.session_state.prediction = None
            st.session_state.pred_label = None
            st.session_state.confidence = None
    
    nav_buttons()

elif page == "Visualisasi Sinyal":
    st.title("Visualisasi Sinyal EKG")
    st.write("Visualisasi ini menampilkan sinyal EKG untuk masing-masing lead guna membantu pemahaman terhadap karakteristik sinyal yang diproses oleh model.")

    if st.session_state.signal is None:
        st.warning("Belum ada sinyal. Silakan unggah file terlebih dahulu.")
    else:
        lead_index = st.selectbox("Pilih Lead", options=list(range(1, 13))) - 1
        st.subheader(f"Visualisasi Lead {lead_index+1}")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(st.session_state.signal[:, lead_index], color='blue')
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("""
        <p style='font-size: 1.1rem;'>
        Grafik di atas menunjukkan sinyal listrik jantung dari Lead yang dipilih. 
        <ul>
            <li><b>Sumbu Horizontal (Waktu/Sampel):</b> Merepresentasikan 1000 titik sampel data, yang umumnya setara dengan 10 detik rekaman sinyal EKG.</li>
            <li><b>Sumbu Vertikal (Amplitudo):</b> Menunjukkan kekuatan sinyal listrik jantung dalam milivolt (mV).</li>
        </ul>
        Pola naik-turun pada grafik ini mencerminkan aktivitas listrik pada setiap detak jantung.
        </p>
        """, unsafe_allow_html=True)

        st.markdown("---") # Garis pemisah untuk keterbacaan

        st.subheader("Visualisasi 12 Lead")
        fig, axs = plt.subplots(4, 3, figsize=(15, 8))
        for i in range(12):
            row, col = divmod(i, 3)
            axs[row, col].plot(st.session_state.signal[:, i], color='blue')
            axs[row, col].set_title(f"Lead {i+1}")
            axs[row, col].grid(True)
        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("Penjelasan 12 Titik Rekaman Sinyal EKG"):
            st.markdown("""
            <p style='font-size: 1.1rem;'>
            Sinyal EKG direkam dari 12 titik berbeda di tubuh untuk melihat kerja jantung dari berbagai sisi:
            </p>
            <ul style='font-size: 1.1rem;'>
            <li><b>Sinyal 1, 2, dan 3</b>: Dipasang di tangan dan kaki. Memberikan gambaran umum tentang jantung dari arah samping dan bawah.  </li>
            <li><b>Sinyal 4, 5, dan 6</b>: Dipasang juga di tangan dan kaki, tapi melihat dari arah berbeda seperti atas dan samping lainnya.  </li>
            <li><b>Sinyal 7 dan 8</b>: Ditempel di bagian tengah dada, melihat bagian depan jantung.  </li>
            <li><b>Sinyal 9 dan 10</b>: Ditempel agak ke kiri di dada, melihat bagian tengah jantung.  </li>
            <li><b>Sinyal 11 dan 12</b>: Ditempel lebih ke kiri dekat sisi tubuh, melihat bagian samping jantung.  </li>
            </ul>
            <p style='font-size: 1.1rem;'>
            Dengan melihat dari 12 arah ini, sistem dapat mengenali lebih jelas bagaimana jantung bekerja, apakah normal atau ada gangguan.
            </p>
            """, unsafe_allow_html=True)

    nav_buttons()

elif page == "Unduh Hasil":
    st.title("Unduh Hasil Prediksi")

    if st.session_state.pred_label is None:
        st.info("Belum ada hasil yang dapat diunduh.")
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        hasil_teks = f"""
Hasil Prediksi Deteksi Penyakit Jantung
----------------------------------------
Nama Pasien      : {st.session_state.nama}
Usia             : {st.session_state.usia} tahun
Jenis Kelamin    : {st.session_state.jenis_kelamin}
Waktu Akses      : {timestamp}

Prediksi         : {st.session_state.pred_label}
Confidence       : {st.session_state.confidence:.2%}

Penjelasan       : {deskripsi_kelas.get(st.session_state.pred_label, '')}
"""
        st.download_button(
            label="Unduh Hasil Prediksi (.txt)",
            data=hasil_teks,
            file_name=f"hasil_prediksi_{st.session_state.nama.replace(' ', '_')}.txt",
            mime="text/plain"
        )

    nav_buttons()