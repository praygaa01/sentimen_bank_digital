import streamlit as st
import pandas as pd
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from google_play_scraper import app

# --- KONSTANTA & INISIALISASI AWAL ---

# Daftar aplikasi yang akan dianalisis dan di-scrape
APPS_INFO = {
    'Jenius (BTPN)': 'com.btpn.dc',
    'blu by BCA Digital': 'com.bcadigital.blu',
    'SeaBank': 'id.co.bankbkemobile.digitalbank',
    'NeoBank': 'com.bnc.finance',
    'Bank Jago': 'com.jago.digitalBanking'
}


# Inisialisasi Sastrawi (dijalankan sekali)
factory_stemmer = StemmerFactory()
stemmer = factory_stemmer.create_stemmer()
factory_stopword = StopWordRemoverFactory()
stopword_remover = factory_stopword.create_stop_word_remover()


# --- FUNGSI-FUNGSI ---

def preprocess_text(text):
    """Fungsi untuk membersihkan dan memproses teks input."""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

@st.cache_resource
def load_model_and_vectorizer():
    """Memuat model dan vectorizer dari file .pkl dengan cache."""
    with open('model_svm.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

@st.cache_data(ttl=3600) # Cache data peringkat selama 1 jam
def get_app_rankings():
    """Fungsi untuk scrape data peringkat dan mengembalikannya sebagai DataFrame."""
    all_app_summary_data = []
    for id_app in APPS_INFO.values():
        try:
            result = app(id_app, lang='id', country='id')
            all_app_summary_data.append(result)
        except Exception as e:
            print(f"Gagal mengambil data untuk {id_app}: {e}")
            
    if not all_app_summary_data:
        return pd.DataFrame()

    df_summary = pd.DataFrame(all_app_summary_data)
    df_summary.dropna(subset=['score'], inplace=True)
    df_sorted_summary = df_summary.sort_values(by=['score', 'ratings'], ascending=[False, False]).reset_index(drop=True)
    df_sorted_summary['Ranking'] = df_sorted_summary.index + 1
    return df_sorted_summary

# --- MEMUAT MODEL ---
model, vectorizer = load_model_and_vectorizer()


# --- TAMPILAN APLIKASI STREAMLIT ---
st.set_page_config(page_title="Analisis Aplikasi Investasi", page_icon="📈", layout="wide")
st.title("📈 Analisis Aplikasi Investasi di Play Store")

# Buat tiga tab untuk memisahkan fungsionalitas
tab1, tab2, tab3 = st.tabs(["🔎 Analisis Sentimen", "🏆 Peringkat Aplikasi", "📊 Detail Data Latih"])

# --- KONTEN TAB 1: ANALISIS SENTIMEN ---
with tab1:
    st.header("Analisis Sentimen Ulasan")
    st.markdown("Masukkan satu atau beberapa ulasan (pisahkan dengan baris baru) untuk dianalisis oleh model **Support Vector Machine (SVM)**.")
    user_input = st.text_area("Masukkan teks ulasan di sini:", "aplikasi yang dipakai lancar dan mudah digunakan\nuang saya hilang", height=150)

    if st.button("Analisis Sentimen", use_container_width=True, type="primary"):
        if user_input:
            comments = [line.strip() for line in user_input.split('\n') if line.strip()]
            if comments:
                results = []
                for comment in comments:
                    processed_text = preprocess_text(comment)
                    vectorized_input = vectorizer.transform([processed_text])
                    prediction = model.predict(vectorized_input)
                    sentiment_map = {0: 'Negatif 👎', 1: 'Netral 😐', 2: 'Positif 👍'}
                    result_text = sentiment_map.get(prediction[0], 'Tidak diketahui')
                    results.append({'Komentar': comment, 'Prediksi Sentimen': result_text})

                df_results = pd.DataFrame(results)
                sentiment_counts = df_results['Prediksi Sentimen'].value_counts()
                
                st.subheader("📊 Ringkasan Hasil Analisis")
                summary_list = [f"{count} {label.split(' ')[0]}" for label, count in sentiment_counts.items()]
                summary_text = ", ".join(summary_list)
                st.success(f"**Hasil: {summary_text}**")
                
                st.subheader("Detail Analisis per Komentar")
                st.dataframe(df_results, use_container_width=True, hide_index=True)
            else:
                st.warning("Mohon masukkan setidaknya satu komentar yang valid.")
        else:
            st.warning("Area input masih kosong. Mohon masukkan teks terlebih dahulu.")

# --- KONTEN TAB 2: PERINGKAT APLIKASI ---
with tab2:
    st.header("Peringkat Aplikasi Berdasarkan Data Play Store")
    with st.spinner("Mengambil data peringkat terbaru dari Play Store..."):
        df_ranked = get_app_rankings()

    if not df_ranked.empty:
        st.dataframe(
            df_ranked[['Ranking', 'title', 'score', 'ratings', 'installs', 'developer']],
            hide_index=True,
            use_container_width=True
        )
    else:
        st.error("Gagal mengambil data peringkat aplikasi saat ini.")

# --- KONTEN TAB 3: DETAIL DATA LATIH ---
with tab3:
    st.header("Detail Data Latih yang Digunakan Model")
    try:
        df_latih = pd.read_csv('ulasan_kuota_kustom.csv')
        st.dataframe(df_latih)
        
        st.subheader("Distribusi Sentimen pada Data Latih")
        sentimen_counts = df_latih['sentimen'].value_counts()
        st.bar_chart(sentimen_counts)
    except FileNotFoundError:
        st.error("File 'ulasan_kuota_kustom.csv' tidak ditemukan di repository. Detail data latih tidak dapat ditampilkan.")
    except KeyError:
        st.error("Kolom 'sentimen' tidak ditemukan di file CSV. Tidak dapat menampilkan distribusi sentimen.")
