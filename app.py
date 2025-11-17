# BU KODUN TAMAMINI KOPYALAYIN VE app.py DOSYASINA YAPIŞTIRIN (v10.0 - Sadece 122 Ürün)

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# --- 0. OTURUM DURUMU (SESSION STATE) BAŞLATMA ---
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="LRN Koku Rehberi v10.0 (Stabil Stok Bazlı)",
    page_icon="👃",
    layout="wide"
)

# --- YARDIMCI GÜVENLİK FONKSİYONU ---
def safe_eval(text):
    """Eval komutunun hata vermesi durumunda boş listelerle başa çıkar."""
    try:
        # String listeyi gerçek listeye çevirip notaları birleştirir
        return ' '.join(eval(str(text))).lower()
    except:
        return ""

# --- 2. VERİ YÜKLEME VE MODEL OLUŞTURMA (SADECE 122 ÜRÜN) ---
@st.cache_resource
def load_data():
    print("Sadece 122 ürün yükleniyor ve model oluşturuluyor...")
    try:
        # 1. Stok Veritabanını Yükle
        df = pd.read_csv("stok_listesi_clean.csv")
        df = df.rename(columns={'orijinal_ad': 'isim'})
        
        # Notaları model için hazırla (metin tabanlı)
        df['notalar_str'] = df['notalar'].apply(safe_eval)
        
        # 2. Model Kurulumu (Sadece 122 ürüne göre)
        vectorizer = CountVectorizer(min_df=1)
        koku_matrix = vectorizer.fit_transform(df['notalar_str'])
        
        # 3. Benzerlik Matrisi Oluştur
        cosine_sim = cosine_similarity(koku_matrix, koku_matrix)
        
        print("Minimal Model hazırlandı. (Toplam: {} parfüm)".format(len(df)))
        
        return df, cosine_sim, vectorizer

    except FileNotFoundError as e:
        st.error(f"HATA: Gerekli 'stok_listesi_clean.csv' dosyası bulunamadı.")
        st.error("Lütfen dosyanın klasörde olduğundan emin olun.")
        st.stop()

# Veri ve Modeli Yükle
stok_df, cosine_sim_matrix, vectorizer = load_data()


# --- 3. YARDIMCI FONKSİYONLAR (KART GÖSTERİMİ) ---

def display_stok_card(parfum_serisi):
    """Stoktaki bir parfümü kart olarak gösterir (Buton Kaldırıldı)."""
    st.markdown(f"**{parfum_serisi['kod']}** ({parfum_serisi['isim']})")
    st.markdown(f"**Kategori:** {parfum_serisi['kategori']}")
    st.markdown(f"**Cinsiyet:** {parfum_serisi['cinsiyet']}")
    
    # Notaları göster (Görünüm için)
    try:
        not_listesi = eval(parfum_serisi['notalar'])
        st.caption(f"Ana Notalar: {', '.join(not_listesi[:4])}...")
    except:
         st.caption("Ana Notalar: Bilgi yok.")
    

# --- 4. BENZERLİK BULMA MOTORU (SADECE STOK BAZLI) ---

def find_similar(search_term, gender_filter="Tümü"):
    
    if search_term and search_term.lower() not in [h.lower() for h in st.session_state.search_history]:
        st.session_state.search_history.insert(0, search_term)
        st.session_state.search_history = st.session_state.search_history[:5]
    
    recommendations = []
    
    # 1. LRN Koduna veya Orijinal Adına Göre Ana Ürünü Bulma
    search_term_lower = search_term.lower()
    
    # İndeksleri isim/kod üzerinden bulma
    match = stok_df[
        (stok_df['kod'].astype(str) == search_term) | 
        (stok_df['isim'].str.contains(search_term, case=False, na=False))
    ]
    
    if not match.empty:
        found_perfume = match.iloc[0]
        perfume_index = found_perfume.name # Modeldeki indeksi
        
        # Kendisi hariç tüm benzerlik skorlarını al
        sim_scores = list(enumerate(cosine_sim_matrix[perfume_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Kendisini listelediğimiz için atla (sim_scores[1:])
        sim_scores_to_check = sim_scores[1:] 

        # En yakın 3 öneri
        count = 0
        for i, score in sim_scores_to_check:
            recommended_parfum = stok_df.iloc[i]
            
            if score > 0.0 and (gender_filter == "Tümü" or recommended_parfum['cinsiyet'] == gender_filter):
                recommendations.append(recommended_parfum)
                count += 1
            
            if count >= 3: # Sadece en yakın 3 koku
                break
        
        return found_perfume, recommendations # Ana ürün ve öneri listesi

    else:
        # 2. Nota/Hissiyat Araması
        st.warning(f"**'{search_term}'** adında bir parfüm veya kod bulunamadı. Nota/Hissiyat olarak arama yapılıyor...")
        
        try:
            # Arama terimini vektöre dönüştür
            search_vector = vectorizer.transform([search_term]) 
            
            # Tüm stok parfümlerine benzerliğini hesapla
            nota_sim_scores = cosine_similarity(search_vector, cosine_sim_matrix.T) 
            
            stock_scores = list(enumerate(nota_sim_scores[0]))
            stock_scores = sorted(stock_scores, key=lambda x: x[1], reverse=True)
            
            # En iyi 5 sonuçtan sadece ilk 3'ü
            count = 0
            for i, score in stock_scores:
                recommended_parfum = stok_df.iloc[i]
                
                if score > 0.0 and (gender_filter == "Tümü" or recommended_parfum['cinsiyet'] == gender_filter):
                    recommendations.append(recommended_parfum)
                    count += 1

                if count >= 3: # Sadece en yakın 3 koku
                    break
            
            return None, recommendations # Ana ürün yok, sadece öneriler

        except Exception:
            return None, []


# --- 5. KULLANICI ARAYÜZÜ ---

st.title("👃 LRN Koku Rehberi v10.0 (Stabil Stok Bazlı)")
st.markdown(f"**Toplam {len(stok_df)}** stoklu ürün. (70K veri devre dışı.)")

st.header("🌟 Stok Arama Motoru")
st.markdown("LRN Kodunu (örn: `255`), Orijinal Adı (`Creed Aventus`) veya Notayı (`vanilya`) girin.")

# --- Arama Formu ---
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    search_query = st.text_input("Arama Kutusu", placeholder="örn: 255 veya odunsu", key="main_search_query")
    
with col2:
    gender_choice = st.selectbox("Cinsiyet Filtresi", ["Tümü", "Kadın", "Erkek", "Unisex"], key="main_gender_filter")

with col3:
    if st.button("Geçmişi Temizle", help="Arama geçmişini temizler"):
        st.session_state.search_history = []
        st.rerun()

search_triggered = False
if st.session_state.search_history:
    with st.expander("Son Aramalarınız"):
        history_cols = st.columns(len(st.session_state.search_history))
        for i, query in enumerate(st.session_state.search_history):
            if history_cols[i].button(query, key=f"hist_{query}"):
                st.session_state.main_search_query = query
                search_triggered = True

# --- Arama Tetikleme ---
if st.button("Koku Bul", type="primary") or search_triggered:
    final_query = st.session_state.main_search_query
    
    if len(final_query) < 2 and not final_query.isdigit():
        st.warning("Lütfen en az 2 harf veya geçerli bir kod girin.")
    else:
        # ARAMA MOTORUNU ÇALIŞTIR
        main_product, recommended_parfumes = find_similar(final_query, st.session_state.main_gender_filter)
        
        st.divider()

        # 1. Ana Ürünü Listele (Varsa)
        if main_product is not None:
             st.subheader("Aranan Ürün:")
             with st.container(border=True):
                 display_stok_card(main_product)
             st.divider()

        # 2. Önerileri Listele
        if not recommended_parfumes:
            if main_product is None:
                st.error(f"'{final_query}' aramasıyla eşleşen bir ürün bulunamadı.")
            else:
                 st.info(f"'{main_product['isim']}' ürününe benzeyen başka ürün bulunamadı.")
        else:
            st.subheader("Size En Çok Benzeyen 3 Koku:")
            
            # Yan Yana Sütunlarda Gösterme Mantığı
            cols = st.columns(3) # 3 sütun oluştur
            
            for i, parfum_row in enumerate(recommended_parfumes):
                with cols[i % 3]:
                    with st.container(border=True):
                        display_stok_card(parfum_row)