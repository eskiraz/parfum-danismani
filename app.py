# BU KODUN TAMAMINI KOPYALAYIN VE app.py DOSYASINA YAPIŞTIRIN (v10.3 - Final UX ve Arama Fix'i)

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# --- 0. SABİTLER ve OTURUM DURUMU ---
GENDER_ICONS = {
    "Erkek": "♂️",
    "Kadın": "♀️",
    "Unisex": "🚻",
    "Niche": "💎" # Yeni Niche ikonu
}

# Son arama sonucunu tutmak için yeni bir durum eklenir (Enter fix için)
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'last_search_query' not in st.session_state:
    st.session_state.last_search_query = ""


# --- YARDIMCI GÜVENLİK FONKSİYONU ---
def safe_eval(text):
    """Eval komutunun hata vermesi durumunda boş listelerle başa çıkar."""
    try:
        return ' '.join(eval(str(text))).lower()
    except:
        return ""

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="LRN Koku Rehberi v10.3 (Final)",
    page_icon="👃",
    layout="wide"
)

# --- 2. VERİ YÜKLEME VE MODEL OLUŞTURMA (SADECE 122 ÜRÜN) ---
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("stok_listesi_clean.csv")
        df = df.rename(columns={'orijinal_ad': 'isim'})
        
        # Notaları ve Kategori adlarını birleştir (Arama hassasiyeti için)
        df['notalar_str'] = df['notalar'].apply(safe_eval) + ' ' + df['kategori'].str.lower()
        
        # Model Kurulumu (Sadece 122 ürüne göre)
        vectorizer = CountVectorizer(min_df=1)
        koku_matrix = vectorizer.fit_transform(df['notalar_str'])
        cosine_sim = cosine_similarity(koku_matrix, koku_matrix)
        
        return df, cosine_sim, vectorizer

    except FileNotFoundError as e:
        st.error(f"HATA: Gerekli 'stok_listesi_clean.csv' dosyası bulunamadı.")
        st.stop()

stok_df, cosine_sim_matrix, vectorizer = load_data()


# --- 3. YARDIMCI FONKSİYONLAR (KART GÖSTERİMİ) ---

def display_stok_card(parfum_serisi):
    """Stoktaki bir parfümü kart olarak gösterir (Niche Logiği Eklendi)."""
    
    gender_icon = GENDER_ICONS.get(parfum_serisi['cinsiyet'], "🚻")

    # Niche Logic: LRN kodu 200 veya altıysa Niche ikonu ekle
    try:
        lrn_code = int(parfum_serisi['kod'])
        niche_icon = GENDER_ICONS.get("Niche") if lrn_code <= 200 else ""
    except ValueError:
        niche_icon = "" 

    
    # İkonlar ve kod yan yana
    st.markdown(f"**{niche_icon} {gender_icon} {parfum_serisi['kod']}** ({parfum_serisi['isim']})")
    st.markdown(f"**Kategori:** {parfum_serisi['kategori']}")
    
    # Notaları göster (Görünüm için)
    try:
        not_listesi = eval(parfum_serisi['notalar'])
        st.caption(f"Ana Notalar: {', '.join(not_listesi[:4])}...")
    except:
         st.caption("Ana Notalar: Bilgi yok.")
    

# --- 4. BENZERLİK BULMA MOTORU (SADECE STOK BAZLI) ---

def find_similar(search_term):
    # Geçmişe Kaydetme
    if search_term and search_term.lower() not in [h.lower() for h in st.session_state.search_history]:
        st.session_state.search_history.insert(0, search_term)
        st.session_state.search_history = st.session_state.search_history[:5]
    
    recommendations = []
    search_term_lower = search_term.lower()
    
    # 1. LRN Koduna veya Orijinal Adına Göre Ana Ürünü Bulma
    match = stok_df[
        (stok_df['kod'].astype(str) == search_term) | 
        (stok_df['isim'].str.contains(search_term, case=False, na=False))
    ]
    
    if not match.empty:
        found_perfume = match.iloc[0]
        perfume_index = found_perfume.name
        
        # Kendisi hariç tüm benzerlik skorlarını al ve önerileri bul
        sim_scores = sorted(list(enumerate(cosine_sim_matrix[perfume_index])), key=lambda x: x[1], reverse=True)
        sim_scores_to_check = sim_scores[1:] 

        count = 0
        for i, score in sim_scores_to_check:
            recommended_parfum = stok_df.iloc[i]
            if score > 0.0:
                recommendations.append(recommended_parfum)
                count += 1
            if count >= 3: 
                break
        
        return found_perfume, recommendations

    else:
        # 2. Nota/Hissiyat veya Kategori Araması (örn: Çiçeksi)
        try:
            search_vector = vectorizer.transform([search_term]) 
            nota_sim_scores = cosine_similarity(search_vector, cosine_sim_matrix.T) 
            
            stock_scores = sorted(list(enumerate(nota_sim_scores[0])), key=lambda x: x[1], reverse=True)
            
            count = 0
            for i, score in stock_scores:
                recommended_parfum = stok_df.iloc[i]
                
                if score > 0.0:
                    recommendations.append(recommended_parfum)
                    count += 1

                if count >= 3:
                    break
            
            return None, recommendations

        except Exception:
            return None, []


# --- 5. KULLANICI ARAYÜZÜ ---

st.title("👃 LRN Koku Rehberi v10.3 (Final)")
st.markdown(f"**Toplam {len(stok_df)}** stoklu ürün. (En yakın 3 kokuyu önerir.)")

st.header("🌟 Stok Arama Motoru")
st.markdown("LRN Kodunu (`255`), Orijinal Adı (`Creed Aventus`) veya Notayı (`vanilya`, `çiçeksi`) girin.")

# --- Arama Formu ---
col1, col2 = st.columns([3, 1])

with col1:
    # Arama butonu kalktı, input'un kendisi tetikleme mekanizmasının parçası
    search_query = st.text_input("Arama Kutusu", placeholder="örn: 255 veya çiçeksi", key="main_search_query")
    
with col2:
    if st.button("Geçmişi Temizle", help="Arama geçmişini temizler", use_container_width=True):
        st.session_state.search_history = []
        st.session_state.last_search_query = ""
        st.rerun()

search_triggered = False
if st.session_state.search_history:
    with st.expander("Son Aramalarınız"):
        history_cols = st.columns(len(st.session_state.search_history))
        for i, query in enumerate(st.session_state.search_history):
            if history_cols[i].button(query, key=f"hist_{query}"):
                st.session_state.main_search_query = query
                search_triggered = True

# --- Arama Tetikleme (Enter/Buton/Geçmiş Hepsini Kapsar) ---
final_query = st.session_state.main_search_query
button_pressed = st.button("Koku Bul", type="primary")

# Yalnızca sorgu değiştiyse VEYA butona basıldıysa çalıştır.
if final_query and (button_pressed or search_triggered or final_query != st.session_state.last_search_query):
    
    if len(final_query) < 2 and not final_query.isdigit():
        st.warning("Lütfen en az 2 harf veya geçerli bir kod girin.")
    else:
        # ARAMA MOTORUNU ÇALIŞTIR
        main_product, recommended_parfumes = find_similar(final_query)
        st.session_state.last_search_query = final_query # Son sorguyu kaydet

        st.divider()

        # 1. Ana Ürünü Listele (Varsa)
        if main_product is not None:
             st.subheader(f"Aranan Ürün: {main_product['isim']}")
             with st.container(border=True):
                 display_stok_card(main_product)
             st.divider()

        # 2. Önerileri Listele (Yan Yana Görüntü)
        if not recommended_parfumes:
            if main_product is None:
                st.error(f"'{final_query}' aramasıyla eşleşen bir ürün bulunamadı.")
            else:
                 st.info(f"'{main_product['isim']}' ürününe benzeyen başka ürün bulunamadı.")
        else:
            st.subheader("Size En Çok Benzeyen 3 Koku:")
            
            cols = st.columns(3) # 3 sütun oluştur
            
            for i, parfum_row in enumerate(recommended_parfumes):
                with cols[i % 3]:
                    with st.container(border=True):
                        display_stok_card(parfum_row)