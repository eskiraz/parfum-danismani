# BU KODUN TAMAMINI KOPYALAYIN VE app.py DOSYASINA YAPIŞTIRIN (v10.19 - Final Logo Fix'i)

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import os # Yeni eklendi: Dosya yolunu daha esnek aramak için

# --- 0. SABİTLER ve OTURUM DURUMU ---
IMAGE_SIZE = 25 
ICON_MAPPING = {
    "Niche": "resimler/niche.jpg", 
    "Erkek": "resimler/erkek.jpg",
    "Kadın": "resimler/kadin.jpg",
    "Unisex": "resimler/unisex.jpg" 
}
# Logoyu doğrudan aramak yerine, dosya sistemini kullanarak bulmaya çalışacağız.
LOGOS_FOLDER = "resimler" 
APP_VERSION = "v10.19" 

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
    page_title="LRN Koku Rehberi",
    page_icon="👃",
    layout="wide"
)

# --- 2. VERİ YÜKLEME VE MODEL OLUŞTURMA (SADECE 122 ÜRÜN) ---
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("stok_listesi_clean.csv")
        df = df.rename(columns={'orijinal_ad': 'isim'})
        
        df['notalar_str'] = df['notalar'].apply(safe_eval) + ' ' + df['kategori'].str.lower()
        
        vectorizer = CountVectorizer(min_df=1)
        koku_matrix = vectorizer.fit_transform(df['notalar_str'])
        cosine_sim = cosine_similarity(koku_matrix, koku_matrix)
        
        return df, cosine_sim, vectorizer

    except FileNotFoundError as e:
        st.error(f"HATA: Gerekli 'stok_listesi_clean.csv' dosyası bulunamadı.")
        st.stop()

stok_df, cosine_sim_matrix, vectorizer = load_data()


# --- 3. YARDIMCI FONKSİYONLAR (KART GÖSTERİMİ) ---

def get_icon_path(parfum_serisi):
    
    try:
        lrn_code = int(parfum_serisi['kod'])
        if lrn_code <= 200:
            return ICON_MAPPING["Niche"]
    except ValueError:
        pass 
    
    gender = parfum_serisi['cinsiyet']
    return ICON_MAPPING.get(gender, ICON_MAPPING["Unisex"])


def display_stok_card(parfum_serisi):
    
    icon_path = get_icon_path(parfum_serisi)
    
    col_icon, col_text = st.columns([1, 6])
    
    with col_icon:
        try:
            st.image(icon_path, width=IMAGE_SIZE)
        except Exception:
             st.markdown("👃") 

    with col_text:
        st.markdown(f"**{parfum_serisi['kod']}** ({parfum_serisi['isim']})")
        st.markdown(f"**Kategori:** {parfum_serisi['kategori']}")
    
    try:
        not_listesi = eval(parfum_serisi['notalar'])
        st.caption(f"Ana Notalar: {', '.join(not_listesi)}")
    except:
         st.caption("Ana Notalar: Bilgi yok.")
    

# --- 4. BENZERLİK BULMA MOTORU (SADECE STOK BAZLI) ---

def find_similar(search_term):
    
    recommendations_list = []
    search_term_lower = search_term.lower()
    
    # 1. LRN Koduna veya Orijinal Adına Göre Ana Ürünü Bulma (Kesin Eşleşme Aranır)
    match = stok_df[
        (stok_df['kod'].astype(str) == search_term) | 
        (stok_df['isim'].str.contains(search_term, case=False, na=False))
    ]
    
    if not match.empty:
        # Kod/İsim bulunduysa, ML model ile benzerlerini öner (TOP 3)
        found_perfume = match.iloc[0]
        perfume_index = found_perfume.name
        
        sim_scores = sorted(list(enumerate(cosine_sim_matrix[perfume_index])), key=lambda x: x[1], reverse=True)
        sim_scores_to_check = sim_scores[1:] 

        count = 0
        for i, score in sim_scores_to_check:
            if score > 0.0:
                recommended_parfum = stok_df.iloc[i]
                recommendations_list.append(recommended_parfum)
                count += 1
            if count >= 3: 
                break
        
        return found_perfume, pd.DataFrame(recommendations_list)

    else:
        # 2. Nota/Hissiyat veya Kategori Araması (Eşleşen TÜMÜNÜ gösterir)
        st.warning(f"**'{search_term}'** adında bir ürün veya kod bulunamadı. Nota/Kategori araması yapılıyor...")
        
        try:
            # Metin araması yapılır (Garanti sonuç)
            results_df = stok_df[
                stok_df['notalar_str'].str.contains(search_term_lower, case=False, na=False) |
                stok_df['kategori'].str.contains(search_term_lower, case=False, na=False)
            ]
            
            return None, results_df

        except Exception:
            return None, pd.DataFrame()

# --- Logo Bulma Fonksiyonu ---
def find_lorinna_logo():
    """resimler klasöründeki lorinna logosunu uzantıdan bağımsız bulur."""
    if not os.path.isdir(LOGOS_FOLDER):
        return None
    
    for filename in os.listdir(LOGOS_FOLDER):
        # Dosya adını küçük harfe çevirip "lorinna" ve "logo" içeriyor mu diye kontrol et
        if 'lorinna' in filename.lower() and 'logo' in filename.lower():
            return os.path.join(LOGOS_FOLDER, filename)
    return None

# --- 5. KULLANICI ARAYÜZÜ ---

# BAŞLIK DÜZENLEMESİ: Logo ve Sürüm bilgisi eklendi
col_logo_title, col_version_text = st.columns([0.2, 1])

logo_path = find_lorinna_logo() # Logonun yolunu bulmaya çalış

with col_logo_title:
    try:
        if logo_path:
            st.image(logo_path, width=50) 
        else:
            st.markdown("👃") # Logo bulunamazsa emoji
    except Exception:
        st.markdown("👃")


with col_version_text:
    st.markdown(f"<h1 style='display: inline;'>LRN Koku Rehberi </h1> <span style='font-size: 0.5em; color: gray;'>({APP_VERSION})</span>", unsafe_allow_html=True)


st.markdown(f"**Toplam {len(stok_df)}** stoklu ürün. (Nota/Kategori aramalarında tüm eşleşenleri gösterir.)")

# ARAMA ÇUBUĞU SADELEŞTİRİLDİ
col_search, col_space = st.columns([1, 3]) 

with col_search:
    search_query = st.text_input("Arama Kutusu", placeholder="örn: 255 veya çiçeksi", key="main_search_query", label_visibility="collapsed")
    
button_pressed = st.button("Koku Bul", type="primary", use_container_width=True)

final_query = st.session_state.main_search_query

if final_query and (button_pressed or final_query != st.session_state.get('last_search_query', '')):
    
    if len(final_query) < 2 and not final_query.isdigit():
        st.warning("Lütfen en az 2 harf veya geçerli bir kod girin.")
    else:
        main_product, recommended_parfumes = find_similar(final_query)
        st.session_state.last_search_query = final_query 

        st.divider()

        # 1. Ana Ürünü Listele (Varsa)
        if main_product is not None:
             st.subheader(f"Aranan Ürün: {main_product['isim']}")
             with st.container(border=True):
                 display_stok_card(main_product)
             st.divider()

        # 2. Önerileri Listele (Yan Yana Görüntü)
        if not recommended_parfumes.empty:
            
            if main_product is not None: 
                 st.subheader(f"Size En Çok Benzeyen ({len(recommended_parfumes)} Adet):")
            else: 
                st.subheader(f"Eşleşen Ürünler ({len(recommended_parfumes)} Adet):")
            
            cols = st.columns(3) 
            
            for i, (index, parfum_row) in enumerate(recommended_parfumes.iterrows()):
                with cols[i % 3]:
                    with st.container(border=True):
                        display_stok_card(parfum_row)
        else:
            if main_product is None:
                st.error(f"'{final_query}' aramasına eşleşen stok ürünü bulunamadı.")
            else:
                 st.info(f"'{main_product['isim']}' ürününe benzeyen başka ürün bulunamadı.")