# BU KODU app.py DOSYASINA YAPIŞTIRIN (v7.4 - v4.2 + Arama Geçmişi)

import streamlit as st
import pandas as pd
import json
import re

# --- YARDIMCI GÜVENLİK FONKSİYONU ---
# Notaları güvenli bir şekilde listeye çevirir.
def safe_eval(text):
    try:
        # Tırnakları, parantezleri temizleyip kelime listesi döndürür
        text = str(text).strip()
        if not text.startswith('[') and not text.endswith(']'):
            # Eğer liste formatında değilse basitçe string olarak döndür
            return text.lower()
        return ' '.join(eval(text)).lower()
    except:
        return ""

# --- 0. OTURUM DURUMU (SESSION STATE) BAŞLATMA ---
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""

# --- 1. VERİ YÜKLEME ---
@st.cache_resource
def load_data_v4():
    try:
        # Sadece Stok Verisini Yükle (RAM dostu)
        stok_db = pd.read_csv("stok_listesi_clean.csv")
        stok_db = stok_db.rename(columns={'orijinal_ad': 'isim'})
        
        # Notları düz metin araması için hazırla
        stok_db['search_content'] = stok_db['isim'] + ' ' + stok_db['kategori'] + ' ' + stok_db['cinsiyet'] + ' ' + stok_db['notalar'].apply(safe_eval)
        
        return stok_db

    except FileNotFoundError:
        st.error("HATA: Gerekli 'stok_listesi_clean.csv' dosyası bulunamadı.")
        st.stop()

stok_df = load_data_v4()

# --- 2. SAYFA AYARLARI ---
st.set_page_config(
    page_title="LRN Koku Rehberi v7.4 (Arama Geçmişli Stabil Sürüm)",
    page_icon="👃",
    layout="wide"
)

# --- 3. YARDIMCI FONKSİYONLAR ---
def display_stok_card(parfum_serisi):
    """Stoktaki bir parfümü (LRN Kodu) kart olarak gösterir."""
    st.markdown(f"#### **{parfum_serisi['kod']}** ({parfum_serisi['isim']})")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**Kategori:** {parfum_serisi['kategori']}")
        st.markdown(f"**Cinsiyet:** {parfum_serisi['cinsiyet']}")
        try:
            # Notaları güvenli bir şekilde gösterir
            not_listesi = eval(parfum_serisi['notalar'])
            st.markdown(f"**Ana Notalar:** {', '.join(not_listesi[:5])}...")
        except:
             st.markdown(f"**Ana Notalar:** Notalar bulunamadı.")
    
    with col2:
        st.button("Satın Al >", key=f"buy_{parfum_serisi['kod']}")

# --- 4. ARAMA MOTORU (Basit Metin Araması) ---
def simple_search(search_term, gender_filter):
    search_term = search_term.lower().strip()
    
    # Geçmişe kaydetme
    if search_term and search_term not in [h.lower() for h in st.session_state.search_history]:
        st.session_state.search_history.insert(0, search_term)
        st.session_state.search_history = st.session_state.search_history[:5]
    
    # 1. Metin araması (isim, kategori, nota içeriği)
    search_results = stok_df[
        stok_df['search_content'].str.contains(search_term, case=False, na=False)
    ]
    
    # 2. Cinsiyet filtresi
    if gender_filter != "Tümü":
        search_results = search_results[search_results['cinsiyet'] == gender_filter]
        
    return search_results.head(10) # En fazla 10 sonuç göster

# --- 5. KULLANICI ARAYÜZÜ ---

st.title("👃 LRN Koku Rehberi v7.4 (Arama Geçmişli Stabil Sürüm)")
st.markdown(f"**Toplam {len(stok_df)}** stoklu ürün. (Sadece stok verisi kullanılmaktadır)")

st.header("🌟 Stok Arama Motoru")
st.markdown("Aradığınız parfümün adını, notayı (`odunsu`, `vanilya`) veya kategoriyi (`Floral`) girin.")

# --- Arama Formu ---
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    # Arama Geçmişi ile bağlanacak arama kutusu
    search_query = st.text_input("Arama Kutusu", placeholder="örn: vanilya veya Baccarat", key="current_search_query")
    
with col2:
    gender_choice = st.selectbox("Cinsiyet Filtresi", ["Tümü", "Kadın", "Erkek", "Unisex"], key="main_gender_filter")

with col3:
    if st.button("Geçmişi Temizle", help="Arama geçmişini temizler"):
        st.session_state.search_history = []
        st.session_state.current_search_query = "" # Arama kutusunu da temizle
        st.rerun()

# --- Arama Geçmişi Bölümü ---
search_triggered = False
if st.session_state.search_history:
    with st.expander("Son Aramalarınız"):
        history_cols = st.columns(len(st.session_state.search_history))
        for i, query in enumerate(st.session_state.search_history):
            # Geçmiş butonuna basıldığında arama kutusunu güncelle
            if history_cols[i].button(query, key=f"hist_{query}"):
                st.session_state.current_search_query = query
                search_triggered = True

# --- Arama Tetikleme ---
if st.button("Koku Bul", type="primary") or search_triggered:
    final_query = st.session_state.current_search_query
    
    if len(final_query) < 2:
        st.warning("Lütfen en az 2 harf girin.")
    else:
        results = simple_search(final_query, st.session_state.main_gender_filter)
        
        st.divider()
        st.subheader(f"'{final_query}' Araması İçin Seçtiklerimiz ({len(results)} adet):")
        
        if results.empty:
            st.error(f"Üzgünüz, '{final_query}' aramasıyla eşleşen bir ürün bulunamadı.")
        else:
            for index, row in results.iterrows():
                with st.container(border=True):
                    display_stok_card(row)