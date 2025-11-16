# BU KODU app.py DOSYASINA YAPIŞTIRIN (v7.2 - Stabil Versiyon)
# Bu kod sadece stok verisini (122 parfüm) kullanarak uygulamayı hemen açar.

import streamlit as st
import pandas as pd
import json
import re

# --- 0. SABİT VERİLERİ YÜKLEME ---
@st.cache_resource
def load_data_reversion():
    try:
        # Sadece Stok Verisini Yükle (70K veriyi yüklemez!)
        stok_db = pd.read_csv("stok_listesi_clean.csv")
        stok_db = stok_db.rename(columns={'orijinal_ad': 'isim'})
        stok_db['notalar_str'] = stok_db['notalar'].apply(lambda x: ' '.join(eval(x)).lower())
        
        return stok_db

    except FileNotFoundError:
        st.error("HATA: Gerekli 'stok_listesi_clean.csv' dosyası bulunamadı.")
        st.error("Lütfen veritabanı dosyasının klasörde olduğundan emin olun.")
        st.stop()

stok_df = load_data_reversion()

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="LRN Koku Rehberi (Stabil Sürüm)",
    page_icon="👃",
    layout="wide"
)

# --- YARDIMCI FONKSİYON ---
def display_stok_card(parfum_serisi):
    """Stoktaki bir parfümü (LRN Kodu) kart olarak gösterir."""
    st.markdown(f"#### **{parfum_serisi['kod']}** ({parfum_serisi['isim']})")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**Kategori:** {parfum_serisi['kategori']}")
        st.markdown(f"**Cinsiyet:** {parfum_serisi['cinsiyet']}")
        try:
            not_listesi = eval(parfum_serisi['notalar'])
            st.markdown(f"**Ana Notalar:** {', '.join(not_listesi[:5])}...")
        except:
             st.markdown(f"**Ana Notalar:** Notalar bulunamadı.")
    
    with col2:
        st.button("Satın Al >", key=f"buy_{parfum_serisi['kod']}")

# --- 5. KULLANICI ARAYÜZÜ ---

st.title("👃 LRN Koku Rehberi (Stabil Sürüm)")
st.markdown(f"**Toplam {len(stok_df)}** stoklu ürün. Büyük veritabanı devre dışı.")

tab1, tab2 = st.tabs(["🌟 Stok Arama", "📚 Koku Sözlüğü"])

# --- SEKME 1: STOK ARAMA (MÜŞTERİ İÇİN KATEGORİ) ---
with tab1:
    st.header("Kategoriye Göre Arama")
    st.markdown("Müşterinizin sorduğu ana notayı veya kategoriyi seçin.")
    
    # Tüm kategorileri al
    all_categories = sorted(stok_df['kategori'].unique())
    all_categories.insert(0, "--- Hepsi ---")

    search_category = st.selectbox("Kategori Seçin", all_categories)
    
    if search_category != "--- Hepsi ---":
        result_df = stok_df[stok_df['kategori'] == search_category]
        st.subheader(f"'{search_category}' Kategorisindeki Ürünler ({len(result_df)} adet):")
        
        for index, row in result_df.iterrows():
            with st.container(border=True):
                display_stok_card(row)

# --- SEKME 2: KOKU SÖZLÜĞÜ (MÜŞTERİ İÇİN BİLGİ) ---
with tab2:
    st.header("📚 Koku Aileleri Sözlüğü")
    st.markdown("Müşterilerinize temel koku aileleri hakkında bilgi vermek için kullanın. (Odunsu, Pudralı, vb.)")

    with st.expander("**Odunsu (Woody)**"):
        st.write("Sandal ağacı, sedir ağacı, paçuli ve vetiver gibi ağaç notalarının belirgin olduğu aile. Genellikle maskülen parfümlerde kullanılsa da unisex ve feminen parfümlerde de sıkça rastlanır.")
        
    with st.expander("**Pudralı (Powdery)**"):
        st.write("İris, vanilya, misk ve tonka fasulyesi gibi notaların yumuşak, bebek pudrası veya kozmetik hissiyatı verdiği aile. Kadın ve unisex parfümlerde sıkça kullanılır.")

    with st.expander("**Çiçeksi (Floral)**"):
        st.write("Gül, yasemin, zambak, leylak gibi çiçek notalarının hakim olduğu, en popüler koku ailesidir. Genellikle feminen bir karakter taşır.")