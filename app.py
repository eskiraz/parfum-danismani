# BU KODUN TAMAMINI KOPYALAYIN VE app.py DOSYASINA YAPIŞTIRIN (v7.5 - Final Kod)

import streamlit as st
import pandas as pd
import joblib 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# --- 0. OTURUM DURUMU (SESSION STATE) BAŞLATMA ---
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="LRN Koku Rehberi v7.5 (Final)",
    page_icon="👃",
    layout="wide"
)

# --- 2. VERİ YÜKLEME VE MODELİ YÜKLEME ---
@st.cache_resource
def load_data():
    try:
        # Modelleri yükle (RAM Fix'li)
        cosine_sim_reduced = joblib.load('cosine_sim_reduced.pkl') 
        vectorizer = joblib.load('vectorizer.pkl')
        all_perfumes_df = joblib.load('all_perfumes_df.pkl')
        stock_indices = joblib.load('stock_indices.pkl') 
        
        # Stok ve Ana DB'yi de ayır (Kart gösterme için)
        stok_db_df = pd.read_csv("stok_listesi_clean.csv")
        ana_db_df = pd.read_csv("ana_db_clean.csv")

        # Stok parfümlerinin isimlerini DataFrame'den çek (Öneri çıktısı için)
        stock_perfumes_df = all_perfumes_df.iloc[stock_indices].reset_index(drop=True)

        return all_perfumes_df, ana_db_df, stok_db_df, cosine_sim_reduced, vectorizer, stock_perfumes_df

    except FileNotFoundError as e:
        st.error(f"HATA: Gerekli model dosyaları (.pkl) veya .csv dosyası bulunamadı. Lütfen tüm .pkl dosyalarının klasörde olduğunu kontrol edin.")
        st.stop()
    except Exception as e:
        st.error(f"Kritik Model Yükleme Hatası: {e}")
        st.stop()

# Veri ve Modeli Yükle
all_perfumes_df, ana_db_df, stok_db_df, cosine_sim_reduced_matrix, vectorizer, stock_perfumes_df = load_data()


# --- 3. YARDIMCI FONKSİYONLAR (KART GÖSTERİMİ) ---

def display_stok_card(parfum_serisi):
    """Stoktaki bir parfümü (LRN Kodu) kart olarak gösterir."""
    # Stoktaki ürünün kodunu stok_db_df'ten bul
    stok_kod = stok_db_df[stok_db_df['isim'] == parfum_serisi['isim']].iloc[0]['kod']
    stok_kategori = stok_db_df[stok_db_df['isim'] == parfum_serisi['isim']].iloc[0]['kategori']
    stok_notalar = stok_db_df[stok_db_df['isim'] == parfum_serisi['isim']].iloc[0]['notalar']
    
    st.markdown(f"**{stok_kod}** ({parfum_serisi['isim']})", help=f"Kodu: {stok_kod}")
    
    # Buton kaldırıldı, sadece içerik gösteriliyor
    st.markdown(f"**Kategori:** {stok_kategori}")
    st.markdown(f"**Cinsiyet:** {parfum_serisi['cinsiyet']}")

def display_original_card(parfum_serisi):
    """Stokta olmayan (Orijinal) bir parfümü kart olarak gösterir."""
    st.info(f"**Aradığınız Parfüm: {parfum_serisi['isim']}** ({parfum_serisi['cinsiyet']})")
    st.markdown("Bu parfüm stoklarımızda bulunmamaktadır. Size en çok benzeyen stoktaki parfümlerimizi aşağıda listeledik:")
    
    aciklama_row = ana_db_df[ana_db_df['isim'] == parfum_serisi['isim']]
    if not aciklama_row.empty and 'aciklama' in aciklama_row.columns:
        aciklama = aciklama_row.iloc[0]['aciklama']
        if pd.notna(aciklama):
            with st.expander("Orijinal Parfümün Açıklaması"):
                st.write(aciklama)

# --- 4. BENZERLİK BULMA MOTORU (LRN KODU VE ORİJİNAL AD ARAMASI) ---

def find_similar(search_term, gender_filter="Tümü"):
    
    if search_term and search_term.lower() not in [h.lower() for h in st.session_state.search_history]:
        st.session_state.search_history.insert(0, search_term)
        st.session_state.search_history = st.session_state.search_history[:5]
    
    recommendations = []
    
    # 1. LRN Koduna Göre Arama (Sadece Stok Verisinde)
    if search_term.isdigit() and len(search_term) <= 4: # LRN Kodu varsayımı
        stok_match = stok_db_df[stok_db_df['kod'] == search_term]
        
        if not stok_match.empty:
            found_perfume_name = stok_match.iloc[0]['isim']
            
            # Orijinal adı kullanarak genel listede indeksi bul
            match_in_all = all_perfumes_df[all_perfumes_df['isim'] == found_perfume_name]
            if not match_in_all.empty:
                 return get_recommendations_by_index(match_in_all.iloc[0].name, found_perfume_name, gender_filter)
            
    
    # 2. Parfüm Adı Araması (Tüm Evrende Ara)
    match = all_perfumes_df[all_perfumes_df['isim'].str.contains(search_term, case=False, flags=re.IGNORECASE)]
    
    if not match.empty:
        found_perfume = match.iloc[0]
        perfume_index = found_perfume.name 
        
        return get_recommendations_by_index(perfume_index, found_perfume['isim'], gender_filter)

    else:
        # 3. Nota/Hissiyat Araması
        st.warning(f"**'{search_term}'** adında bir parfüm veya kod bulunamadı. Nota/Hissiyat olarak arama yapılıyor...")
        
        try:
            search_vector = vectorizer.transform([search_term]) 
            
            # Sadece stok parfümlerinin matrixini elde et
            stock_matrix = cosine_sim_reduced_matrix[stock_perfumes_df.index, :]

            # Arama vektörünün stok parfümlerine benzerliğini hesapla (1 x 122)
            nota_sim_scores_122 = cosine_similarity(search_vector, stock_matrix) 
            
            stock_scores = list(enumerate(nota_sim_scores_122[0]))
            stock_scores = sorted(stock_scores, key=lambda x: x[1], reverse=True)
            
            for stock_sim_index, score in stock_scores:
                if score > 0.0:
                    recommended_parfum = stock_perfumes_df.iloc[stock_sim_index]
                    
                    if gender_filter == "Tümü" or recommended_parfum['cinsiyet'] == gender_filter:
                        recommendations.append(recommended_parfum)

                if len(recommendations) >= 5:
                    break
            
            return recommendations

        except Exception:
            return []

# Yardımcı fonksiyon: İndekse göre öneri listesi oluşturur
def get_recommendations_by_index(perfume_index, found_perfume_name, gender_filter):
    recommendations = []
    
    # 1. Kartı göster (Stokta varsa)
    is_in_stock = not stok_db_df[stok_db_df['isim'] == found_perfume_name].empty
    
    if is_in_stock:
        st.success(f"**Aradığınız Ürün ({stok_db_df[stok_db_df['isim'] == found_perfume_name].iloc[0]['kod']}) Stokta Mevcut!**")
        with st.container(border=True):
             display_stok_card(stok_db_df[stok_db_df['isim'] == found_perfume_name].iloc[0])
        st.divider()
        st.subheader("Size En Çok Benzeyen Ürünler:")
    else:
        display_original_card(all_perfumes_df.iloc[perfume_index])
        st.divider()


    # 2. Benzerlik skorlarını al (70k'lık satırdan 122 kolonluk skorları alır)
    sim_scores_122 = list(enumerate(cosine_sim_reduced_matrix[perfume_index]))
    sim_scores_122 = sorted(sim_scores_122, key=lambda x: x[1], reverse=True) 

    # En iyi 5 stok parfümü filtrele
    count = 0
    for stock_sim_index, score in sim_scores_122:
        if score > 0.0:
            recommended_parfum = stock_perfumes_df.iloc[stock_sim_index]
            
            # Eğer aranan ürün stokta varsa, öneri listesinde tekrar listelenmemeli (kendisi hariç)
            if is_in_stock and recommended_parfum['isim'] == found_perfume_name:
                continue

            if gender_filter == "Tümü" or recommended_parfum['cinsiyet'] == gender_filter:
                recommendations.append(recommended_parfum)
                count += 1
        
        if count >= 5:
            break
            
    return recommendations


# --- 5. KULLANICI ARAYÜZÜ ---

st.title("👃 LRN Koku Rehberi v7.5 (Final)")
st.markdown(f"**Toplam {len(all_perfumes_df)}** parfüm içeren Koku Evreni.")

tab1, tab2 = st.tabs(["🌟 Akıllı Arama Motoru", "📚 Koku Sözlüğü"])

# --- SEKME 1: AKILLI ARAMA MOTORU ---
with tab1:
    st.header("Akıllı Arama Motoru")
    st.markdown("Aradığınız orijinal parfümün adını, LRN kodunu (örn: `255`) veya sevdiğiniz bir notayı (`vanilya`) yazın.")
    
    results_container = st.empty()
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    # Arama kutusu ve filtreler
    with col1:
        search_query = st.text_input("Arama Kutusu", placeholder="örn: Creed Aventus, 255 veya odunsu", key="main_search_query")
    with col2:
        gender_choice = st.selectbox("Cinsiyet Filtresi", ["Tümü", "Kadın", "Erkek", "Unisex"], key="main_gender_filter")
    with col3:
        if st.button("Geçmişi Temizle", help="Arama geçmişini temizler"):
            st.session_state.search_history = []
            st.rerun()

    search_triggered = False
    
    # Arama Geçmişini Göster
    if st.session_state.search_history:
        with st.expander("Son Aramalarınız"):
            history_cols = st.columns(len(st.session_state.search_history))
            for i, query in enumerate(st.session_state.search_history):
                if history_cols[i].button(query, key=f"hist_{query}"):
                    st.session_state.main_search_query = query
                    search_triggered = True

    
    # Arama butonuna basıldıysa VEYA geçmiş aramaya tıklanıp search_triggered = True ise
    if st.button("Koku Bul", type="primary") or search_triggered:
        final_query = st.session_state.main_search_query
        
        if len(final_query) < 2 and not final_query.isdigit():
            st.warning("Lütfen en az 2 harf veya geçerli bir kod girin.")
        else:
            # ARAMA MOTORUNU ÇALIŞTIR
            recommended_parfumes = find_similar(final_query, st.session_state.main_gender_filter)
            
            with results_container.container():
                if recommended_parfumes is not None:
                    
                    if not recommended_parfumes:
                        st.error(f"Üzgünüz, '{final_query}' aramasıyla eşleşen veya benzeyen stokta bir ürün bulamadık.")
                    else:
                        # Yan Yana Sütunlarda Gösterme Mantığı
                        st.subheader(f"Önerilen Ürünler:")
                        cols = st.columns(3) # 3 sütun oluştur
                        
                        for i, parfum_row in enumerate(recommended_parfumes):
                            with cols[i % 3]:
                                with st.container(border=True):
                                    display_stok_card(parfum_row)


# --- SEKME 2: KOKU SÖZLÜĞÜ ---
with tab2:
    st.header("📚 Koku Aileleri Sözlüğü")
    st.markdown("Parfüm dünyasındaki ana koku ailelerini tanıyın.")
    # (Diğer sözlük expender'ları buraya eklenebilir)
    with st.expander("**Odunsu (Woody)**"):
        st.write("Sandal ağacı, sedir ağacı, paçuli ve vetiver gibi ağaç notalarının belirgin olduğu aile.")
    with st.expander("**Çiçeksi (Floral)**"):
        st.write("Gül, yasemin, zambak, leylak gibi çiçek notalarının hakim olduğu, en popüler koku ailesidir.")