# BU KODUN TAMAMINI KOPYALAYIN VE app.py DOSYASINA YAPIŞTIRIN (v6.1 - DOM Hata Giderici)

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# --- 0. OTURUM DURUMU (SESSION STATE) BAŞLATMA ---
# Arama geçmişini tutmak için
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="LRN Koku Rehberi v6.1 (Stabilite Fix)",
    page_icon="👃",
    layout="wide"
)

# --- 2. VERİ YÜKLEME VE İŞLEME ---
@st.cache_resource
def load_data():
    print("Veri yükleniyor ve model (koku evreni) oluşturuluyor...")
    try:
        # 1. Ana Veritabanını (68k) Yükle
        ana_db = pd.read_csv("ana_db_clean.csv")
        ana_db = ana_db.rename(columns={'Parfüm İsmi': 'isim', 'cinsiyet': 'cinsiyet'})
        ana_db['tip'] = 'Original'
        ana_db = ana_db[['isim', 'cinsiyet', 'notalar_str', 'tip', 'aciklama']]
        
        # 2. Stok Veritabanını (122) Yükle
        stok_db = pd.read_csv("stok_listesi_clean.csv")
        stok_db = stok_db.rename(columns={'orijinal_ad': 'isim', 'cinsiyet': 'cinsiyet'})
        stok_db['tip'] = 'Stok'
        
        # 3. İki veritabanını birleştir (Tek bir "koku evreni" için)
        all_perfumes = pd.concat([
            ana_db[['isim', 'cinsiyet', 'notalar_str', 'tip']],
            stok_db[['isim', 'cinsiyet', 'notalar_str', 'tip']]
        ], ignore_index=True)
        
        # 'cinsiyet' sütunundaki olası NaN (boş) değerleri 'Unisex' ile doldur
        all_perfumes['cinsiyet'] = all_perfumes['cinsiyet'].fillna('Unisex')
        
        # 4. Makine Öğrenimi Modelini (TF-IDF Vectorizer) Kur
        # 'notalar_str' sütunundaki boş (NaN) değerleri temizle
        all_perfumes['notalar_str'] = all_perfumes['notalar_str'].fillna('')
        
        vectorizer = CountVectorizer(min_df=2, max_df=0.8) # Notaları vektöre dönüştürür
        koku_matrix = vectorizer.fit_transform(all_perfumes['notalar_str'])
        
        # 5. Benzerlik Matrisini (Cosine Similarity) Oluştur
        cosine_sim = cosine_similarity(koku_matrix, koku_matrix)
        
        print("Model hazırlandı. (Toplam: {} parfüm)".format(len(all_perfumes)))
        
        return all_perfumes, ana_db, stok_db, cosine_sim, vectorizer

    except FileNotFoundError as e:
        st.error(f"HATA: Gerekli .csv dosyası bulunamadı. '{e.filename}'")
        st.error("Lütfen 'py islem.py' komutunu çalıştırdığınızdan ve .csv dosyalarının oluştuğundan emin olun.")
        st.stop()
    except Exception as e:
        st.error(f"Veri yüklenirken kritik bir hata oluştu: {e}")
        st.stop()

# Veri ve Modeli Yükle
all_perfumes_df, ana_db_df, stok_db_df, cosine_sim_matrix, vectorizer = load_data()


# --- 3. YARDIMCI FONKSİYONLAR (KART GÖSTERİMİ) ---

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

def display_original_card(parfum_serisi):
    """Stokta olmayan (Orijinal) bir parfümü kart olarak gösterir."""
    st.info(f"**Aradığınız Parfüm: {parfum_serisi['isim']}** ({parfum_serisi['cinsiyet']})")
    st.markdown("Bu parfüm stoklarımızda bulunmamaktadır. Size en çok benzeyen stoktaki parfümlerimizi aşağıda listeledik:")
    
    # 'aciklama' verisini ana_db_df'den çek
    aciklama_row = ana_db_df[ana_db_df['isim'] == parfum_serisi['isim']]
    if not aciklama_row.empty and 'aciklama' in aciklama_row.columns:
        aciklama = aciklama_row.iloc[0]['aciklama']
        if pd.notna(aciklama):
            with st.expander("Orijinal Parfümün Açıklaması"):
                st.write(aciklama)

# --- 4. BENZERLİK BULMA MOTORU (ANA BEYİN) ---

def find_similar(search_term, gender_filter="Tümü"):
    """
    Ana Arama Motoru. İsimle veya notayla arama yapar.
    """
    
    # --- Arama Geçmişine Ekleme ---
    if search_term and search_term.lower() not in [h.lower() for h in st.session_state.search_history]:
        st.session_state.search_history.insert(0, search_term)
        st.session_state.search_history = st.session_state.search_history[:5]
    # -------------------------------


    # 1. Arama Terimi İsim Listesinde Var mı? (Parfüm Adı Araması)
    match = all_perfumes_df[all_perfumes_df['isim'].str.contains(search_term, case=False, flags=re.IGNORECASE)]
    
    if not match.empty:
        found_perfume = match.iloc[0]
        perfume_index = found_perfume.name
        
        # Kartı göster
        if found_perfume['tip'] == 'Stok':
            stok_row = stok_db_df[stok_db_df['isim'] == found_perfume['isim']].iloc[0]
            st.success("**Aradığınız Parfüm Stoklarımızda Mevcut!**")
            display_stok_card(stok_row)
            return
        else:
            display_original_card(found_perfume)

        # Benzerlik skorlarını al
        sim_scores = list(enumerate(cosine_sim_matrix[perfume_index]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Sadece STOKTA olanları ve kendine benzemeyenleri filtrele
        recommendations = []
        for i, score in sim_scores[1:]:
            if all_perfumes_df.iloc[i]['tip'] == 'Stok':
                if gender_filter == "Tümü" or all_perfumes_df.iloc[i]['cinsiyet'] == gender_filter:
                    recommendations.append(i)
            if len(recommendations) >= 5:
                break
        
        return recommendations

    else:
        # 2. Eşleşme bulunamadı (Nota/Hissiyat Araması)
        st.warning(f"**'{search_term}'** adında bir parfüm bulunamadı. Nota/Hissiyat olarak arama yapılıyor...")
        
        try:
            search_vector = vectorizer.transform([search_term])
            nota_sim_scores = cosine_similarity(search_vector, cosine_sim_matrix.T)
            stok_indices = all_perfumes_df[all_perfumes_df['tip'] == 'Stok'].index
            
            stok_scores = []
            for i in stok_indices:
                if gender_filter == "Tümü" or all_perfumes_df.iloc[i]['cinsiyet'] == gender_filter:
                    stok_scores.append( (i, nota_sim_scores[0][i]) )
            
            stok_scores = sorted(stok_scores, key=lambda x: x[1], reverse=True)
            
            recommendations = [i for i in stok_scores[:5] if i[1] > 0.0]
            recommendations = [i[0] for i in recommendations]
            return recommendations

        except Exception:
            return []


# --- 5. KULLANICI ARAYÜZÜ (STREAMLIT) ---

st.title("👃 LRN Koku Rehberi v6.1 (Stabilite Fix)")
st.markdown(f"**Toplam {len(ana_db_df)}** orijinal parfüm ve **{len(stok_db_df)}** LRN parfümü içeren Koku Evreni.")

# --- SEKMELİ YAPI ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🌟 Akıllı Arama Motoru", 
    "📚 Koku Sözlüğü", 
    "🔎 Popüler Notaları Keşfet", 
    "🔥 LRN Vitrin"
])


# --- SEKME 1: AKILLI ARAMA MOTORU ---
with tab1:
    st.header("Akıllı Arama Motoru")
    st.markdown("Aradığınız orijinal parfümün adını (örn: `Creed Aventus`) veya sevdiğiniz bir notayı (örn: `vanilya`) yazın.")
    
    # Sonuçların gösterileceği alanı izole et
    results_container = st.empty()
    
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        search_query = st.text_input("Arama Kutusu", placeholder="örn: Baccarat Rouge 540 veya odunsu", key="main_search_query")
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
        
        if len(final_query) < 2:
            st.warning("Lütfen en az 2 harf girin.")
        else:
            # ARAMA MOTORUNU ÇALIŞTIR
            recommended_indices = find_similar(final_query, st.session_state.main_gender_filter)
            
            with results_container.container():
                st.divider()
                
                if not recommended_indices:
                    st.error(f"Üzgünüz, '{final_query}' aramasıyla eşleşen veya benzeyen stokta bir ürün bulamadık.")
                else:
                    st.subheader(f"'{final_query}' Araması İçin Seçtiklerimiz:")
                    for index in recommended_indices:
                        parfum_ismi = all_perfumes_df.iloc[index]['isim']
                        stok_row_list = stok_db_df[stok_db_df['isim'] == parfum_ismi]
                        if not stok_row_list.empty:
                            stok_row = stok_row_list.iloc[0]
                            with st.container(border=True):
                                display_stok_card(stok_row)
                        else:
                            st.warning(f"Stok bilgisi bulunamadı: {parfum_ismi}")


# --- SEKME 2: KOKU SÖZLÜĞÜ ---
with tab2:
    st.header("📚 Koku Aileleri Sözlüğü")
    st.markdown("Parfüm dünyasındaki ana koku ailelerini tanıyın.")

    with st.expander("**Çiçeksi (Floral)**"):
        st.write("Gül, yasemin, zambak, leylak gibi çiçek notalarının hakim olduğu, en popüler koku ailesidir. Genellikle feminen bir karakter taşır.")

    with st.expander("**Oryantal (Amber / Amber)**"):
        st.write("Sıcak, zengin ve baharatlı notalar içerir. Vanilya, tarçın, misk, amber ve egzotik reçineler bu ailenin temel taşlarıdır. Yoğun ve kalıcı kokulardır.")

    with st.expander("**Odunsu (Woody)**"):
        st.write("Sandal ağacı, sedir ağacı, paçuli ve vetiver gibi ağaç notalarının belirgin olduğu aile. Genellikle maskülen parfümlerde kullanılsa da unisex ve feminen parfümlerde de sıkça rastlanır.")
        
    with st.expander("**Narenciye (Citrus)**"):
        st.write("Limon, portakal, bergamot, mandalina gibi taze ve canlandırıcı narenciye notalarından oluşur. Genellikle 'spor' veya 'yazlık' kokular olarak bilinirler.")

    with st.expander("**Şipre (Chypre)**"):
        st.write("Adını Kıbrıs'tan alır. Genellikle bergamot (üst nota), meşe yosunu ve paçuli (alt notalar) kombinasyonuna dayanır. Zıtlıkların uyumudur; hem taze hem de derindir.")

    with st.expander("**Füjer (Fougère)**"):
        st.write("Fransızca 'eğrelti otu' anlamına gelir. Genellikle lavanta, meşe yosunu ve kumarin (tonka fasulyesi) notalarını içerir. Klasik erkek parfümlerinin temel ailelerinden biridir.")


# --- SEKME 3: POPÜLER NOTALARI KEŞFET ---
with tab3:
    st.header("🔎 Popüler Notaları Keşfet")
    st.markdown("Aşağıdaki popüler notalara tıklayarak, bu notaları içeren stoktaki parfümleri keşfedin.")
    
    populer_notalar = ["Vanilya", "Ud", "Misk", "Amber", "Paçuli", "Gül", "Lavanta", "Bergamot", "Deri", "Yasemin"]
    
    col_count = 5
    cols = st.columns(col_count)
    
    for i, nota in enumerate(populer_notalar):
        col = cols[i % col_count]
        if col.button(nota, key=f"nota_{nota}", use_container_width=True):
            st.divider()
            st.subheader(f"'{nota}' Notalı Parfümler:")
            
            # NOTA ARAMASINI ÇALIŞTIR
            recommended_indices = find_similar(nota, "Tümü")
            
            if not recommended_indices:
                st.error(f"Stoklarımızda '{nota}' içeren belirgin bir parfüm bulunamadı.")
            else:
                for index in recommended_indices:
                    parfum_ismi = all_perfumes_df.iloc[index]['isim']
                    stok_row = stok_db_df[stok_db_df['isim'] == parfum_ismi].iloc[0]
                    with st.container(border=True):
                        display_stok_card(stok_row)

# --- SEKME 4: LRN VİTRİN ---
with tab4:
    st.header("🔥 LRN Vitrin: Editörün Seçimleri")
    st.markdown("Sizin için seçtiğimiz en popüler LRN parfümleri.")
    
    try:
        # Kodun kırılmaması için ilk 4 kodu alıyoruz
        vitrin_kodlari = stok_db_df['kod'].head(4).tolist() 
        
        if not vitrin_kodlari:
            st.warning("Vitrine eklenecek LRN parfümü bulunamadı.")
        else:
            for kod in vitrin_kodlari:
                stok_row = stok_db_df[stok_db_df['kod'] == kod].iloc[0]
                with st.container(border=True):
                    display_stok_card(stok_row)
                    
    except Exception as e:
        st.error(f"Vitrin yüklenirken bir hata oluştu: {e}")