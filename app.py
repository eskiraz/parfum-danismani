# BU KODUN TAMAMINI KOPYALAYIN VE app.py DOSYASINA YAPIŞTIRIN (v7.1 - İndirgenmiş Matris Okuyucu)

import streamlit as st
import pandas as pd
import joblib 
from sklearn.metrics.pairwise import cosine_similarity # Nota araması için hala gerekli
import numpy as np
import re

# --- 0. OTURUM DURUMU (SESSION STATE) BAŞLATMA ---
if 'search_history' not in st.session_state:
    st.session_state.search_history = []

# --- 1. SAYFA AYARLARI ---
st.set_page_config(
    page_title="LRN Koku Rehberi v7.1 (Final Fix)",
    page_icon="👃",
    layout="wide"
)

# --- 2. VERİ YÜKLEME VE MODELİ YÜKLEME ---
@st.cache_resource
def load_data():
    print("Önceden hesaplanmış model dosyaları yükleniyor...")
    try:
        # Modelleri yükle
        cosine_sim_reduced = joblib.load('cosine_sim_reduced.pkl') # YENİ VE İNDİRGENMİŞ MATRİS
        vectorizer = joblib.load('vectorizer.pkl')
        all_perfumes_df = joblib.load('all_perfumes_df.pkl')
        stock_indices = joblib.load('stock_indices.pkl') # Stok listesinin indexleri
        
        # Stok ve Ana DB'yi de ayır (Kart gösterme için)
        stok_db_df = pd.read_csv("stok_listesi_clean.csv")
        ana_db_df = pd.read_csv("ana_db_clean.csv")

        # Stok parfümlerinin isimlerini DataFrame'den çek
        stock_perfumes_df = all_perfumes_df.iloc[stock_indices].reset_index(drop=True)

        print("Model hazırlandı. Uygulama başlatılıyor.")
        
        return all_perfumes_df, ana_db_df, stok_db_df, cosine_sim_reduced, vectorizer, stock_perfumes_df

    except FileNotFoundError:
        st.error("HATA: Model dosyaları (.pkl) bulunamadı.")
        st.error("Lütfen 'py model_olustur.py' komutunu çalıştırdığınızdan ve .pkl dosyalarını GitHub'a yüklediğinizden emin olun.")
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
    
    st.markdown(f"#### **{stok_kod}** ({parfum_serisi['isim']})")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"**Kategori:** {stok_kategori}")
        st.markdown(f"**Cinsiyet:** {parfum_serisi['cinsiyet']}")
        try:
            not_listesi = eval(stok_notalar)
            st.markdown(f"**Ana Notalar:** {', '.join(not_listesi[:5])}...")
        except:
             st.markdown(f"**Ana Notalar:** Notalar bulunamadı.")
    
    with col2:
        st.button("Satın Al >", key=f"buy_{stok_kod}")

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

# --- 4. BENZERLİK BULMA MOTORU (ANA BEYİN) ---

def find_similar(search_term, gender_filter="Tümü"):
    
    if search_term and search_term.lower() not in [h.lower() for h in st.session_state.search_history]:
        st.session_state.search_history.insert(0, search_term)
        st.session_state.search_history = st.session_state.search_history[:5]
    
    recommendations = []
    
    # 1. Parfüm Adı Araması (Tüm Evrende Ara)
    match = all_perfumes_df[all_perfumes_df['isim'].str.contains(search_term, case=False, flags=re.IGNORECASE)]
    
    if not match.empty:
        found_perfume = match.iloc[0]
        perfume_index = found_perfume.name # 70k'lık büyük indekste nerede?
        
        # Eğer stokta varsa direkt göster
        if found_perfume['tip'] == 'Stok':
            st.success("**Aradığınız Parfüm Stoklarımızda Mevcut!**")
            display_stok_card(found_perfume)
            return

        # Benzerlik skorlarını al (70k'lık satırdan 122 kolonluk skorları alır)
        sim_scores_122 = list(enumerate(cosine_sim_reduced_matrix[perfume_index]))
        sim_scores_122 = sorted(sim_scores_122, key=lambda x: x[1], reverse=True) # Stok listesi indeksine göre sıralı
        
        display_original_card(found_perfume)

        # En iyi 5 stok parfümü filtrele
        for stock_sim_index, score in sim_scores_122:
             if score > 0.0:
                 # stock_sim_index: 0 ile 121 arasındaki indeks
                 recommended_parfum = stock_perfumes_df.iloc[stock_sim_index]
                 
                 if gender_filter == "Tümü" or recommended_parfum['cinsiyet'] == gender_filter:
                     recommendations.append(recommended_parfum)
             
             if len(recommendations) >= 5:
                 break
        
        return recommendations

    else:
        # 2. Nota/Hissiyat Araması
        st.warning(f"**'{search_term}'** adında bir parfüm bulunamadı. Nota/Hissiyat olarak arama yapılıyor...")
        
        try:
            search_vector = vectorizer.transform([search_term]) # Arama terimini vektöre dönüştür
            
            # Sadece stok parfümlerinin matrixini elde et (cosine_sim_reduced'ın kolonları)
            koku_matrix_stock_only = cosine_sim_reduced_matrix[stock_perfumes_df.index, :]

            # Arama vektörünün stok parfümlerine benzerliğini hesapla (1 x 122)
            nota_sim_scores_122 = cosine_similarity(search_vector, koku_matrix_stock_only) 
            
            # Skorları sırala
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

        except Exception as e:
            # st.error(f"Nota arama hatası: {e}") # Hata gösterimini kapattık
            return []


# --- 5. KULLANICI ARAYÜZÜ (STREAMLIT) ---

st.title("👃 LRN Koku Rehberi v7.1 (Final Fix)")
st.markdown(f"**Toplam {len(all_perfumes_df)}** parfüm içeren Koku Evreni.")

tab1, tab2, tab3, tab4 = st.tabs(["🌟 Akıllı Arama Motoru", "📚 Koku Sözlüğü", "🔎 Popüler Notaları Keşfet", "🔥 LRN Vitrin"])

# ... (Kullanıcı arayüzü kısmı aynı kalır) ...

with tab1:
    st.header("Akıllı Arama Motoru")
    st.markdown("Aradığınız orijinal parfümün adını (örn: `Creed Aventus`) veya sevdiğiniz bir notayı (örn: `vanilya`) yazın.")
    
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
    
    if st.session_state.search_history:
        with st.expander("Son Aramalarınız"):
            history_cols = st.columns(len(st.session_state.search_history))
            for i, query in enumerate(st.session_state.search_history):
                if history_cols[i].button(query, key=f"hist_{query}"):
                    st.session_state.main_search_query = query
                    search_triggered = True

    
    if st.button("Koku Bul", type="primary") or search_triggered:
        final_query = st.session_state.main_search_query
        
        if len(final_query) < 2:
            st.warning("Lütfen en az 2 harf girin.")
        else:
            recommended_parfumes = find_similar(final_query, st.session_state.main_gender_filter)
            
            with results_container.container():
                st.divider()
                
                if not recommended_parfumes:
                    st.error(f"Üzgünüz, '{final_query}' aramasıyla eşleşen veya benzeyen stokta bir ürün bulamadık.")
                else:
                    st.subheader(f"'{final_query}' Araması İçin Seçtiklerimiz:")
                    for parfum_row in recommended_parfumes:
                        with st.container(border=True):
                            display_stok_card(parfum_row)


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


with tab3:
    st.header("🔎 Popüler Notaları Keşfet")
    st.markdown("Aşağıdaki popüler notalara tıklayarak, bu notaları içeren stoktaki parfümleri keşfedin.")
    
    populer_notalar = ["Vanilya", "Ud", "Misk", "Amber", "Paçuli", "Gül", "Lavanta", "Bergamot", "Deri", "Yasemin"]
    
    col_count = 5
    cols = st.columns(col_count)
    
    for i, nota in enumerate(populer_notalar):
        col = cols[i % col_count]
        if col.button(nota, key=f"nota_{nota}"):
            st.divider()
            st.subheader(f"'{nota}' Notalı Parfümler:")
            
            recommended_parfumes = find_similar(nota, "Tümü")
            
            if not recommended_parfumes:
                st.error(f"Stoklarımızda '{nota}' içeren belirgin bir parfüm bulunamadı.")
            else:
                for parfum_row in recommended_parfumes:
                    with st.container(border=True):
                        display_stok_card(parfum_row)

with tab4:
    st.header("🔥 LRN Vitrin: Editörün Seçimleri")
    st.markdown("Sizin için seçtiğimiz en popüler LRN parfümleri.")
    
    try:
        vitrin_kodlari = stok_db_df['kod'].head(4).tolist() 
        
        if not vitrin_kodlari:
            st.warning("Vitrine eklenecek LRN parfümü bulunamadı.")
        else:
            for kod in vitrin_kodlari:
                stok_row = stok_db_df[stok_db_df['kod'] == kod].iloc[0]
                with st.container(border=True):
                    # Stokta olanların gösterimi için yeni bir DataFrame oluşturmaya gerek yok
                    # stok_row'u doğrudan kullanıyoruz.
                    st.markdown(f"#### **{kod}** ({stok_row['orijinal_ad']})")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(f"**Kategori:** {stok_row['kategori']}")
                        st.markdown(f"**Cinsiyet:** {stok_row['cinsiyet']}")
                        try:
                            not_listesi = eval(stok_row['notalar'])
                            st.markdown(f"**Ana Notalar:** {', '.join(not_listesi[:5])}...")
                        except:
                            st.markdown(f"**Ana Notalar:** Notalar bulunamadı.")
                    with col2:
                        st.button("Satın Al >", key=f"buy_vitrin_{kod}")
                    
    except Exception as e:
        st.error(f"Vitrin yüklenirken bir hata oluştu: {e}")