import streamlit as st
import pandas as pd
from thefuzz import process
import numpy as np

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Lorinna Akıllı Parfüm Bulucu", page_icon="✨", layout="centered")

# --- CSS (Görsel Tasarım) ---
st.markdown("""
<style>
    .result-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        border-left: 5px solid #28a745;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .alt-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #007bff;
    }
    .match-percent {
        font-weight: bold;
        color: #007bff;
    }
</style>
""", unsafe_allow_html=True)

# --- VERİYİ YÜKLEME ---
@st.cache_data
def load_data():
    try:
        # Yeni oluşturduğumuz Master dosyayı okuyoruz
        df = pd.read_excel("Lorinna_Master_Veri.xlsx")
        
        # Nan (boş) değerleri temizleyelim
        df['Orijinal_Ad_Lorinna'] = df['Orijinal_Ad_Lorinna'].fillna("").astype(str)
        df['Notalar_KULLANMAK'] = df['Notalar_KULLANMAK'].fillna("").astype(str)
        return df
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}. Lütfen Lorinna_Master_Veri.xlsx dosyasının GitHub'da olduğuna emin olun.")
        return pd.DataFrame()

df_master = load_data()

# --- BENZERLİK HESAPLAMA (BASİT NOTA KESİŞİMİ) ---
def calculate_note_similarity(target_notes_str, current_notes_str):
    if not target_notes_str or not current_notes_str or target_notes_str == 'nan' or current_notes_str == 'nan':
        return 0
    
    # Notaları virgül veya listelerden ayırıp set haline getiriyoruz
    import re
    def extract_words(text):
        words = re.findall(r"\'(.*?)\'", text) # ['misk', 'odunsu'] formatı için
        if not words:
            words = text.replace('[', '').replace(']', '').replace("'", "").split(',')
        return set([w.strip().lower() for w in words if w.strip()])

    target_set = extract_words(target_notes_str)
    current_set = extract_words(current_notes_str)
    
    if not target_set: return 0
    
    # Ortak notaları bul ve yüzde hesapla (Jaccard veya Kesişim)
    intersection = target_set.intersection(current_set)
    # Kaç hedefin kaçı eşleşti
    score = (len(intersection) / len(target_set)) * 100
    return min(100, score) # Maksimum 100

# --- ARAYÜZ VE ARAMA MANTIĞI ---
st.title("✨ Lorinna Akıllı Parfüm Bulucu")
st.write("Aklınızdaki parfümü veya Lorinna kodunu yazın, size en uygun seçeneği bulalım.")

if not df_master.empty:
    search_query = st.text_input("Aramak istediğiniz parfüm markası, adı veya Lorinna kodu:", placeholder="Örn: Baccarat Rouge, Black Muscs veya LRN.09.001")
    
    if search_query:
        # 1. Doğrudan Kod veya Tam İsim Araması
        search_query_lower = search_query.lower()
        orijinal_isimler_listesi = df_master['Orijinal_Ad_Lorinna'].tolist()
        kodlar_listesi = df_master['Lorinna_Kodu'].astype(str).tolist()
        
        # Önce Kod ile arama var mı?
        exact_code_match = df_master[df_master['Lorinna_Kodu'].astype(str).str.lower().str.contains(search_query_lower)]
        
        # Eğer LRN kodu ile aramadıysa, isme göre en iyi eşleşmeyi bulalım (Fuzzy Search)
        best_match_name, match_score = process.extractOne(search_query_lower, orijinal_isimler_listesi)
        
        # --- SENARYO A: DOĞRUDAN STOKTA VAR ---
        # Eğer skor 80'den büyükse, yani aradığı şey Lorinna stoklarında birebir varsa
        if match_score >= 80:
            st.success("✅ Aradığınız parfüm doğrudan Lorinna stoklarında mevcut!")
            eslesen_urun = df_master[df_master['Orijinal_Ad_Lorinna'] == best_match_name].iloc[0]
            
            st.markdown(f"""
            <div class="result-card">
                <h4>Aradığınız Koku: {eslesen_urun['Orijinal_Ad_Lorinna'].title()}</h4>
                <h3>Lorinna Karşılığı: <span style="color: #d32f2f;">{eslesen_urun['Lorinna_Kodu']}</span></h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("Bu parfümün koku profili (İçeriği):", eslesen_urun['Notalar_KULLANMAK'].replace("[", "").replace("]", "").replace("'", ""))

        # --- SENARYO B: STOKTA YOK, NOTALARA GÖRE ÖNERİ YAP ---
        # Kullanıcı kodu yazmışsa (exact_code_match) veya aradığı şey tam uymadıysa nota benzerliği çalışsın
        else:
            st.warning("Aradığınız ismin doğrudan bir Lorinna karşılığı bulunamadı. Ancak koku profilinize en yakın harika alternatiflerimizi listeledik:")
            
            # (Gelişmiş Senaryo: Eğer 60 binlik veritabanına bağlı bir API olsaydı önce orada arardık. 
            # Şu an elimizdeki Master Data içinde aratılan kelimeye en yakın koku profillerini bulacağız).
            
            # NOT: Bu demo versiyonda, "stokta olmayan" bir şey arandığında, Master içindeki diğer tüm parfümlerle 
            # basit kelime benzerliği ve nota eşleşmesi yaparak en mantıklı 3 tanesini getireceğiz.
            
            # Sadece Master veri içinde kelime/nota bazlı basit bir arama yapıyoruz
            df_master['Toplam_Puan'] = 0.0
            for idx, row in df_master.iterrows():
                # Arama metni notalarda veya isimde geçiyorsa puan ver
                text_to_search = str(row['Orijinal_Ad_Lorinna']).lower() + " " + str(row['Notalar_KULLANMAK']).lower() + " " + str(row['Parfum_Tanimi']).lower()
                
                # Aranan kelimenin kaç harfi uyuşuyor basit puanı
                _, p_score = process.extractOne(search_query_lower, [text_to_search])
                df_master.at[idx, 'Toplam_Puan'] = p_score
                
            top_3 = df_master.sort_values(by='Toplam_Puan', ascending=False).head(3)
            
            for index, row in top_3.iterrows():
                st.markdown(f"""
                <div class="alt-card">
                    <h5>Lorinna Kodu: <span style="color:#d32f2f;">{row['Lorinna_Kodu']}</span></h5>
                    <p><strong>Benzer Koku Grubu:</strong> {row['Orijinal_Ad_Lorinna'].title()}</p>
                    <p style="font-size:0.9em; color:gray;">Ana Akortlar: {str(row['Notalar_KULLANMAK']).replace("[", "").replace("]", "").replace("'", "")}</p>
                </div>
                """, unsafe_allow_html=True)
        
