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
        color: #333333;
    }
    .result-card h4 {
        color: #333333;
    }
    .alt-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #007bff;
        color: #333333;
    }
    .alt-card h5, .alt-card p {
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

# --- VERİYİ YÜKLEME ---
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("Lorinna_Master_Veri.xlsx")
        df['Orijinal_Ad_Lorinna'] = df['Orijinal_Ad_Lorinna'].fillna("").astype(str)
        df['Notalar_KULLANMAK'] = df['Notalar_KULLANMAK'].fillna("").astype(str)
        df['Lorinna_Kodu'] = df['Lorinna_Kodu'].fillna("").astype(str)
        return df
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
        return pd.DataFrame()

df_master = load_data()

# --- ARAYÜZ VE ARAMA MANTIĞI ---
st.title("✨ Lorinna Akıllı Parfüm Bulucu")
st.write("Aklınızdaki parfümü veya Lorinna kodunu yazın, size en uygun seçeneği bulalım.")

if not df_master.empty:
    search_query = st.text_input("Aramak istediğiniz parfüm markası, adı veya Lorinna kodu:", placeholder="Örn: Baccarat Rouge veya LRN.09.001")
    
    if search_query:
        search_query_lower = search_query.lower().strip()
        
        # --- 1. ADIM: KULLANICI KOD MU YAZDI İSİM Mİ? ---
        # Eğer kullanıcı "lrn" yazdıysa VEYA sadece rakamlardan oluşan bir şey yazdıysa (örn: 001, 283) 
        # Kesin (Exact) Arama yapmalıyız, Fuzzy search (esnek arama) YAPMAMALIYIZ!
        is_code_search = "lrn" in search_query_lower or search_query_lower.replace(".", "").isdigit()
        
        eslesen_urun = None
        exact_match_found = False

        if is_code_search:
            # Sadece Kod sütununda tam eşleşme arıyoruz
            # Kodun içinde örneğin "283" veya "lrn.09.283" geçiyor mu diye bakarız.
            mask = df_master['Lorinna_Kodu'].str.lower().str.contains(search_query_lower, na=False)
            matched_rows = df_master[mask]
            
            if not matched_rows.empty:
                # Tam kod bulundu!
                eslesen_urun = matched_rows.iloc[0]
                exact_match_found = True
            
        else:
            # Kullanıcı isim girdi (Örn: Tom Ford). Burada esnek (Fuzzy) arama çalışır.
            orijinal_isimler_listesi = df_master['Orijinal_Ad_Lorinna'].tolist()
            best_match_name, match_score = process.extractOne(search_query_lower, orijinal_isimler_listesi)
            
            if match_score >= 80:
                eslesen_urun = df_master[df_master['Orijinal_Ad_Lorinna'] == best_match_name].iloc[0]
                exact_match_found = True


        # --- 2. ADIM: SONUÇLARI GÖSTER ---
        
        if exact_match_found:
            # Ya kodu tam bulduk ya da ismi %80 üzeri tam eşleştirdik
            st.success("✅ Aradığınız parfüm doğrudan Lorinna stoklarında mevcut!")
            
            st.markdown(f"""
            <div class="result-card">
                <h4>Aradığınız Koku: {eslesen_urun['Orijinal_Ad_Lorinna'].title()}</h4>
                <h3>Lorinna Karşılığı: <span style="color: #d32f2f;">{eslesen_urun['Lorinna_Kodu']}</span></h3>
            </div>
            """, unsafe_allow_html=True)
            
            notalar_temiz = str(eslesen_urun['Notalar_KULLANMAK']).replace("[", "").replace("]", "").replace("'", "")
            if notalar_temiz.lower() != 'nan' and notalar_temiz != '':
                st.write(f"**Bu parfümün koku profili:** {notalar_temiz}")
            else:
                st.write("**Bu parfümün koku profili:** Detaylı nota bilgisi bulunamadı.")

        else:
            # --- 3. ADIM: STOKTA YOKSA NOTAYA/İSME GÖRE ÖNERİ YAP ---
            if is_code_search:
                st.error("Aradığınız kodda bir parfüm stoklarımızda bulunamadı.")
            else:
                st.warning(f"'{search_query}' isimli parfümün doğrudan bir Lorinna karşılığı bulunamadı. Ancak bu koku profiline yakın alternatiflerimizi inceleyin:")
                
                # Master veri içinde basit bir puanlama yapıyoruz
                df_master['Toplam_Puan'] = 0.0
                for idx, row in df_master.iterrows():
                    text_to_search = str(row['Orijinal_Ad_Lorinna']).lower() + " " + str(row['Notalar_KULLANMAK']).lower() + " " + str(row['Parfum_Tanimi']).lower()
                    _, p_score = process.extractOne(search_query_lower, [text_to_search])
                    df_master.at[idx, 'Toplam_Puan'] = p_score
                    
                top_3 = df_master.sort_values(by='Toplam_Puan', ascending=False).head(3)
                
                for index, row in top_3.iterrows():
                    notalar_alt = str(row['Notalar_KULLANMAK']).replace("[", "").replace("]", "").replace("'", "")
                    if notalar_alt.lower() == 'nan': notalar_alt = "Bilgi yok"
                    
                    st.markdown(f"""
                    <div class="alt-card">
                        <h5>Lorinna Kodu: <span style="color:#d32f2f;">{row['Lorinna_Kodu']}</span></h5>
                        <p><strong>Benzer Koku Grubu:</strong> {row['Orijinal_Ad_Lorinna'].title()}</p>
                        <p style="font-size:0.9em; color:gray;">Ana Akortlar: {notalar_alt}</p>
                    </div>
                    """, unsafe_allow_html=True)
