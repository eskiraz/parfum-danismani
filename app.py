import streamlit as st
import pandas as pd
import time
from thefuzz import process
import re

st.set_page_config(page_title="Lorinna AI Parfüm Motoru", page_icon="⚙️", layout="wide")

# --- GELİŞMİŞ CSS TASARIMI (YAN YANA GÖRÜNÜM) ---
st.markdown("""
<style>
    .engine-header { text-align: center; color: #f39c12; font-size: 2.5em; font-weight: bold; margin-bottom: 5px; }
    .engine-sub { text-align: center; color: #bdc3c7; font-size: 1.1em; margin-bottom: 30px; }
    .main-card { background-color: #2c3e50; border-radius: 12px; padding: 25px; border-left: 6px solid #f1c40f; box-shadow: 0 4px 8px rgba(0,0,0,0.2); height: 100%; }
    .main-card h3 { color: #ffffff; margin-top: 0; font-size: 1.2em;}
    .main-card h2 { color: #f1c40f; font-size: 1.8em; margin-bottom: 5px; }
    .alt-card { background-color: #34495e; border-radius: 10px; padding: 15px; margin-bottom: 10px; border: 1px solid #455a64; border-left: 5px solid #3498db; }
    .alt-card h4 { color: #ecf0f1; margin: 0 0 5px 0; font-size: 1em; }
    .alt-card h3 { color: #3498db; margin: 0 0 10px 0; font-size: 1.3em;}
    .note-tag { display: inline-block; background-color: #1abc9c; color: white; padding: 3px 8px; border-radius: 15px; font-size: 0.8em; margin: 2px; }
    .match-percent { font-size: 0.6em; color: #2ecc71; vertical-align: middle; }
</style>
""", unsafe_allow_html=True)

# ÖNBELLEĞİ ZORLA TEMİZLEYEN YENİ KOD (@st.cache_data YERİNE)
# Bu sayede GitHub'a yüklediğiniz yeni Excel dosyasını kesinlikle okuyacak!
def load_data():
    df = pd.read_excel("Lorinna_Master_Veri.xlsx")
    df['Orijinal_Ad_Lorinna'] = df['Orijinal_Ad_Lorinna'].fillna("").astype(str)
    df['Notalar_KULLANMAK'] = df['Notalar_KULLANMAK'].fillna("Bilgi Yok").astype(str)
    return df

df_master = load_data()

def clean_notes(note_str):
    if not note_str or note_str == 'nan' or note_str == 'Bilgi Yok': return []
    words = re.findall(r"\'(.*?)\'", note_str)
    if not words:
        words = note_str.replace('[', '').replace(']', '').replace("'", "").split(',')
    return [w.strip().lower() for w in words if w.strip()]

st.markdown('<div class="engine-header">⚙️ Lorinna AI Eşleştirme Motoru</div>', unsafe_allow_html=True)
st.markdown('<div class="engine-sub">Parfüm ismini veya LRN kodunu girin, yapay zeka koku profillerini analiz etsin.</div>', unsafe_allow_html=True)

col_space1, col_search, col_space2 = st.columns([1, 2, 1])
with col_search:
    search_query = st.text_input("Arama Yapın:", placeholder="Örn: Hypnotic Poison veya LRN.09.283")

st.write("---")

if search_query:
    search_query_lower = search_query.lower()
    
    with st.spinner('Yapay Zeka kütüphanesi taranıyor...'):
        time.sleep(0.5)
    
    # 1. DOĞRUDAN KOD ARAMASI
    is_code_search = "lrn" in search_query_lower or search_query_lower.replace(".", "").isdigit()
    match_found = False
    
    if is_code_search:
        exact_code_match = df_master[df_master['Lorinna_Kodu'].astype(str).str.lower().str.contains(search_query_lower)]
        if not exact_code_match.empty:
            eslesen_urun = exact_code_match.iloc[0]
            match_found = True
    else:
        # İSİM ARAMASI
        orijinal_isimler = df_master['Orijinal_Ad_Lorinna'].tolist()
        best_match_name, match_score = process.extractOne(search_query_lower, orijinal_isimler)
        if match_score >= 70:
            eslesen_urun = df_master[df_master['Orijinal_Ad_Lorinna'] == best_match_name].iloc[0]
            match_found = True

    if match_found:
        target_name = eslesen_urun['Orijinal_Ad_Lorinna'].title()
        target_code = eslesen_urun['Lorinna_Kodu']
        target_notes = clean_notes(eslesen_urun['Notalar_KULLANMAK'])
        
        # EKRANI İKİYE BÖL (Sol 1 birim, Sağ 1.5 birim genişlikte)
        col_main, col_alts = st.columns([1, 1.5])
        
        with col_main:
            st.success("✅ Doğrudan Eşleşme Bulundu!")
            tags_html = "".join([f'<span class="note-tag">{n}</span>' for n in target_notes]) if target_notes else "<span style='color:#bdc3c7'>Bu parfüm için nota verisi bulunamadı. (Öneri yapılamıyor)</span>"
            
            st.markdown(f"""
            <div class="main-card">
                <h3>Aranan Koku: {target_name}</h3>
                <h2>{target_code}</h2>
                <div style="margin-top:10px;"><strong>Tespit Edilen Koku Akortları:</strong><br><br>{tags_html}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_alts:
            if target_notes:
                st.markdown("### 🔬 Koku Profiline En Yakın Lorinna Alternatifleri:")
                
                df_master['Ortak_Nota_Sayisi'] = 0
                target_set = set(target_notes)
                
                for idx, row in df_master.iterrows():
                    if row['Lorinna_Kodu'] != target_code:
                        current_set = set(clean_notes(row['Notalar_KULLANMAK']))
                        if current_set:
                            kesisim = target_set.intersection(current_set)
                            # Kaç hedef nota tuttu?
                            skor = (len(kesisim) / len(target_set)) * 100 if len(target_set) > 0 else 0
                            df_master.at[idx, 'Ortak_Nota_Sayisi'] = skor
                
                # Ortak notası olan en iyi 3 tanesi
                top_3 = df_master[df_master['Ortak_Nota_Sayisi'] > 0].sort_values(by='Ortak_Nota_Sayisi', ascending=False).head(3)
                
                if not top_3.empty:
                    for index, row in top_3.iterrows():
                        benzer_notalar = clean_notes(row['Notalar_KULLANMAK'])
                        # Sadece ortak olanları turuncu yapalım
                        b_tags = ""
                        for n in benzer_notalar:
                            if n in target_set:
                                b_tags += f'<span class="note-tag" style="background-color:#e67e22;">{n}</span>'
                            else:
                                b_tags += f'<span class="note-tag" style="background-color:#7f8c8d;">{n}</span>'
                        
                        st.markdown(f"""
                        <div class="alt-card">
                            <h4>{row['Orijinal_Ad_Lorinna'].title()}</h4>
                            <h3>{row['Lorinna_Kodu']} <span class="match-percent">(%{int(row['Ortak_Nota_Sayisi'])} Nota Uyumu)</span></h3>
                            <div>{b_tags}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Bu koku profiline yeterince benzeyen başka bir alternatif bulunamadı.")
