# model_olustur.py (v9 - Final Süper Temizlik)
# BU KODUN TAMAMINI KOPYALAYIP model_olustur.py DOSYASINA YAPIŞTIRIN

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import sys
import re

NOTES_COLUMN = 'notalar_str' 

# --- YARDIMCI FONKSİYON: AGRESİF İSİM TEMİZLİĞİ ---
def super_clean_name(name):
    """Bütün özel karakterleri, boşlukları kaldırır ve küçük harfe çevirir."""
    name = str(name).lower().strip()
    # Sadece harf ve rakamları tutar (Boşluklar ve noktalamalar silinir)
    name = re.sub(r'[^a-z0-9]+', '', name) 
    return name

# --- 1. VERİ YÜKLEME ---
try:
    ana_db_df = pd.read_csv("ana_db_clean.csv")
    stok_db_df = pd.read_csv("stok_listesi_clean.csv")
except Exception as e:
    print(f"HATA: Gerekli CSV dosyaları bulunamadı. Detay: {e}")
    sys.exit()

# --- 2. SÜTUN VE İÇERİK FİX'LERİ (KESİN ÇÖZÜM) ---

# Sütun Adlarını Sabitleme (HATA ÇÖZÜMÜ)
ana_db_df = ana_db_df.rename(columns={'Parfüm İsmi': 'isim'}) 
stok_db_df = stok_db_df.rename(columns={'orijinal_ad': 'isim'}) 

# KRİTİK İÇERİK TEMİZLİĞİ: Eşleşme sorununu çözmek için isimleri süper temizle
print("Ürün isimleri eşleşme için temizleniyor (Süper Agresif)...")
ana_db_df['isim_clean'] = ana_db_df['isim'].apply(super_clean_name)
stok_db_df['isim_clean'] = stok_db_df['isim'].apply(super_clean_name)


# --- 3. VERİ ÖN İŞLEME VE TEMİZLEME ---
if NOTES_COLUMN not in ana_db_df.columns:
    print(f"KRİTİK HATA: {NOTES_COLUMN} sütunu ana veri dosyasında bulunamadı.")
    sys.exit()

print("Veri yüklendi, model oluşturuluyor...")

# Notaları boş olanları temizle
ana_db_df.dropna(subset=[NOTES_COLUMN], inplace=True)
ana_db_df = ana_db_df[ana_db_df[NOTES_COLUMN].astype(str).str.strip() != ''] 

# --- 4. MODEL OLUŞTURMA ---
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
tfidf_matrix = vectorizer.fit_transform(ana_db_df[NOTES_COLUMN]) 

# Cosine Benzerlik Matrisi (RAM SORUNU ÇÖZÜMÜ)
stok_isimleri_clean = stok_db_df['isim_clean'].tolist() # Temizlenmiş isim listesi kullanılır

# Eşleştirme artık temizlenmiş 'isim_clean' sütunu üzerinden yapılır (Hata çözülmelidir)
stok_indeksleri_all_data = ana_db_df[ana_db_df['isim_clean'].isin(stok_isimleri_clean)].index.tolist()

if not stok_indeksleri_all_data:
     print("HATA: Stoktaki hiçbir ürün, 70k'lık ana veride bulunamadı. Lütfen ürün isimlerini kontrol edin.")
     sys.exit()

koku_matrix_stok_only = tfidf_matrix[stok_indeksleri_all_data]
cosine_sim_reduced = cosine_similarity(tfidf_matrix, koku_matrix_stok_only)

# --- 5. KAYIT İŞLEMLERİ (PKL OLUŞTURMA) ---
joblib.dump(cosine_sim_reduced, 'cosine_sim_reduced.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(ana_db_df.reset_index(drop=True), 'all_perfumes_df.pkl')
joblib.dump(stok_indeksleri_all_data, 'stock_indices.pkl')

print(f"\n--- MODEL DOSYALARI ({len(stok_indeksleri_all_data)} adet ürün için) BAŞARIYLA OLUŞTURULDU ---")