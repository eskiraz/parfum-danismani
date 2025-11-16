# Dosya Adı: model_olustur.py (v2)
# Amacı: RAM sorununu çözmek için matrisi 70k x 122'ye indirger.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib 
import sys

def create_model():
    print("Veri yükleniyor ve birleştiriliyor...")
    try:
        # 1. Veri Yükleme
        ana_db = pd.read_csv("ana_db_clean.csv")
        stok_db = pd.read_csv("stok_listesi_clean.csv")
        
        # 2. Birleştirme (Tip Kolonu Eklendi)
        ana_db_cols = ana_db.rename(columns={'Parfüm İsmi': 'isim'})[['isim', 'cinsiyet', 'notalar_str']]
        ana_db_cols['tip'] = 'Original'
        stok_db_cols = stok_db.rename(columns={'orijinal_ad': 'isim'})[['isim', 'cinsiyet', 'notalar_str']]
        stok_db_cols['tip'] = 'Stok'

        all_perfumes = pd.concat([ana_db_cols, stok_db_cols], ignore_index=True)
        all_perfumes['notalar_str'] = all_perfumes['notalar_str'].fillna('')
        all_perfumes['cinsiyet'] = all_perfumes['cinsiyet'].fillna('Unisex') # NaN'ları doldur
        
        # 3. Vectorizer'ı (Koku Evreni) Bütün Veriye Göre Kur
        print("Koku Evreni Vectorizer kuruluyor...")
        vectorizer = CountVectorizer(min_df=2, max_df=0.8)
        koku_matrix_all = vectorizer.fit_transform(all_perfumes['notalar_str'])

        # 4. Sadece Stoktaki Parfümlerin İndekslerini Belirle
        stock_indices_in_all = all_perfumes[all_perfumes['tip'] == 'Stok'].index.tolist()
        
        # 5. İndirgenmiş Benzerlik Matrisi Oluşturma (70k vs 122)
        print("İndirgenmiş Benzerlik Matrisi (70k vs 122) oluşturuluyor. (Bu 1-2 dakika sürebilir)...")
        
        # Sadece Stoktaki parfümlere karşılık gelen matrix satırları
        koku_matrix_stock_only = koku_matrix_all[stock_indices_in_all, :] 

        # Bütün Parfümlerin Benzerliğini, Sadece Stok Parfümlerine göre hesapla
        cosine_sim_reduced = cosine_similarity(koku_matrix_all, koku_matrix_stock_only) 
        
        # 6. Modeli Kaydetme
        joblib.dump(cosine_sim_reduced, 'cosine_sim_reduced.pkl')
        joblib.dump(vectorizer, 'vectorizer.pkl')
        joblib.dump(all_perfumes, 'all_perfumes_df.pkl')
        joblib.dump(stock_indices_in_all, 'stock_indices.pkl') # Stok index listesini de kaydet

        print("\n--- İNDİRGENMİŞ MODEL OLUŞTURMA TAMAMLANDI ---")

    except FileNotFoundError as e:
        print(f"HATA: Gerekli .csv dosyası bulunamadı.")
        sys.exit()
    except Exception as e:
        print(f"Kritik Model Oluşturma Hatası: {e}")
        sys.exit()

if __name__ == "__main__":
    create_model()