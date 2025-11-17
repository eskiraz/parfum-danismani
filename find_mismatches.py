# find_mismatches.py
# BU KODUN TAMAMINI KOPYALAYIP find_mismatches.py DOSYASINA YAPIŞTIRIN

import pandas as pd
import re
import sys

# --- YARDIMCI FONKSİYON: AGRESİF İSİM TEMİZLİĞİ ---
def super_clean_name(name):
    """Bütün özel karakterleri, boşlukları kaldırır ve küçük harfe çevirir."""
    name = str(name).lower().strip()
    name = re.sub(r'[^a-z0-9]+', '', name) 
    return name

# --- 1. VERİ YÜKLEME ---
try:
    ana_db_df = pd.read_csv("ana_db_clean.csv")
    stok_db_df = pd.read_csv("stok_listesi_clean.csv")
except Exception as e:
    print(f"HATA: Gerekli CSV dosyaları bulunamadı. Detay: {e}")
    sys.exit()

# --- 2. SÜTUN VE İÇERİK TEMİZLİĞİ ---
# 1. Sütun adlarını sabitle
ana_db_df = ana_db_df.rename(columns={'Parfüm İsmi': 'isim'}) 
stok_db_df = stok_db_df.rename(columns={'orijinal_ad': 'isim'}) 

# 2. Ürün isimlerini eşleşme için temizle
ana_db_df['isim_clean'] = ana_db_df['isim'].apply(super_clean_name)
stok_db_df['isim_clean'] = stok_db_df['isim'].apply(super_clean_name)

# --- 3. KRİTİK VERİ HATASI TESPİTİ ---
print("--- KRİTİK VERİ HATASI TESPİTİ ---")

# 70k'lık ana veritabanındaki tüm temizlenmiş isimleri al
clean_main_names = set(ana_db_df['isim_clean'])

# Eşleşmeyen stok ürünlerini bul
mismatched_stock = stok_db_df[
    ~stok_db_df['isim_clean'].isin(clean_main_names)
]

if mismatched_stock.empty:
    print("HATA TESPİT EDİLEMEDİ: Eşleşen tüm ürünler bulundu. (Bu kez modelin oluşturulması gerekir)")
    # Model oluşturma kısmına geçer

else:
    print(f"HATA: {len(mismatched_stock)} adet stok ürünü 70k'lık ana veritabanında EŞLEŞMİYOR.")
    print("Bu ürünlerin isimleri İKİ DOSYADA FARKLI YAZILMIŞTIR. Lütfen MANUEL DÜZELTİN:")
    
    # Eşleşmeyen ilk 10 ürünü göster
    display_df = mismatched_stock[['kod', 'isim']].head(10).to_markdown(index=False, numalign="left", stralign="left")
    print("\nEşleşmeyen İlk 10 Stok Ürünü:")
    print(display_df)
    sys.exit() # Programı durdur