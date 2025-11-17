# clean_data.py - Final Veri Temizliği ve Formatlama
# BU KODU KOPYALAYIP clean_data.py DOSYASINA YAPIŞTIRIN

import pandas as pd
import json
import sys

# --- 1. KAYNAK DOSYA YÜKLEME ---
try:
    # Ana veriyi yükle (Sizin ana_veri tabani.xlsx dosyanızı kullanır)
    ana_df = pd.read_excel("ana_veri tabani.xlsx") #
    # Stok listesini yükle (Sizin stok_veritabani.json dosyanızı kullanır)
    with open("stok_veritabani.json", 'r', encoding='utf-8') as f: #
        stok_json_data = json.load(f)
except FileNotFoundError:
    print("HATA: Kaynak dosyalar (ana_veri tabani.xlsx veya stok_veritabani.json) bulunamadı.")
    print("Lütfen ana klasörünüzde olduklarını kontrol edin.")
    sys.exit()

# --- 2. SÜTUN YAPISINI OLUŞTURMA VE İHRACAT ---

stok_df = pd.DataFrame(stok_json_data)

# 'notalar_str' sütununu oluştur (Kodun ihtiyacı olan temizlenmiş nota metni)
def create_note_string(notes_list):
    try:
        return ' '.join(notes_list).lower()
    except:
        return ""
        
stok_df['notalar_str'] = stok_df['notalar'].apply(create_note_string)


# Stok listesi için sadece istediğiniz sütunları seçerek yeni CSV'yi kaydetme
stok_df_final = stok_df[['kod', 'orijinal_ad', 'cinsiyet', 'kategori', 'notalar', 'notalar_str']]
# Kritik Adım: Ayraç olarak KESİNLİKLE virgül (,) ve utf-8 kodlamasını kullanır.
stok_df_final.to_csv("stok_listesi_clean.csv", index=False, encoding='utf-8', sep=',')

# Ana veriyi de aynı formatta kaydetme (Modelin çalışması için zorunlu)
ana_df_final = ana_df[['Parfüm İsmi', 'cinsiyet', 'notalar_str', 'aciklama']] #
ana_df_final.to_csv("ana_db_clean.csv", index=False, encoding='utf-8', sep=',')


print("--- VERİ TEMİZLİĞİ BAŞARILI ---")
print("stok_listesi_clean.csv ve ana_db_clean.csv başarıyla yeniden oluşturuldu. Sütunlar artık ayrıdır.")

#### 2. Yeni Script'i Çalıştırma

1.  **`parfum-danismani`** klasörünüzün adres çubuğuna **`cmd`** yazıp Komut İstemi'ni açın.
2.  Aşağıdaki komutu çalıştırın:
    ```bash
    py clean_data.py
    ```
    *Ekranda **"VERİ TEMİZLİĞİ BAŞARILI"** mesajını gördüğünüzde, Excel dosyanızı tekrar açıp sütunların artık ayrılmış olduğunu kontrol edebilirsiniz.*

### 3. Sonuç

Verileriniz düzeltildiğine göre, **`model_olustur.py`** script'i artık sütunları doğru okuyacak ve model dosyaları oluşacaktır.