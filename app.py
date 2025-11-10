import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# --- ADIM 1: "YAPAY ZEKA"NIN BEYNİ (v2.7) ---
# VERİTABANI OLDUĞU GİBİ YÜKLENECEK. RESİM YOLLARI HİÇ OKUNMAYACAK.
parfum_veritabani_json = """
[
  {
    "kod": "002",
    "orijinal_ad": "Amouage Honour Man",
    "cinsiyet": "Erkek",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Baharatlı, Odunsu, Taze",
    "notalar": ["Pembe Biber", "Sardunya", "Elemi", "Muskat", "Tütsü", "Güve Otu", "Sedir", "Misk", "Tonka Fasulyesi", "Paçuli"]
  },
  {
    "kod": "008",
    "orijinal_ad": "Creed Aventus",
    "cinsiyet": "Erkek",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Şipre, Meyveli, Taze",
    "notalar": ["Ananas", "Huş Ağacı", "Bergamot", "Siyah Frenk Üzümü", "Meşe Yosunu", "Misk", "Ambergris"]
  },
  {
    "kod": "010",
    "orijinal_ad": "Ex Nihilo Fleur Narcotique",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Çiçeksi, Meyveli",
    "notalar": ["Liçi", "Şakayık", "Şeftali", "Portakal Çiçeği", "Misk", "Yasemin"]
  },
  {
    "kod": "012",
    "orijinal_ad": "Frederic Malle Portrait of a Lady",
    "cinsiyet": "Kadın",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Çiçeksi, Amber, Baharatlı",
    "notalar": ["Gül", "Karanfil", "Ahududu", "Siyah Frenk Üzümü", "Tarçın", "Paçuli", "Tütsü", "Sandal Ağacı", "Misk", "Amber", "Benzoin"]
  },
  {
    "kod": "013",
    "orijinal_ad": "Maison Francis Kurkdjian Baccarat Rouge 540",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Çiçeksi, Odunsu, Amber",
    "notalar": ["Safran", "Yasemin", "Amberwood", "Ambergris", "Reçine", "Sedir"]
  },
  {
    "kod": "021",
    "orijinal_ad": "Nasomatto Black Afgano",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Odunsu, Tütsü, Baharatlı",
    "notalar": ["Kenevir", "Yeşil Notalar", "Reçine", "Odunsu Notalar", "Tütün", "Kahve", "Ud", "Tütsü"]
  },
  {
    "kod": "024",
    "orijinal_ad": "Tom Ford Black Orchid",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Amber, Çiçeksi, Gurme",
    "notalar": ["Yasemin", "Gardenya", "Ylang Ylang", "Bergamot", "Frenk Üzümü", "Yumru", "Baharat", "Meyveli Notalar", "Orkide", "Vetiver", "Sandal Ağacı", "Paçuli", "Amber", "Tütsü", "Vanilya", "Çikolata"]
  },
  {
    "kod": "027",
    "orijinal_ad": "Xerjoff Erba Pura",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Narenciye, Meyveli, Misk",
    "notalar": ["Sicilya Portakalı", "Calabria Bergamotu", "Sicilya Limonu", "Tropikal Meyveler", "Beyaz Misk", "Amber", "Madagaskar Vanilyası"]
  },
  {
    "kod": "031",
    "orijinal_ad": "Memo Paris Marfa",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Çiçeksi, Odunsu, Misk",
    "notalar": ["Sümbülteber", "Agave", "Vanilya", "Portakal Çiçeği", "Sandal Ağacı", "Beyaz Misk"]
  },
  {
    "kod": "040",
    "orijinal_ad": "Parfums de Marly Delina",
    "cinsiyet": "Kadın",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Çiçeksi, Meyveli, Taze",
    "notalar": ["Liçi", "Rhubarb", "Bergamot", "Muskat", "Türk Gülü", "Şakayık", "Vanilya", "Kaşmir", "Sedir", "Vetiver", "Tütsü", "Misk"]
  },
  {
    "kod": "041",
    "orijinal_ad": "Zadig & Voltaire This is Her",
    "cinsiyet": "Kadın",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Odunsu, Vanilya, Gurme",
    "notalar": ["Yasemin", "Yumuşak Vanilya", "Kestane", "Sandal Ağacı"]
  },
  {
    "kod": "045",
    "orijinal_ad": "Gucci Intense Oud",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Ud, Amber, Oryantal",
    "notalar": ["Armut", "Ahududu", "Safran", "Bulgar Gülü", "Portakal Çiçeği", "Doğal Ud", "Paçuli"]
  },
  {
    "kod": "049",
    "orijinal_ad": "Xerjoff Casamorati Lira",
    "cinsiyet": "Kadın",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Amber, Gurme, Narenciyeli",
    "notalar": ["Karamel", "Vanilya", "Kan Portakalı", "Tarçın", "Lavanta", "Meyan Kökü"]
  },
  {
    "kod": "052",
    "orijinal_ad": "Tom Ford Lost Cherry",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Amber, Gurme, Meyveli",
    "notalar": ["Vişne", "Acı Badem", "Likör", "Tonka Fasulyesi", "Vanilya", "Gül", "Yasemin"]
  },
  {
    "kod": "055",
    "orijinal_ad": "Xerjoff More Than Words",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Amber, Odunsu, Baharatlı",
    "notalar": ["Ud", "Meyvemsi Notalar", "Amber", "Güve Otu", "Olibanum"]
  },
  {
    "kod": "068",
    "orijinal_ad": "Tom Ford Noir Extreme",
    "cinsiyet": "Erkek",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Amber, Baharatlı, Odunsu",
    "notalar": ["Kakule", "Muskat", "Safran", "Mandalina", "Neroli", "Gül", "Yasemin", "Damla Sakızı", "Vanilya", "Amber", "Odunsu Notalar", "Sandal Ağacı"]
  },
  {
    "kod": "078",
    "orijinal_ad": "Maison Francis Kurkdjian Baccarat Rouge 540 Extrait",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Amber, Odunsu, Baharatlı",
    "notalar": ["Safran", "Acı Badem", "Mısır Yasemini", "Sedir", "Ambergris", "Misk"]
  },
  {
    "kod": "079",
    "orijinal_ad": "Orto Parisi Megamare",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Aromatik, Akuatik (Deniz), Misk",
    "notalar": ["Bergamot", "Limon", "Yosun", "Calone", "Hedione", "Ambrox", "Sedir", "Misk"]
  },
  {
    "kod": "080",
    "orijinal_ad": "Marc-Antoine Barrois Ganymede",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Odunsu, Baharatlı, Mineral",
    "notalar": ["Mineral Notalar", "Safran", "Menekşe Yaprağı", "Mandalina", "Ölümsüz Otu"]
  },
  {
    "kod": "085",
    "orijinal_ad": "Initio Oud for Greatness",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Odunsu, Baharatlı, Oryantal",
    "notalar": ["Ud", "Safran", "Muskat", "Lavanta", "Paçuli"]
  },
  {
    "kod": "091",
    "orijinal_ad": "Nishane Hacivat",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Şipre, Meyveli",
    "notalar": ["Ananas", "Greyfurt", "Meşe Yosunu", "Bergamot", "Odunsu Notalar", "Paçuli"]
  },
  {
    "kod": "092",
    "orijinal_ad": "Tom Ford Ombre Leather",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Deri, Odunsu",
    "notalar": ["Deri", "Kakule", "Yasemin", "Amber", "Paçuli", "Yosun"]
  },
  {
    "kod": "099",
    "orijinal_ad": "Maison Francis Kurkdjian Oud Silk Mood",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Gül, Oud, Misk",
    "notalar": ["Gül", "Papatya", "Bergamot", "Hedione", "Guaiac Ağacı", "Oud", "Papirüs"]
  },
  {
    "kod": "102",
    "orijinal_ad": "Richard White Chocola",
    "cinsiyet": "Kadın",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Gurme, Vanilya, Çiçeksi",
    "notalar": ["Beyaz Çikolata", "Vanilya", "Badem", "Şeftali", "Fındık", "Orkide"]
  },
  {
    "kod": "106",
    "orijinal_ad": "Tom Ford Electric Cherry",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Meyveli, Çiçeksi, Misk",
    "notalar": ["Kiraz", "Zencefil", "Yasemin Sambac", "Ambrette", "Pembe Biber", "Misk", "Odunsu Notalar"]
  },
  {
    "kod": "114",
    "orijinal_ad": "Initio Musk Therapy",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Misk, Odunsu, Çiçeksi",
    "notalar": ["Bergamot", "Greyfurt", "Sedir Ağacı", "Gül", "Paçuli", "Sandal Ağacı", "Vanilya", "Amber", "Ambergris"]
  },
  {
    "kod": "116",
    "orijinal_ad": "Tom Ford Vanilla Sex",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Amber, Vanilya, Gurme",
    "notalar": ["Badem", "Yasemin", "Portakal Çiçeği", "Vanilya", "Sandal Ağacı", "Amber"]
  },
  {
    "kod": "117",
    "orijinal_ad": "Kilian Angels' Share",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Gurme, Amber, Baharatlı",
    "notalar": ["Konyak", "Tarçın", "Tonka Fasulyesi", "Meşe", "Pralin", "Vanilya", "Sandal Ağacı"]
  },
  {
    "kod": "120",
    "orijinal_ad": "Marc-Antoine Barrois Tilia",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Çiçeksi, Odunsu",
    "notalar": ["Lime", "Katırtırnağı", "Yasemin", "Vetiver", "Kediotu", "Sedir Ağacı", "Ambroxan"]
  },
  {
    "kod": "122",
    "orijinal_ad": "Parfums de Marly Layton",
    "cinsiyet": "Erkek",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Oryantal, Çiçeksi, Baharatlı",
    "notalar": ["Elma", "Bergamot", "Lavanta", "Yasemin", "Menekşe", "Gülhatmi", "Vanilya", "Biber", "Guaiac Ağacı", "Paçuli"]
  },
  {
    "kod": "123",
    "orijinal_ad": "Montale Arabians Tonka",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Oryantal, Odunsu, Tonka",
    "notalar": ["Safran", "Bergamot", "Ud", "Bulgar Gülü", "Tonka Fasulyesi", "Şeker Kamışı", "Amber", "Beyaz Misk", "Meşe Yosunu"]
  },
  {
    "kod": "124",
    "orijinal_ad": "Louis Vuitton Imagination",
    "cinsiyet": "Erkek",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Narenciye, Amber, Çay",
    "notalar": ["Ağaç Kavunu", "Bergamot", "Portakal", "Zencefil", "Neroli", "Tarçın", "Siyah Çay", "Ambroksan", "Olibanum", "Guaiac Ağacı"]
  },
  {
    "kod": "125",
    "orijinal_ad": "Amouage Guidance",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Çiçeksi, Baharatlı, Gourmand",
    "notalar": ["Armut", "Fındık Sütü", "Safran", "Gül", "Yasemin", "Osmanthus", "Sandal Ağacı", "Vanilya", "Deri", "Tütsü", "Ambergris"]
  },
  {
    "kod": "127",
    "orijinal_ad": "Kayali Vanilla",
    "cinsiyet": "Unisex",
    "resim_yolu": "eski/yol/onemsiz",
    "kategori": "Amber, Vanilya",
    "notalar": ["Vanilya", "Yasemin", "Orkide", "Esmer Şeker", "Tonka Fasulyesi", "Amber", "Misk", "Paçuli"]
  },
  {
    "kod": "128",
    "or
