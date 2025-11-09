import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# --- ADIM 1: "YAPAY ZEKA"NIN BEYNİ (76 PARFÜMLÜK TAM VERİTABANI) ---
# TÜM "kod" ALANLARI GÜNCELLENDİ (Örn: "LRN.09.008" -> "008")
parfum_veritabani_json = """
[
  {
    "kod": "008",
    "orijinal_ad": "Creed Aventus",
    "kategori": "Şipre, Meyveli, Taze",
    "notalar": ["Ananas", "Huş Ağacı", "Bergamot", "Siyah Frenk Üzümü", "Meşe Yosunu", "Misk", "Ambergris"]
  },
  {
    "kod": "010",
    "orijinal_ad": "Ex Nihilo Fleur Narcotique",
    "kategori": "Çiçeksi, Meyveli",
    "notalar": ["Liçi", "Şakayık", "Şeftali", "Portakal Çi çi", "Misk", "Yasemin"]
  },
  {
    "kod": "031",
    "orijinal_ad": "Memo Paris Marfa",
    "kategori": "Çiçeksi, Odunsu, Misk",
    "notalar": ["Sümbülteber", "Agave", "Vanilya", "Portakal Çiçeği", "Sandal Ağacı", "Beyaz Misk"]
  },
  {
    "kod": "049",
    "orijinal_ad": "Xerjoff Casamorati Lira",
    "kategori": "Amber, Gurme, Narenciyeli",
    "notalar": ["Karamel", "Vanilya", "Kan Portakalı", "Tarçın", "Lavanta", "Meyan Kökü"]
  },
  {
    "kod": "052",
    "orijinal_ad": "Tom Ford Lost Cherry",
    "kategori": "Amber, Gurme, Meyveli",
    "notalar": ["Vişne", "Acı Badem", "Likör", "Tonka Fasulyesi", "Vanilya", "Gül", "Yasemin"]
  },
  {
    "kod": "055",
    "orijinal_ad": "Xerjoff More Than Words",
    "kategori": "Amber, Odunsu, Baharatlı",
    "notalar": ["Ud", "Meyvemsi Notalar", "Amber", "Güve Otu", "Olibanum"]
  },
  {
    "kod": "078",
    "orijinal_ad": "Maison Francis Kurkdjian Baccarat Rouge 540 Extrait",
    "kategori": "Amber, Odunsu, Baharatlı",
    "notalar": ["Safran", "Acı Badem", "Mısır Yasemini", "Sedir", "Ambergris", "Misk"]
  },
  {
    "kod": "080",
    "orijinal_ad": "Marc-Antoine Barrois Ganymede",
    "kategori": "Odunsu, Baharatlı, Mineral",
    "notalar": ["Mineral Notalar", "Safran", "Menekşe Yaprağı", "Mandalina", "Ölümsüz Otu"]
  },
  {
    "kod": "085",
    "orijinal_ad": "Initio Oud for Greatness",
    "kategori": "Odunsu, Baharatlı, Oryantal",
    "notalar": ["Ud", "Safran", "Muskat", "Lavanta", "Paçuli"]
  },
  {
    "kod": "091",
    "orijinal_ad": "Nishane Hacivat",
    "kategori": "Şipre, Meyveli",
    "notalar": ["Ananas", "Greyfurt", "Meşe Yosunu", "Bergamot", "Odunsu Notalar", "Paçuli"]
  },
  {
    "kod": "092",
    "orijinal_ad": "Tom Ford Ombre Leather",
    "kategori": "Deri, Odunsu",
    "notalar": ["Deri", "Kakule", "Yasemin", "Amber", "Paçuli", "Yosun"]
  },
  {
    "kod": "099",
    "orijinal_ad": "Maison Francis Kurkdjian Oud Silk Mood",
    "kategori": "Amber, Çiçeksi, Odunsu",
    "notalar": ["Ud", "Bulgar Gülü", "Papatya", "Papirüs", "Guaiac Ağacı"]
  },
  {
    "kod": "102",
    "orijinal_ad": "Richard White Chocola",
    "kategori": "Gurme, Vanilya, Çiçeksi",
    "notalar": ["Beyaz Çikolata", "Vanilya", "Badem", "Şeftali", "Fındık", "Orkide"]
  },
  {
    "kod": "106",
    "orijinal_ad": "Tom Ford Electric Cherry",
    "kategori": "Çiçeksi, Meyveli",
    "notalar": ["Vişne", "Zencefil", "Yasemin", "Pembe Biber", "Misk"]
  },
  {
    "kod": "116",
    "orijinal_ad": "Tom Ford Vanilla Sex",
    "kategori": "Amber, Vanilya, Gurme",
    "notalar": ["Vanilya", "Acı Badem", "Sandal Ağacı", "Tonka Fasulyesi", "Çiçeksi Notalar"]
  },
  {
    "kod": "122",
    "orijinal_ad": "Parfums de Marly Layton",
    "kategori": "Amber, Çiçeksi, Baharatlı",
    "notalar": ["Elma", "Vanilya", "Lavanta", "Kakule", "Sandal Ağacı", "Bergamot"]
  },
  {
    "kod": "123",
    "orijinal_ad": "Montale Arabians Tonka",
    "kategori": "Amber, Gurme, Odunsu",
    "notalar": ["Tonka Fasulyesi", "Şeker Kamışı", "Safran", "Ud", "Gül", "Amber"]
  },
  {
    "kod": "127",
    "orijinal_ad": "Kayali Vanilla 28",
    "kategori": "Amber, Vanilya, Gurme",
    "notalar": ["Vanilya Orkidelesi", "Kahverengi Şeker", "Tonka Fasulyesi", "Amber", "Paçuli"]
  },
  {
    "kod": "128",
    "orijinal_ad": "Parfums de Marly Althair",
    "kategori": "Amber, Vanilya, Gurme",
    "notalar": ["Vanilya", "Tarçın", "Pralin", "Kakule", "Portakal Çiçeği", "Misk"]
  },
  {
    "kod": "206",
    "orijinal_ad": "Donna Karan Be Delicious Green",
    "kategori": "Taze, Çiçeksi, Meyveli",
    "notalar": ["Elma", "Salatalık", "Greyfurt", "Manolya", "Gül", "Sandal Ağacı", "Beyaz Amber"]
  },
  {
    "kod": "207",
    "orijinal_ad": "Giorgio Armani Acqua di Gio",
    "kategori": "Aromatik, Akuatik (Deniz), Taze",
    "notalar": ["Deniz Notaları", "Limon", "Bergamot", "Mandalina", "Yasemin", "Beyaz Misk", "Sedir"]
  },
  {
    "kod": "209",
    "orijinal_ad": "Giorgio Armani Si Parfum",
    "kategori": "Şipre, Meyveli, Vanilya",
    "notalar": ["Siyah Frenk Üzümü", "Gül", "Vanilya", "Paçuli", "Frezya", "Mandalina"]
  },
  {
    "kod": "215",
    "orijinal_ad": "Gucci by Flora",
    "kategori": "Çiçeksi, Meyveli, Tatlı",
    "notalar": ["Gardenya", "Armut Çiçeği", "Esmer Şeker", "Kırmızı Meyveler", "Paçuli", "Yasemin"]
  },
  {
    "kod": "217",
    "orijinal_ad": "Guerlain Robe Noir",
    "kategori": "Çiçeksi, Meyveli, Tatlı",
    "notalar": ["Vişne", "Gül", "Badem", "Siyah Frenk Üzümü", "Misk", "Paçuli"]
  },
  {
    "kod": "218",
    "orijinal_ad": "Hermes Terre de Hermes",
    "kategori": "Odunsu, Baharatlı, Narenciye",
    "notalar": ["Portakal", "Greyfurt", "Vetiver", "Biber", "Sedir", "Paçuli"]
  },
  {
    "kod": "222",
    "orijinal_ad": "Lacoste L.12.12 Blanc - White",
    "kategori": "Odunsu, Aromatik, Taze",
    "notalar": ["Greyfurt", "Kakule", "Sümbülteber", "Ylang-Ylang", "Süet", "Vetiver"]
  },
  {
    "kod": "224",
    "orijinal_ad": "Lacoste Pour Femme",
    "kategori": "Çiçeksi, Odunsu, Pudralı",
    "notalar": ["Frezya", "Karabiber", "Yasemin", "Süet", "Sedir Ağacı", "Heliotrop"]
  },
  {
    "kod": "225",
    "orijinal_ad": "Lancome Tresor La Nuit",
    "kategori": "Amber, Gurme, Vanilya",
    "notalar": ["Pralin", "Karamel", "Vanilya", "Orkide", "Gül", "Liçi", "Paçuli", "Kahve"]
  },
  {
    "kod": "226",
    "orijinal_ad": "Lancome La Vie Est Belle",
    "kategori": "Çiçeksi, Gurme, Tatlı",
    "notalar": ["İris", "Pralin", "Vanilya", "Paçuli", "Portakal Çiçeği", "Siyah Frenk Üzümü"]
  },
  {
    "kod": "229",
    "orijinal_ad": "Moschino Love Love",
    "kategori": "Çiçeksi, Odunsu, Narenciye",
    "notalar": ["Greyfurt", "Portakal", "Limon", "Şeker Kamışı", "Misk", "Sedir", "Kırmızı Frenk Üzümü"]
  },
  {
    "kod": "231",
    "orijinal_ad": "Paco Rabanne Invictus",
    "kategori": "Akuatik (Deniz), Odunsu, Taze",
    "notalar": ["Deniz Notaları", "Greyfurt", "Defne Yaprağı", "Ambergris", "Guaiac Ağacı", "Meşe Yosunu"]
  },
  {
    "kod": "233",
    "orijinal_ad": "Paco Rabanne Olympea",
    "kategori": "Amber, Çiçeksi, Gurme",
    "notalar": ["Tuzlu Vanilya", "Su Yasemini", "Mandalina", "Zambak", "Kaşmir Ağacı", "Ambergris"]
  },
  {
    "kod": "234",
    "orijinal_ad": "Paco Rabanne Lady Million",
    "kategori": "Çiçeksi, Meyveli, Tatlı",
    "notalar": ["Bal", "Paçuli", "Portakal Çiçeği", "Ahududu", "Yasemin", "Amber"]
  },
  {
    "kod": "238",
    "orijinal_ad": "Versace Eros",
    "kategori": "Aromatik, Fougère, Taze",
    "notalar": ["Nane", "Yeşil Elma", "Limon", "Tonka Fasulyesi", "Vanilya", "Amber", "Sedir"]
  },
  {
    "kod": "242",
    "orijinal_ad": "Yves Saint Laurent Black Opium",
    "kategori": "Amber, Gurme, Vanilya",
    "notalar": ["Kahve", "Vanilya", "Portakal Çiçeği", "Armut", "Yasemin", "Misk", "Sedir"]
  },
  {
    "kod": "248",
    "orijinal_ad": "Calvin Klein Euphoria",
    "kategori": "Amber, Çiçeksi, Meyveli",
    "notalar": ["Nar", "Siyah Orkide", "Lotus Çiçeği", "Amber", "Misk", "Paçuli", "Maun"]
  },
  {
    "kod": "249",
    "orijinal_ad": "Carrolina Herrera 212 Sexy Magnetik",
    "kategori": "Amber, Çiçeksi, Tatlı",
    "notalar": ["Pamuk Şekeri", "Pembe Biber", "Vanilya", "Misk", "Gardenya", "Sandal Ağacı", "Mandalina"]
  },
  {
    "kod": "251",
    "orijinal_ad": "Carrolina Herrera 212 Sexy",
    "kategori": "Amber, Çiçeksi, Tatlı",
    "notalar": ["Pamuk Şekeri", "Pembe Biber", "Vanilya", "Misk", "Gardenya", "Sandal Ağacı", "Mandalina"]
  },
  {
    "kod": "253",
    "orijinal_ad": "Chanel Bleu de Chanel",
    "kategori": "Aromatik, Odunsu, Taze",
    "notalar": ["Limon", "Bergamot", "Nane", "Zencefil", "Sandal Ağacı", "Sedir", "Amberwood"]
  },
  {
    "kod": "262",
    "orijinal_ad": "Chanel Mademoiselle",
    "kategori": "Amber, Çiçeksi, Şipre",
    "notalar": ["Portakal", "Bergamot", "Yasemin", "Gül", "Paçuli", "Beyaz Misk", "Vetiver"]
  },
  {
    "kod": "263",
    "orijinal_ad": "Chanel Chance Eau Tendre",
    "kategori": "Çiçeksi, Meyveli, Taze",
    "notalar": ["Ayva", "Greyfurt", "Yasemin", "Gül", "Beyaz Misk", "Amber"]
  },
  {
    "kod": "264",
    "orijinal_ad": "Chanel Chance Parfum",
    "kategori": "Şipre, Çiçeksi, Baharatlı",
    "notalar": ["Pembe Biber", "Yasemin", "Paçuli", "Amber", "Beyaz Misk", "Vanilya", "İris"]
  },
  {
    "kod": "270",
    "orijinal_ad": "Emporio Armani Stronger With You",
    "kategori": "Aromatik, Gurme, Vanilya",
    "notalar": ["Kestane", "
