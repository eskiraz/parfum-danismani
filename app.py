import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os 

# --- ADIM 1: "YAPAY ZEKA"NIN BEYNİ (v2.5 - SAĞLANAN RESİM SİSTEMİ) ---
# TÜM PARFÜMLERİN "resim_yolu" ALANI SİZİN SAĞLADIĞINIZ RESMİN URL'Sİ İLE DEĞİŞTİRİLDİ.
# Bu resim yapay zeka tarafından oluşturuldu ve herkese açık bir URL'ye sahip.
# SAĞLADIĞINIZ RESMİN URL'si BURAYA YAPIŞTIRILDI:
GENEL_RESIM_URL = "https://i.imgur.com/uQy5n8t.jpg" 

parfum_veritabani_json = """
[
  {
    "kod": "002",
    "orijinal_ad": "Amouage Honour Man",
    "cinsiyet": "Erkek",
    "resim_yolu": "resimler/002.jpg", 
    "kategori": "Baharatlı, Odunsu, Taze",
    "notalar": ["Pembe Biber", "Sardunya", "Elemi", "Muskat", "Tütsü", "Güve Otu", "Sedir", "Misk", "Tonka Fasulyesi", "Paçuli"]
  },
  {
    "kod": "008",
    "orijinal_ad": "Creed Aventus",
    "cinsiyet": "Erkek",
    "resim_yolu": "resimler/008.jpg", 
    "kategori": "Şipre, Meyveli, Taze",
    "notalar": ["Ananas", "Huş Ağacı", "Bergamot", "Siyah Frenk Üzümü", "Meşe Yosunu", "Misk", "Ambergris"]
  },
  {
    "kod": "010",
    "orijinal_ad": "Ex Nihilo Fleur Narcotique",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/010.jpg", 
    "kategori": "Çiçeksi, Meyveli",
    "notalar": ["Liçi", "Şakayık", "Şeftali", "Portakal Çiçeği", "Misk", "Yasemin"]
  },
  {
    "kod": "012",
    "orijinal_ad": "Frederic Malle Portrait of a Lady",
    "cinsiyet": "Kadın",
    "resim_yolu": "resimler/012.jpg", 
    "kategori": "Çiçeksi, Amber, Baharatlı",
    "notalar": ["Gül", "Karanfil", "Ahududu", "Siyah Frenk Üzümü", "Tarçın", "Paçuli", "Tütsü", "Sandal Ağacı", "Misk", "Amber", "Benzoin"]
  },
  {
    "kod": "013",
    "orijinal_ad": "Maison Francis Kurkdjian Baccarat Rouge 540",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/013.jpg", 
    "kategori": "Çiçeksi, Odunsu, Amber",
    "notalar": ["Safran", "Yasemin", "Amberwood", "Ambergris", "Reçine", "Sedir"]
  },
  {
    "kod": "021",
    "orijinal_ad": "Nasomatto Black Afgano",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/021.jpg", 
    "kategori": "Odunsu, Tütsü, Baharatlı",
    "notalar": ["Kenevir", "Yeşil Notalar", "Reçine", "Odunsu Notalar", "Tütün", "Kahve", "Ud", "Tütsü"]
  },
  {
    "kod": "024",
    "orijinal_ad": "Tom Ford Black Orchid",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/024.jpg", 
    "kategori": "Amber, Çiçeksi, Gurme",
    "notalar": ["Yasemin", "Gardenya", "Ylang Ylang", "Bergamot", "Frenk Üzümü", "Yumru", "Baharat", "Meyveli Notalar", "Orkide", "Vetiver", "Sandal Ağacı", "Paçuli", "Amber", "Tütsü", "Vanilya", "Çikolata"]
  },
  {
    "kod": "027",
    "orijinal_ad": "Xerjoff Erba Pura",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/027.jpg", 
    "kategori": "Narenciye, Meyveli, Misk",
    "notalar": ["Sicilya Portakalı", "Calabria Bergamotu", "Sicilya Limonu", "Tropikal Meyveler", "Beyaz Misk", "Amber", "Madagaskar Vanilyası"]
  },
  {
    "kod": "031",
    "orijinal_ad": "Memo Paris Marfa",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/031.jpg", 
    "kategori": "Çiçeksi, Odunsu, Misk",
    "notalar": ["Sümbülteber", "Agave", "Vanilya", "Portakal Çiçeği", "Sandal Ağacı", "Beyaz Misk"]
  },
  {
    "kod": "040",
    "orijinal_ad": "Parfums de Marly Delina",
    "cinsiyet": "Kadın",
    "resim_yolu": "resimler/040.jpg", 
    "kategori": "Çiçeksi, Meyveli, Taze",
    "notalar": ["Liçi", "Rhubarb", "Bergamot", "Muskat", "Türk Gülü", "Şakayık", "Vanilya", "Kaşmir", "Sedir", "Vetiver", "Tütsü", "Misk"]
  },
  {
    "kod": "041",
    "orijinal_ad": "Zadig & Voltaire This is Her",
    "cinsiyet": "Kadın",
    "resim_yolu": "resimler/041.jpg", 
    "kategori": "Odunsu, Vanilya, Gurme",
    "notalar": ["Yasemin", "Yumuşak Vanilya", "Kestane", "Sandal Ağacı"]
  },
  {
    "kod": "045",
    "orijinal_ad": "Gucci Intense Oud",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/045.jpg", 
    "kategori": "Ud, Amber, Oryantal",
    "notalar": ["Armut", "Ahududu", "Safran", "Bulgar Gülü", "Portakal Çiçeği", "Doğal Ud", "Paçuli"]
  },
  {
    "kod": "049",
    "orijinal_ad": "Xerjoff Casamorati Lira",
    "cinsiyet": "Kadın",
    "resim_yolu": "resimler/049.jpg", 
    "kategori": "Amber, Gurme, Narenciyeli",
    "notalar": ["Karamel", "Vanilya", "Kan Portakalı", "Tarçın", "Lavanta", "Meyan Kökü"]
  },
  {
    "kod": "052",
    "orijinal_ad": "Tom Ford Lost Cherry",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/052.jpg", 
    "kategori": "Amber, Gurme, Meyveli",
    "notalar": ["Vişne", "Acı Badem", "Likör", "Tonka Fasulyesi", "Vanilya", "Gül", "Yasemin"]
  },
  {
    "kod": "055",
    "orijinal_ad": "Xerjoff More Than Words",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/055.jpg", 
    "kategori": "Amber, Odunsu, Baharatlı",
    "notalar": ["Ud", "Meyvemsi Notalar", "Amber", "Güve Otu", "Olibanum"]
  },
  {
    "kod": "068",
    "orijinal_ad": "Tom Ford Noir Extreme",
    "cinsiyet": "Erkek",
    "resim_yolu": "resimler/068.jpg", 
    "kategori": "Amber, Baharatlı, Odunsu",
    "notalar": ["Kakule", "Muskat", "Safran", "Mandalina", "Neroli", "Gül", "Yasemin", "Damla Sakızı", "Vanilya", "Amber", "Odunsu Notalar", "Sandal Ağacı"]
  },
  {
    "kod": "078",
    "orijinal_ad": "Maison Francis Kurkdjian Baccarat Rouge 540 Extrait",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/078.jpg", 
    "kategori": "Amber, Odunsu, Baharatlı",
    "notalar": ["Safran", "Acı Badem", "Mısır Yasemini", "Sedir", "Ambergris", "Misk"]
  },
  {
    "kod": "079",
    "orijinal_ad": "Orto Parisi Megamare",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/079.jpg", 
    "kategori": "Aromatik, Akuatik (Deniz), Misk",
    "notalar": ["Bergamot", "Limon", "Yosun", "Calone", "Hedione", "Ambrox", "Sedir", "Misk"]
  },
  {
    "kod": "080",
    "orijinal_ad": "Marc-Antoine Barrois Ganymede",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/080.jpg", 
    "kategori": "Odunsu, Baharatlı, Mineral",
    "notalar": ["Mineral Notalar", "Safran", "Menekşe Yaprağı", "Mandalina", "Ölümsüz Otu"]
  },
  {
    "kod": "085",
    "orijinal_ad": "Initio Oud for Greatness",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/085.jpg", 
    "kategori": "Odunsu, Baharatlı, Oryantal",
    "notalar": ["Ud", "Safran", "Muskat", "Lavanta", "Paçuli"]
  },
  {
    "kod": "091",
    "orijinal_ad": "Nishane Hacivat",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/091.jpg", 
    "kategori": "Şipre, Meyveli",
    "notalar": ["Ananas", "Greyfurt", "Meşe Yosunu", "Bergamot", "Odunsu Notalar", "Paçuli"]
  },
  {
    "kod": "092",
    "orijinal_ad": "Tom Ford Ombre Leather",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/092.jpg", 
    "kategori": "Deri, Odunsu",
    "notalar": ["Deri", "Kakule", "Yasemin", "Amber", "Paçuli", "Yosun"]
  },
  {
    "kod": "099",
    "orijinal_ad": "Maison Francis Kurkdjian Oud Silk Mood",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/099.jpg", 
    "kategori": "Gül, Oud, Misk",
    "notalar": ["Gül", "Papatya", "Bergamot", "Hedione", "Guaiac Ağacı", "Oud", "Papirüs"]
  },
  {
    "kod": "102",
    "orijinal_ad": "Richard White Chocola",
    "cinsiyet": "Kadın",
    "resim_yolu": "resimler/102.jpg", 
    "kategori": "Gurme, Vanilya, Çiçeksi",
    "notalar": ["Beyaz Çikolata", "Vanilya", "Badem", "Şeftali", "Fındık", "Orkide"]
  },
  {
    "kod": "106",
    "orijinal_ad": "Tom Ford Electric Cherry",
    "cinsiyet": "Unisex",
    "resim_yolu": "resimler/10
