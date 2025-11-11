import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os 
import pandas as pd

# --- ADIM 1: VERİTABANI (v4.0 - stokta_mi ALANI EKLENDİ) ---
# "stokta_mi: true" -> Bizim satılık kodlu ürünlerimiz (119 adet)
# "stokta_mi: false" -> Müşterinin arayabileceği, notalarını bildiğimiz popüler parfümler (Örnek 5 adet eklendi)
parfum_veritabani_json = """
[
  {
    "kod": "002",
    "orijinal_ad": "Amouage Honour Man",
    "cinsiyet": "Erkek",
    "kategori": "Baharatlı, Odunsu, Taze",
    "notalar": ["Pembe Biber", "Sardunya", "Elemi", "Muskat", "Tütsü", "Güve Otu", "Sedir", "Misk", "Tonka Fasulyesi", "Paçuli"],
    "stokta_mi": true
  },
  {
    "kod": "008",
    "orijinal_ad": "Creed Aventus",
    "cinsiyet": "Erkek",
    "kategori": "Şipre, Meyveli, Taze",
    "notalar": ["Ananas", "Huş Ağacı", "Bergamot", "Siyah Frenk Üzümü", "Meşe Yosunu", "Misk", "Ambergris"],
    "stokta_mi": true
  },
  {
    "kod": "010",
    "orijinal_ad": "Ex Nihilo Fleur Narcotique",
    "cinsiyet": "Unisex",
    "kategori": "Çiçeksi, Meyveli",
    "notalar": ["Liçi", "Şakayık", "Şeftali", "Portakal Çiçeği", "Misk", "Yasemin"],
    "stokta_mi": true
  },
  {
    "kod": "012",
    "orijinal_ad": "Frederic Malle Portrait of a Lady",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Amber, Baharatlı",
    "notalar": ["Gül", "Karanfil", "Ahududu", "Siyah Frenk Üzümü", "Tarçın", "Paçuli", "Tütsü", "Sandal Ağacı", "Misk", "Amber", "Benzoin"],
    "stokta_mi": true
  },
  {
    "kod": "013",
    "orijinal_ad": "Maison Francis Kurkdjian Baccarat Rouge 540",
    "cinsiyet": "Unisex",
    "kategori": "Çiçeksi, Odunsu, Amber",
    "notalar": ["Safran", "Yasemin", "Amberwood", "Ambergris", "Reçine", "Sedir"],
    "stokta_mi": true
  },
  {
    "kod": "021",
    "orijinal_ad": "Nasomatto Black Afgano",
    "cinsiyet": "Unisex",
    "kategori": "Odunsu, Tütsü, Baharatlı",
    "notalar": ["Kenevir", "Yeşil Notalar", "Reçine", "Odunsu Notalar", "Tütün", "Kahve", "Ud", "Tütsü"],
    "stokta_mi": true
  },
  {
    "kod": "024",
    "orijinal_ad": "Tom Ford Black Orchid",
    "cinsiyet": "Unisex",
    "kategori": "Amber, Çiçeksi, Gurme",
    "notalar": ["Yasemin", "Gardenya", "Ylang Ylang", "Bergamot", "Frenk Üzümü", "Yumru", "Baharat", "Meyveli Notalar", "Orkide", "Vetiver", "Sandal Ağacı", "Paçuli", "Amber", "Tütsü", "Vanilya", "Çikolata"],
    "stokta_mi": true
  },
  {
    "kod": "027",
    "orijinal_ad": "Xerjoff Erba Pura",
    "cinsiyet": "Unisex",
    "kategori": "Narenciye, Meyveli, Misk",
    "notalar": ["Sicilya Portakalı", "Calabria Bergamotu", "Sicilya Limonu", "Tropikal Meyveler", "Beyaz Misk", "Amber", "Madagaskar Vanilyası"],
    "stokta_mi": true
  },
  {
    "kod": "031",
    "orijinal_ad": "Memo Paris Marfa",
    "cinsiyet": "Unisex",
    "kategori": "Çiçeksi, Odunsu, Misk",
    "notalar": ["Sümbülteber", "Agave", "Vanilya", "Portakal Çiçeği", "Sandal Ağacı", "Beyaz Misk"],
    "stokta_mi": true
  },
  {
    "kod": "040",
    "orijinal_ad": "Parfums de Marly Delina",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Meyveli, Taze",
    "notalar": ["Liçi", "Rhubarb", "Bergamot", "Muskat", "Türk Gülü", "Şakayık", "Vanilya", "Kaşmir", "Sedir", "Vetiver", "Tütsü", "Misk"],
    "stokta_mi": true
  },
  {
    "kod": "041",
    "orijinal_ad": "Zadig & Voltaire This is Her",
    "cinsiyet": "Kadın",
    "kategori": "Odunsu, Vanilya, Gurme",
    "notalar": ["Yasemin", "Yumuşak Vanilya", "Kestane", "Sandal Ağacı"],
    "stokta_mi": true
  },
  {
    "kod": "045",
    "orijinal_ad": "Gucci Intense Oud",
    "cinsiyet": "Unisex",
    "kategori": "Ud, Amber, Oryantal",
    "notalar": ["Armut", "Ahududu", "Safran", "Bulgar Gülü", "Portakal Çiçeği", "Doğal Ud", "Paçuli"],
    "stokta_mi": true
  },
  {
    "kod": "049",
    "orijinal_ad": "Xerjoff Casamorati Lira",
    "cinsiyet": "Kadın",
    "kategori": "Amber, Gurme, Narenciyeli",
    "notalar": ["Karamel", "Vanilya", "Kan Portakalı", "Tarçın", "Lavanta", "Meyan Kökü"],
    "stokta_mi": true
  },
  {
    "kod": "052",
    "orijinal_ad": "Tom Ford Lost Cherry",
    "cinsiyet": "Unisex",
    "kategori": "Amber, Gurme, Meyveli",
    "notalar": ["Vişne", "Acı Badem", "Likör", "Tonka Fasulyesi", "Vanilya", "Gül", "Yasemin"],
    "stokta_mi": true
  },
  {
    "kod": "055",
    "orijinal_ad": "Xerjoff More Than Words",
    "cinsiyet": "Unisex",
    "kategori": "Amber, Odunsu, Baharatlı",
    "notalar": ["Ud", "Meyvemsi Notalar", "Amber", "Güve Otu", "Olibanum"],
    "stokta_mi": true
  },
  {
    "kod": "068",
    "orijinal_ad": "Tom Ford Noir Extreme",
    "cinsiyet": "Erkek",
    "kategori": "Amber, Baharatlı, Odunsu",
    "notalar": ["Kakule", "Muskat", "Safran", "Mandalina", "Neroli", "Gül", "Yasemin", "Damla Sakızı", "Vanilya", "Amber", "Odunsu Notalar", "Sandal Ağacı"],
    "stokta_mi": true
  },
  {
    "kod": "078",
    "orijinal_ad": "Maison Francis Kurkdjian Baccarat Rouge 540 Extrait",
    "cinsiyet": "Unisex",
    "kategori": "Amber, Odunsu, Baharatlı",
    "notalar": ["Safran", "Acı Badem", "Mısır Yasemini", "Sedir", "Ambergris", "Misk"],
    "stokta_mi": true
  },
  {
    "kod": "079",
    "orijinal_ad": "Orto Parisi Megamare",
    "cinsiyet": "Unisex",
    "kategori": "Aromatik, Akuatik (Deniz), Misk",
    "notalar": ["Bergamot", "Limon", "Yosun", "Calone", "Hedione", "Ambrox", "Sedir", "Misk"],
    "stokta_mi": true
  },
  {
    "kod": "080",
    "orijinal_ad": "Marc-Antoine Barrois Ganymede",
    "cinsiyet": "Unisex",
    "kategori": "Odunsu, Baharatlı, Mineral",
    "notalar": ["Mineral Notalar", "Safran", "Menekşe Yaprağı", "Mandalina", "Ölümsüz Otu"],
    "stokta_mi": true
  },
  {
    "kod": "085",
    "orijinal_ad": "Initio Oud for Greatness",
    "cinsiyet": "Unisex",
    "kategori": "Odunsu, Baharatlı, Oryantal",
    "notalar": ["Ud", "Safran", "Muskat", "Lavanta", "Paçuli"],
    "stokta_mi": true
  },
  {
    "kod": "091",
    "orijinal_ad": "Nishane Hacivat",
    "cinsiyet": "Unisex",
    "kategori": "Şipre, Meyveli",
    "notalar": ["Ananas", "Greyfurt", "Meşe Yosunu", "Bergamot", "Odunsu Notalar", "Paçuli"],
    "stokta_mi": true
  },
  {
    "kod": "092",
    "orijinal_ad": "Tom Ford Ombre Leather",
    "cinsiyet": "Unisex",
    "kategori": "Deri, Odunsu",
    "notalar": ["Deri", "Kakule", "Yasemin", "Amber", "Paçuli", "Yosun"],
    "stokta_mi": true
  },
  {
    "kod": "099",
    "orijinal_ad": "Maison Francis Kurkdjian Oud Silk Mood",
    "cinsiyet": "Unisex",
    "kategori": "Gül, Oud, Misk",
    "notalar": ["Gül", "Papatya", "Bergamot", "Hedione", "Guaiac Ağacı", "Oud", "Papirüs"],
    "stokta_mi": true
  },
  {
    "kod": "102",
    "orijinal_ad": "Richard White Chocola",
    "cinsiyet": "Kadın",
    "kategori": "Gurme, Vanilya, Çiçeksi",
    "notalar": ["Beyaz Çikolata", "Vanilya", "Badem", "Şeftali", "Fındık", "Orkide"],
    "stokta_mi": true
  },
  {
    "kod": "106",
    "orijinal_ad": "Tom Ford Electric Cherry",
    "cinsiyet": "Unisex",
    "kategori": "Meyveli, Çiçeksi, Misk",
    "notalar": ["Kiraz", "Zencefil", "Yasemin Sambac", "Ambrette", "Pembe Biber", "Misk", "Odunsu Notalar"],
    "stokta_mi": true
  },
  {
    "kod": "114",
    "orijinal_ad": "Initio Musk Therapy",
    "cinsiyet": "Unisex",
    "kategori": "Misk, Odunsu, Çiçeksi",
    "notalar": ["Bergamot", "Greyfurt", "Sedir Ağacı", "Gül", "Paçuli", "Sandal Ağacı", "Vanilya", "Amber", "Ambergris"],
    "stokta_mi": true
  },
  {
    "kod": "116",
    "orijinal_ad": "Tom Ford Vanilla Sex",
    "cinsiyet": "Unisex",
    "kategori": "Amber, Vanilya, Gurme",
    "notalar": ["Badem", "Yasemin", "Portakal Çiçeği", "Vanilya", "Sandal Ağacı", "Amber"],
    "stokta_mi": true
  },
  {
    "kod": "117",
    "orijinal_ad": "Kilian Angels' Share",
    "cinsiyet": "Unisex",
    "kategori": "Gurme, Amber, Baharatlı",
    "notalar": ["Konyak", "Tarçın", "Tonka Fasulyesi", "Meşe", "Pralin", "Vanilya", "Sandal Ağacı"],
    "stokta_mi": true
  },
  {
    "kod": "120",
    "orijinal_ad": "Marc-Antoine Barrois Tilia",
    "cinsiyet": "Unisex",
    "kategori": "Çiçeksi, Odunsu",
    "notalar": ["Lime", "Katırtırnağı", "Yasemin", "Vetiver", "Kediotu", "Sedir Ağacı", "Ambroxan"],
    "stokta_mi": true
  },
  {
    "kod": "122",
    "orijinal_ad": "Parfums de Marly Layton",
    "cinsiyet": "Erkek",
    "kategori": "Oryantal, Çiçeksi, Baharatlı",
    "notalar": ["Elma", "Bergamot", "Lavanta", "Yasemin", "Menekşe", "Gülhatmi", "Vanilya", "Biber", "Guaiac Ağacı", "Paçuli"],
    "stokta_mi": true
  },
  {
    "kod": "123",
    "orijinal_ad": "Montale Arabians Tonka",
    "cinsiyet": "Unisex",
    "kategori": "Oryantal, Odunsu, Tonka",
    "notalar": ["Safran", "Bergamot", "Ud", "Bulgar Gülü", "Tonka Fasulyesi", "Şeker Kamışı", "Amber", "Beyaz Misk", "Meşe Yosunu"],
    "stokta_mi": true
  },
  {
    "kod": "124",
    "orijinal_ad": "Louis Vuitton Imagination",
    "cinsiyet": "Erkek",
    "kategori": "Narenciye, Amber, Çay",
    "notalar": ["Ağaç Kavunu", "Bergamot", "Portakal", "Zencefil", "Neroli", "Tarçın", "Siyah Çay", "Ambroksan", "Olibanum", "Guaiac Ağacı"],
    "stokta_mi": true
  },
  {
    "kod": "125",
    "orijinal_ad": "Amouage Guidance",
    "cinsiyet": "Unisex",
    "kategori": "Çiçeksi, Baharatlı, Gourmand",
    "notalar": ["Armut", "Fındık Sütü", "Safran", "Gül", "Yasemin", "Osmanthus", "Sandal Ağacı", "Vanilya", "Deri", "Tütsü", "Ambergris"],
    "stokta_mi": true
  },
  {
    "kod": "127",
    "orijinal_ad": "Kayali Vanilla",
    "cinsiyet": "Unisex",
    "kategori": "Amber, Vanilya",
    "notalar": ["Vanilya", "Yasemin", "Orkide", "Esmer Şeker", "Tonka Fasulyesi", "Amber", "Misk", "Paçuli"],
    "stokta_mi": true
  },
  {
    "kod": "128",
    "orijinal_ad": "Parfums de Marly Althair",
    "cinsiyet": "Erkek",
    "kategori": "Vanilya, Baharatlı, Odunsu",
    "notalar": ["Portakal Çiçeği", "Bergamot", "Tarçın", "Bourbon Vanilya", "Elemi", "Guaiac Wood", "Ambrox", "Pralin", "Misk"],
    "stokta_mi": true
  },
  {
    "kod": "134",
    "orijinal_ad": "Louis Vuitton L'Immensité",
    "cinsiyet": "Erkek",
    "kategori": "Akuatik, Aromatik, Narenciye",
    "notalar": ["Greyfurt", "Zencefil", "Bergamot", "Su Notaları", "Adaçayı", "Biberiye", "Ambroxan", "Kehribar", "Labdanum"],
    "stokta_mi": true
  },
  {
    "kod": "202",
    "orijinal_ad": "Dolce & Gabbana The One EDP",
    "cinsiyet": "Erkek",
    "kategori": "Oryantal, Baharatlı, Odunsu",
    "notalar": ["Greyfurt", "Kişniş", "Fesleğen", "Zencefil", "Kakule", "Portakal Çiçeği", "Tütün", "Amber", "Sedir Ağacı"],
    "stokta_mi": true
  },
  {
    "kod": "206",
    "orijinal_ad": "Donna Karan Be Delicious Green",
    "cinsiyet": "Kadın",
    "kategori": "Taze, Çiçeksi, Meyveli",
    "notalar": ["Elma", "Salatalık", "Greyfurt", "Manolya", "Gül", "Sandal Ağacı", "Beyaz Amber"],
    "stokta_mi": true
  },
  {
    "kod": "207",
    "orijinal_ad": "Giorgio Armani Acqua di Gio",
    "cinsiyet": "Erkek",
    "kategori": "Aromatik, Akuatik (Deniz), Taze",
    "notalar": ["Deniz Notaları", "Limon", "Bergamot", "Mandalina", "Yasemin", "Beyaz Misk", "Sedir"],
    "stokta_mi": true
  },
  {
    "kod": "208",
    "orijinal_ad": "Giorgio Armani Code Profumo",
    "cinsiyet": "Erkek",
    "kategori": "Amber, Baharatlı, Odunsu",
    "notalar": ["Tonka Fasulyesi", "Kakule", "Odunsu Notalar"],
    "stokta_mi": true
  },
  {
    "kod": "209",
    "orijinal_ad": "Giorgio Armani Si Parfum",
    "cinsiyet": "Kadın",
    "kategori": "Şipre, Meyveli, Vanilya",
    "notalar": ["Siyah Frenk Üzümü", "Gül", "Vanilya", "Paçuli", "Frezya", "Mandalina"],
    "stokta_mi": true
  },
  {
    "kod": "210",
    "orijinal_ad": "Giorgio Armani Si Intense",
    "cinsiyet": "Kadın",
    "kategori": "Amber, Çiçeksi, Vanilya",
    "notalar": ["Siyah Frenk Üzümü", "Gül", "Davana", "Vanilya", "Siyah Çay", "Paçuli"],
    "stokta_mi": true
  },
  {
    "kod": "211",
    "orijinal_ad": "Giorgio Armani Code for Women",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Oryantal",
    "notalar": ["Zambak", "Yasemin", "Taze Zencefil", "Portakal Çiçeği", "Vanilya", "Sandal Ağacı"],
    "stokta_mi": true
  },
  {
    "kod": "215",
    "orijinal_ad": "Gucci by Flora",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Meyveli, Tatlı",
    "notalar": ["Gardenya", "Armut Çiçeği", "Esmer Şeker", "Kırmızı Meyveler", "Paçuli", "Yasemin"],
    "stokta_mi": true
  },
  {
    "kod": "217",
    "orijinal_ad": "Guerlain Robe Noir",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Meyveli, Tatlı",
    "notalar": ["Vişne", "Gül", "Badem", "Siyah Frenk Üzümü", "Misk", "Paçuli"],
    "stokta_mi": true
  },
  {
    "kod": "218",
    "orijinal_ad": "Hermes Terre de Hermes",
    "cinsiyet": "Erkek",
    "kategori": "Odunsu, Baharatlı, Narenciye",
    "notalar": ["Portakal", "Greyfurt", "Vetiver", "Biber", "Sedir", "Paçuli"],
    "stokta_mi": true
  },
  {
    "kod": "222",
    "orijinal_ad": "Lacoste L.12.12 Blanc - White",
    "cinsiyet": "Erkek",
    "kategori": "Odunsu, Aromatik, Taze",
    "notalar": ["Greyfurt", "Kakule", "Sümbülteber", "Ylang-Ylang", "Süet", "Vetiver"],
    "stokta_mi": true
  },
  {
    "kod": "224",
    "orijinal_ad": "Lacoste Pour Femme",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Odunsu, Pudralı",
    "notalar": ["Frezya", "Karabiber", "Yasemin", "Süet", "Sedir Ağacı", "Heliotrop"],
    "stokta_mi": true
  },
  {
    "kod": "225",
    "orijinal_ad": "Lancome Tresor La Nuit",
    "cinsiyet": "Kadın",
    "kategori": "Amber, Gurme, Vanilya",
    "notalar": ["Pralin", "Karamel", "Vanilya", "Orkide", "Gül", "Liçi", "Paçuli", "Kahve"],
    "stokta_mi": true
  },
  {
    "kod": "226",
    "orijinal_ad": "Lancome La Vie Est Belle",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Gurme, Tatlı",
    "notalar": ["İris", "Pralin", "Vanilya", "Paçuli", "Portakal Çiçeği", "Siyah Frenk Üzümü"],
    "stokta_mi": true
  },
  {
    "kod": "229",
    "orijinal_ad": "Moschino Love Love",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Odunsu, Narenciye",
    "notalar": ["Greyfurt", "Portakal", "Limon", "Şeker Kamışı", "Misk", "Sedir", "Kırmızı Frenk Üzümü"],
    "stokta_mi": true
  },
  {
    "kod": "231",
    "orijinal_ad": "Paco Rabanne Invictus",
    "cinsiyet": "Erkek",
    "kategori": "Akuatik (Deniz), Odunsu, Taze",
    "notalar": ["Deniz Notaları", "Greyfurt", "Defne Yaprağı", "Ambergris", "Guaiac Ağacı", "Meşe Yosunu"],
    "stokta_mi": true
  },
  {
    "kod": "233",
    "orijinal_ad": "Paco Rabanne Olympea",
    "cinsiyet": "Kadın",
    "kategori": "Amber, Çiçeksi, Gurme",
    "notalar": ["Tuzlu Vanilya", "Su Yasemini", "Mandalina", "Zambak", "Kaşmir Ağacı", "Ambergris"],
    "stokta_mi": true
  },
  {
    "kod": "234",
    "orijinal_ad": "Paco Rabanne Lady Million",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Meyveli, Tatlı",
    "notalar": ["Bal", "Paçuli", "Portakal Çiçeği", "Ahududu", "Yasemin", "Amber"],
    "stokta_mi": true
  },
  {
    "kod": "235",
    "orijinal_ad": "Thierry Mugler Alien",
    "cinsiyet": "Kadın",
    "kategori": "Odunsu, Beyaz Çiçek, Amber",
    "notalar": ["Yasemin", "Kaşmir", "Beyaz Amber", "Odunsu Notalar"],
    "stokta_mi": true
  },
  {
    "kod": "238",
    "orijinal_ad": "Versace Eros",
    "cinsiyet": "Erkek",
    "kategori": "Aromatik, Fougère, Taze",
    "notalar": ["Nane", "Yeşil Elma", "Limon", "Tonka Fasulyesi", "Vanilya", "Amber", "Sedir"],
    "stokta_mi": true
  },
  {
    "kod": "241",
    "orijinal_ad": "Versace Crystal Noir",
    "cinsiyet": "Kadın",
    "kategori": "Baharatlı, Çiçeksi, Amber",
    "notalar": ["Kakule", "Karabiber", "Zencefil", "Gardenya", "Hindistan Cevizi", "Amber", "Misk"],
    "stokta_mi": true
  },
  {
    "kod": "242",
    "orijinal_ad": "Yves Saint Laurent Black Opium",
    "cinsiyet": "Kadın",
    "kategori": "Amber, Gurme, Vanilya",
    "notalar": ["Kahve", "Vanilya", "Portakal Çiçeği", "Armut", "Yasemin", "Misk", "Sedir"],
    "stokta_mi": true
  },
  {
    "kod": "243",
    "orijinal_ad": "Carolina Herrera 212 VIP",
    "cinsiyet": "Kadın",
    "kategori": "Vanilya, Rom, Gurme",
    "notalar": ["Rom", "Vanilya", "Çarkıfelek", "Tonka Fasulyesi", "Gardenya", "Misk"],
    "stokta_mi": true
  },
  {
    "kod": "246",
    "orijinal_ad": "Bvlgari Aqva Pour Homme",
    "cinsiyet": "Erkek",
    "kategori": "Akuatik (Deniz), Aromatik, Taze",
    "notalar": ["Deniz Yosunu", "Mandalina", "Pamuk Çiçeği", "Sedir", "Amber"],
    "stokta_mi": true
  },
  {
    "kod": "248",
    "orijinal_ad": "Calvin Klein Euphoria",
    "cinsiyet": "Kadın",
    "kategori": "Amber, Çiçeksi, Meyveli",
    "notalar": ["Nar", "Siyah Orkidesi", "Lotus Çiçeği", "Amber", "Misk", "Paçuli", "Maun"],
    "stokta_mi": true
  },
  {
    "kod": "249",
    "orijinal_ad": "Carrolina Herrera 212 Sexy Magnetik",
    "cinsiyet": "Kadın",
    "kategori": "Amber, Çiçeksi, Tatlı",
    "notalar": ["Pamuk Şekeri", "Pembe Biber", "Vanilya", "Misk", "Gardenya", "Sandal Ağacı", "Mandalina"],
    "stokta_mi": true
  },
  {
    "kod": "251",
    "orijinal_ad": "Carolina Herrera 212 Sexy",
    "cinsiyet": "Kadın",
    "kategori": "Oryantal, Çiçeksi, Tatlı",
    "notalar": ["Gül", "Biber", "Bergamot", "Gardenya", "Sardunya", "Pamuk Şekeri", "Vanilya", "Baharat"],
    "stokta_mi": true
  },
  {
    "kod": "253",
    "orijinal_ad": "Chanel Bleu de Chanel (EDT)",
    "cinsiyet": "Erkek",
    "kategori": "Aromatik, Odunsu, Amber",
    "notalar": ["Limon", "Bergamot", "Nane", "Pelin Otu", "Lavanta", "Sardunya", "Ananas", "Sandal Ağacı", "Sedir", "Amberwood", "Tonka Fasulyesi"],
    "stokta_mi": true
  },
  {
    "kod": "255",
    "orijinal_ad": "Christian Dior J'adore",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Meyveli",
    "notalar": ["Ylang-Ylang", "Yasemin", "Gül", "Şeftali", "Armut", "Misk", "Sedir"],
    "stokta_mi": true
  },
  {
    "kod": "256",
    "orijinal_ad": "Christian Dior Addict",
    "cinsiyet": "Kadın",
    "kategori": "Amber, Çiçeksi, Vanilya",
    "notalar": ["Vanilya", "Tonka Fasulyesi", "Yasemin", "Portakal Çiçeği", "Sandal Ağacı", "Bourbon Vanilyası"],
    "stokta_mi": true
  },
  {
    "kod": "260",
    "orijinal_ad": "Christian Dior Homme Intense",
    "cinsiyet": "Erkek",
    "kategori": "Odunsu, Çiçeksi, Misk",
    "notalar": ["İris", "Lavanta", "Sedir", "Vetiver", "Kakao", "Amber"],
    "stokta_mi": true
  },
  {
    "kod": "261",
    "orijinal_ad": "Christian Dior Fahrenheit",
    "cinsiyet": "Erkek",
    "kategori": "Deri, Aromatik, Odunsu",
    "notalar": ["Menekşe Yaprağı", "Deri", "Muskat", "Sedir", "Vetiver", "Lavanta"],
    "stokta_mi": true
  },
  {
    "kod": "262",
    "orijinal_ad": "Chanel Coco Mademoiselle",
    "cinsiyet": "Kadın",
    "kategori": "Şipre, Çiçeksi, Narenciye",
    "notalar": ["Narenciye", "Portakal", "Bergamot", "Yasemin", "Gül", "Liçi", "Amber", "Beyaz Misk", "Vetiver", "Paçuli"],
    "stokta_mi": true
  },
  {
    "kod": "263",
    "orijinal_ad": "Chanel Chance Eau Tendre",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Meyveli",
    "notalar": ["Greyfurt", "Ayva", "Yasemin", "Gül", "Beyaz Misk", "Hafif Odunsu Notalar", "Amber"],
    "stokta_mi": true
  },
  {
    "kod": "264",
    "orijinal_ad": "Chanel Chance Eau de Parfum",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Baharatlı, Amber",
    "notalar": ["Pembe Biber", "Yasemin", "Ambersi Paçuli", "Beyaz Misk", "Vanilya"],
    "stokta_mi": true
  },
  {
    "kod": "265",
    "orijinal_ad": "Chanel No. 5",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Aldehit, Sabunsu",
    "notalar": ["Aldehitler", "Ylang-Ylang", "Neroli", "Gül", "Yasemin", "Sandal Ağacı", "Vanilya", "Amber"],
    "stokta_mi": true
  },
  {
    "kod": "267",
    "orijinal_ad": "Chloé Eau de Parfum",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Gül, Pudralı",
    "notalar": ["Şakayık", "Liçi", "Gül", "Manolya", "Sedir", "Amber"],
    "stokta_mi": true
  },
  {
    "kod": "268",
    "orijinal_ad": "Chanel Egoiste",
    "cinsiyet": "Erkek",
    "kategori": "Odunsu, Baharatlı, Sandal Ağacı",
    "notalar": ["Sandal Ağacı", "Gül", "Tarçın", "Vanilya", "Tütün", "Limon"],
    "stokta_mi": true
  },
  {
    "kod": "270",
    "orijinal_ad": "Emporio Armani Stronger With You",
    "cinsiyet": "Erkek",
    "kategori": "Odunsu, Aromatik, Baharatlı",
    "notalar": ["Nane", "Menekşe Yaprağı", "Pembe Biber", "Kakule", "Tarçın", "Lavanta", "Ananas", "Kavun", "Adaçayı", "Amber", "Sedir", "Kestane", "Vanilya"],
    "stokta_mi": true
  },
  {
    "kod": "271",
    "orijinal_ad": "YSL Libre",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Odunsu, Misk",
    "notalar": ["Mandalina Yağı", "Tahıl Yağı", "Fransız Lavanta Yağı", "Kuşüzümü", "Lavanta Yağı", "Zambak", "Yasemin", "Portakal Çiçeği", "Vanilya Özü", "Sedir Ağacı Yağı", "Amber", "Misk"],
    "stokta_mi": true
  },
  {
    "kod": "274",
    "orijinal_ad": "Burberry Classic",
    "cinsiyet": "Kadın",
    "kategori": "Meyveli, Çiçeksi, Odunsu",
    "notalar": ["Yeşil Elma", "Bergamot", "Şeftali", "Kayısı", "Erik", "Yasemin", "Sandal Ağacı", "Sedir", "Misk", "Vanilya"],
    "stokta_mi": true
  },
  {
    "kod": "275",
    "orijinal_ad": "Burberry Classic Men",
    "cinsiyet": "Erkek",
    "kategori": "Odunsu, Aromatik",
    "notalar": ["Bergamot", "Taze Nane", "Lavanta", "Dağ Kekiği", "Itır Çiçeği", "Sandal Ağacı", "Amber", "Sedir"],
    "stokta_mi": true
  },
  {
    "kod": "276",
    "orijinal_ad": "Chloé Love",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Baharatlı",
    "notalar": ["Mor Salkımlı Sümbüller", "Leylaklar", "Portakal Çiçeği", "Sıcak Baharatlar"],
    "stokta_mi": true
  },
  {
    "kod": "278",
    "orijinal_ad": "Paco Rabanne Black XS for Him",
    "cinsiyet": "Erkek",
    "kategori": "Oryantal, Odunsu, Tatlı",
    "notalar": ["Turunçgiller", "Limon", "Adaçayı", "Kadife Çiçeği", "Pralin", "Tarçın", "Tolu Balsamı", "Siyah Kakule", "Paçuli", "Siyah Kehribar", "Abanoz Ağacı", "Palisander Gül Ağacı"],
    "stokta_mi": true
  },
  {
    "kod": "281",
    "orijinal_ad": "Giorgio Armani Sì Passione",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Meyveli, Tatlı",
    "notalar": ["Ananas", "Gül", "Armut", "Vanilya", "Sedir", "Amberwood"],
    "stokta_mi": true
  },
  {
    "kod": "282",
    "orijinal_ad": "Gucci Guilty Pour Homme",
    "cinsiyet": "Erkek",
    "kategori": "Odunsu, Aromatik, Taze",
    "notalar": ["Limon", "Lavanta", "Neroli", "Sedir", "Paçuli", "Amber"],
    "stokta_mi": true
  },
  {
    "kod": "284",
    "orijinal_ad": "Givenchy Insensé Ultramarine",
    "cinsiyet": "Erkek",
    "kategori": "Akuatik (Deniz), Taze, Meyveli",
    "notalar": ["Kırmızı Meyveler", "Deniz Notaları", "Nane", "Manolya", "Vetiver", "Tütün"],
    "stokta_mi": true
  },
  {
    "kod": "285",
    "orijinal_ad": "Bvlgari Man in Black",
    "cinsiyet": "Erkek",
    "kategori": "Amber, Baharatlı, Deri",
    "notalar": ["Baharatlar", "Rom", "Tütün", "Deri", "İris", "Sümbülteber", "Tonka Fasulyesi", "Guaiac Ağacı", "Benzoin"],
    "stokta_mi": true
  },
  {
    "kod": "286",
    "orijinal_ad": "Narciso Rodriguez For Her",
    "cinsiyet": "Kadın",
    "kategori": "Misk, Çiçeksi, Odunsu",
    "notalar": ["Vişne", "Erik", "Frezya", "Orkide", "İris", "Vanilya", "Misk", "Amber"],
    "stokta_mi": true
  },
  {
    "kod": "288",
    "orijinal_ad": "Jean Paul Gaultier Le Male",
    "cinsiyet": "Erkek",
    "kategori": "Aromatik, Odunsu, Tatlı",
    "notalar": ["Kakule", "Lavanta", "İris", "Vanilya", "Doğu Notaları", "Odunsu Notalar"],
    "stokta_mi": true
  },
  {
    "kod": "289",
    "orijinal_ad": "Carolina Herrera 212 Men",
    "cinsiyet": "Erkek",
    "kategori": "Odunsu, Aromatik, Baharatlı",
    "notalar": ["Narenciye Yaprakları", "Kesik Çim", "Baharat Yaprakları", "Taze Biber", "Zencefil", "Gardenya", "Sandal Ağacı", "Gayak Ağacı", "Tütsülenmiş Beyaz Misk"],
    "stokta_mi": true
  },
  {
    "kod": "291",
    "orijinal_ad": "Rochas Femme",
    "cinsiyet": "Kadın",
    "kategori": "Şipre, Meyveli, Baharatlı",
    "notalar": ["Erik", "Şeftali", "Tarçın", "Karanfil", "Gül", "Meşe Yosunu", "Amber", "Misk"],
    "stokta_mi": true
  },
  {
    "kod": "292",
    "orijinal_ad": "Victoria's Secret Bombshell",
    "cinsiyet": "Kadın",
    "kategori": "Meyveli, Çiçeksi",
    "notalar": ["Çarkıfelek Meyvesi", "Greyfurt", "Ananas", "Mandalina", "Çilek", "Şakayık", "Vanilya Orkidesi", "Kırmızı Meyveler", "Yasemin", "Müge Çiçeği", "Misk", "Odunsu Notalar", "Meşe Yosunu"],
    "stokta_mi": true
  },
  {
    "kod": "293",
    "orijinal_ad": "Victoria's Secret Sexy Little Things",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Meyveli, Tatlı",
    "notalar": ["Armut", "Liçi", "Kırmızı Elma", "Mandalina", "Gardenya", "Yasemin", "Frezya", "Manolya", "Vanilya", "Pralin", "Amber", "Misk", "Sandal Ağacı", "Benzoin"],
    "stokta_mi": true
  },
  {
    "kod": "298",
    "orijinal_ad": "Lancôme Idôle",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Şipre, Misk",
    "notalar": ["Armut", "Bergamot", "Isparta Gülü", "Yasemin Çiçeği", "Beyaz Şipre", "Beyaz Misk", "Vanilya"],
    "stokta_mi": true
  },
  {
    "kod": "299",
    "orijinal_ad": "Narciso Rodriguez Poudrée",
    "cinsiyet": "Kadın",
    "kategori": "Pudralı, Misk, Odunsu",
    "notalar": ["Şehvetli Çiçek Buketi", "Beyaz Yasemin Yaprakları", "Bulgar Gülü", "Pudramsı Misk", "Vetiver", "Sedir Ağacı"],
    "stokta_mi": true
  },
  {
    "kod": "301",
    "orijinal_ad": "YSL L'Homme",
    "cinsiyet": "Erkek",
    "kategori": "Odunsu, Baharatlı, Narenciye",
    "notalar": ["Beyaz Biber", "Limon", "Ağaç Kavunu", "Bergamot", "Meyvemsi Davana Notaları", "Likör", "Portakal Çiçeği", "Islak Otsu Notalar", "Sedir", "Aselbent", "Amber"],
    "stokta_mi": true
  },
  {
    "kod": "304",
    "orijinal_ad": "Issey Miyake L'Eau d'Issey Pour Homme",
    "cinsiyet": "Erkek",
    "kategori": "Odunsu, Akuatik, Narenciye",
    "notalar": ["Yuzu", "Limon", "Mine Çiçeği", "Mandalina", "Selvi", "Calone", "Kişniş", "Tarhun", "Adaçayı", "Mavi Lotus", "Muskat", "Müge Çiçeği", "Geranyum", "Safran", "Tarçın", "Vetiver", "Tütün"],
    "stokta_mi": true
  },
  {
    "kod": "305",
    "orijinal_ad": "Jean Paul Gaultier Scandal Pour Homme",
    "cinsiyet": "Erkek",
    "kategori": "Aromatik, Odunsu, Karamel",
    "notalar": ["Adaçayı", "Mandalina", "Karamel", "Tonka Fasulyesi", "Vetiver"],
    "stokta_mi": true
  },
  {
    "kod": "306",
    "orijinal_ad": "Jean Paul Gaultier Ultra Male",
    "cinsiyet": "Erkek",
    "kategori": "Oryantal, Fougere, Meyveli",
    "notalar": ["Armut", "Siyah Lavanta", "Nane", "Bergamot", "Kimyon", "Tarçın", "Adaçayı", "Siyah Vanilya", "Amber", "Odunsu Notalar"],
    "stokta_mi": true
  },
  {
    "kod": "308",
    "orijinal_ad": "Diesel Fuel for Life Homme",
    "cinsiyet": "Erkek",
    "kategori": "Aromatik, Odunsu, Fougère",
    "notalar": ["Anason", "Greyfurt", "Ahududu", "Lavanta", "Guaiac Ağacı", "Vetiver"],
    "stokta_mi": true
  },
  {
    "kod": "309",
    "orijinal_ad": "Viktor&Rolf Spicebomb",
    "cinsiyet": "Erkek",
    "kategori": "Oryantal, Baharatlı",
    "notalar": ["Bergamot", "Greyfurt", "Tarçın", "Pembe Biber", "Lavanta", "Elemi", "Vetiver", "Tütün", "Deri"],
    "stokta_mi": true
  },
  {
    "kod": "310",
    "orijinal_ad": "Paco Rabanne 1 Million Lucky",
    "cinsiyet": "Erkek",
    "kategori": "Odunsu, Taze, Tatlı",
    "notalar": ["Ozonik Notalar", "Erik", "Bergamot", "Greyfurt", "Portakal Çiçeği", "Bal", "Yasemin", "Kaşmir Ahşap", "Sedir", "Fındık", "Amber Ahşap", "Vetiver", "Paçuli", "Meşe Yosunu"],
    "stokta_mi": true
  },
  {
    "kod": "313",
    "orijinal_ad": "Jean Paul Gaultier Scandal",
    "cinsiyet": "Kadın",
    "kategori": "Şipre, Çiçeksi, Bal",
    "notalar": ["Mandalina", "Kan Portakalı", "Şeftali", "Portakal Çiçeği", "Yasemin", "Gardenya", "Bal", "Meyankökü", "Karamel", "Balmumu", "Paçuli"],
    "stokta_mi": true
  },
  {
    "kod": "314",
    "orijinal_ad": "Giorgio Armani My Way",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Odunsu",
    "notalar": ["Sümbülteber", "Yasemin", "Bergamot", "Portakal Çiçeği", "Vanilya", "Beyaz Misk", "Sedir Ağacı"],
    "stokta_mi": true
  },
  {
    "kod": "315",
    "orijinal_ad": "Roberto Cavalli Eau de Parfum",
    "cinsiyet": "Kadın",
    "kategori": "Amber, Çiçeksi, Baharatlı",
    "notalar": ["Pembe Biber", "Yeşil Mandalina", "Portakal Çiçeği Özü", "Mirabelle Eriği", "Kavrulmuş Tonka Tanesi", "Laos Benzoini"],
    "stokta_mi": true
  },
  {
    "kod": "316",
    "orijinal_ad": "Givenchy Very Irrésistible",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Gül, Aromatik",
    "notalar": ["Anason", "Verbena", "Gül", "Şakayık", "Vanilya", "Paçuli"],
    "stokta_mi": true
  },
  {
    "kod": "317",
    "orijinal_ad": "Hugo Boss Bottled Intense",
    "cinsiyet": "Erkek",
    "kategori": "Odunsu, Baharatlı, Meyveli",
    "notalar": ["Elma", "Portakal Çiçeği", "Tarçın", "Karanfil", "Sardunya", "Vanilya", "Sandal Ağacı", "Sedir Ağacı", "Güve Otu"],
    "stokta_mi": true
  },
  {
    "kod": "318",
    "orijinal_ad": "Givenchy L'Interdit Parfum",
    "cinsiyet": "Kadın",
    "kategori": "Beyaz Çiçek, Odunsu, Amber",
    "notalar": ["Armut", "Sümbülteber", "Yasemin", "Portakal Çiçeği", "Vetiver", "Paçuli", "Vanilya"],
    "stokta_mi": true
  },
  {
    "kod": "319",
    "orijinal_ad": "Versace Dylan Blue",
    "cinsiyet": "Erkek",
    "kategori": "Akuatik, Aromatik, Odunsu",
    "notalar": ["Kalabriyen Bergamot", "Greyfurt", "İncir Yaprağı", "Su Notaları", "Menekşe Yaprakları", "Kara Biber", "Papirus Odunu", "Ambrox", "Paçuli Özü", "Mineral Misk", "Tonka Fasulyesi", "Safran", "Projen Tütsüsü"],
    "stokta_mi": true
  },
  {
    "kod": "321",
    "orijinal_ad": "Prada Paradoxe",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Amber",
    "notalar": ["Armut", "Neroli", "Bergamot", "Yosun", "Yasemin", "Neroli Özü", "Ambrofix", "Serenolide", "Amber", "Bourbon Vanilya", "Vanilya"],
    "stokta_mi": true
  },
  {
    "kod": "323",
    "orijinal_ad": "Christian Dior Miss Dior Blooming Bouquet",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Misk",
    "notalar": ["Gül", "Şakayık", "Bergamot", "Beyaz Misk"],
    "stokta_mi": true
  },
  {
    "kod": "326",
    "orijinal_ad": "Giorgio Armani Acqua di Gio Profumo",
    "cinsiyet": "Erkek",
    "kategori": "Aromatik, Akuatik, Tütsü",
    "notalar": ["Sucul Notalar", "Bergamot", "Biberiye", "Adaçayı", "Sardunya", "Tütsü", "Paçuli"],
    "stokta_mi": true
  },
  {
    "kod": "327",
    "orijinal_ad": "Jean Paul Gaultier Le Male Elixir",
    "cinsiyet": "Erkek",
    "kategori": "Woody, Amber, Aromatik",
    "notalar": ["Tonka Fasulyesi", "Lavanta", "Benzoin"],
    "stokta_mi": true
  },
  {
    "kod": "328",
    "orijinal_ad": "YSL Myself",
    "cinsiyet": "Erkek",
    "kategori": "Odunsu, Çiçeksi",
    "notalar": ["Kalabria Bergamotu", "Tunus Portakal Çiçeği", "Endonezya Paçulisi", "Ambrofix"],
    "stokta_mi": true
  },
  {
    "kod": "329",
    "orijinal_ad": "Yves Saint Laurent Y Eau de Parfum",
    "cinsiyet": "Erkek",
    "kategori": "Aromatik, Baharatlı, Taze",
    "notalar": ["Zencefil", "Adaçayı", "Elma", "Lavanta", "Greyfurt", "Amberwood", "Tütsü"],
    "stokta_mi": true
  },
  {
    "kod": "331",
    "orijinal_ad": "Dior Sauvage Elixir",
    "cinsiyet": "Erkek",
    "kategori": "Baharatlı, Lavanta, Odunsu",
    "notalar": ["Tarçın", "Muskat", "Kakule", "Greyfurt", "Lavanta", "Meyan Kökü", "Sandal Ağacı", "Kehribar", "Paçuli", "Haiti Vetiveri"],
    "stokta_mi": true
  },
  {
    "kod": "332",
    "orijinal_ad": "Armani Stronger With You Absolutely",
    "cinsiyet": "Erkek",
    "kategori": "Amber, Baharatlı, Odunsu",
    "notalar": ["Bergamot", "Amber", "Likör", "Meyveli Notalar", "Kestane", "Sedir Ağacı"],
    "stokta_mi": true
  },
  {
    "kod": "335",
    "orijinal_ad": "Burberry Goddess",
    "cinsiyet": "Kadın",
    "kategori": "Oryantal, Vanilya",
    "notalar": ["Ahududu", "Lavanta", "Vanilya Çiçeği", "Süet", "Kakao", "Zencefil", "Vanilyalı Havyar"],
    "stokta_mi": true
  },
  {
    "kod": "336",
    "orijinal_ad": "Carolina Herrera Good Girl Blush",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Amber, Vanilya",
    "notalar": ["Bergamot", "Ylang Ylang", "Portakal Çiçeği", "Şakayık", "Gardenya", "Gül Suyu", "Tonka Fasulyesi", "Amber", "Vanilya"],
    "stokta_mi": true
  },
  {
    "kod": "338",
    "orijinal_ad": "Azzaro The Most Wanted Parfum",
    "cinsiyet": "Erkek",
    "kategori": "Baharatlı, Odunsu, Citrus",
    "notalar": ["Zencefil", "Odunsu Notalar", "Vanilya"],
    "stokta_mi": true
  },
  {
    "kod": "340",
    "orijinal_ad": "Valentino Uomo Born in Roma Intense",
    "cinsiyet": "Erkek",
    "kategori": "Amber, Vanilya, Aromatik",
    "notalar": ["Vanilya", "Vetiver", "Adaçayı", "Lavanta"],
    "stokta_mi": true
  },
  {
    "kod": "342",
    "orijinal_ad": "Jean Paul Gaultier La Belle",
    "cinsiyet": "Kadın",
    "kategori": "Oryantal, Vanilya, Meyveli",
    "notalar": ["Armut", "Bergamot", "Vanilya Orkidesi", "Tonka Fasulyesi", "Vetiver", "Amber"],
    "stokta_mi": true
  },
  {
    "kod": "343",
    "orijinal_ad": "Jean Paul Gaultier Divine",
    "cinsiyet": "Kadın",
    "kategori": "Çiçeksi, Misk, Aquatik",
    "notalar": ["Calypsone", "Kırmızı Meyveler", "Bergamot", "Zambak", "Yasemin", "Ylang-Ylang", "Beze", "Misk", "Paçuli"],
    "stokta_mi": true
  },
  {
    "kod": "345",
    "orijinal_ad": "Victoria's Secret Tease",
    "cinsiyet": "Kadın",
    "kategori": "Meyveli, Çiçeksi, Tatlı",
    "notalar": ["Armut", "Mandalina", "Liçi", "Kırmızı Elma", "Gardenya", "Bezelye", "Yasemin", "Frezya", "Manolya", "Vanilya", "Benzoin", "Misk", "Pralin", "Kehribar", "Sandal Ağacı"],
    "stokta_mi": true
  },
  {
    "kod": "346",
    "orijinal_ad": "YSL Libre Intense",
    "cinsiyet": "Kadın",
    "kategori": "Amber, Fougère, Vanilya",
    "notalar": ["Lavanta", "Vanilya", "Orkide", "Tonka Fasulyesi", "Amber", "Vetiver"],
    "stokta_mi": true
  },
  
  
  
  {
    "kod": "S-001",
    "orijinal_ad": "Dior Sauvage (EDT)",
    "cinsiyet": "Erkek",
    "kategori": "Aromatik, Füjer, Taze",
    "notalar": ["Bergamot", "Biber", "Ambroxan", "Lavanta", "Sardunya", "Paçuli", "Sedir"],
    "stokta_mi": false
  },
  {
    "kod": "S-002",
    "orijinal_ad": "Yves Saint Laurent Y (EDP)",
    "cinsiyet": "Erkek",
    "kategori": "Aromatik, Füjer, Odunsu",
    "notalar": ["Elma", "Zencefil", "Bergamot", "Adaçayı", "Sardunya", "Tonka Fasulyesi", "Amberwood", "Sedir"],
    "stokta_mi": false
  },
  {
    "kod": "S-003",
    "orijinal_ad": "Bleu de Chanel (EDP)",
    "cinsiyet": "Erkek",
    "kategori": "Odunsu, Aromatik, Narenciye",
    "notalar": ["Greyfurt", "Limon", "Nane", "Zencefil", "Muskat", "Yasemin", "Sandal Ağacı", "Tütsü", "Sedir", "Amber"],
    "stokta_mi": false
  },
  {
    "kod": "S-004",
    "orijinal_ad": "Paco Rabanne Invictus (Original)",
    "cinsiyet": "Erkek",
    "kategori": "Akuatik, Odunsu, Taze",
    "notalar": ["Deniz Notaları", "Greyfurt", "Mandalina", "Defne Yaprağı", "Yasemin", "Guaiac Ağacı", "Meşe Yosunu", "Paçuli", "Ambergris"],
    "stokta_mi": false
  },
  {
    "kod": "S-005",
    "orijinal_ad": "Carolina Herrera Good Girl",
    "cinsiyet": "Kadın",
    "kategori": "Amber, Çiçeksi, Gurme",
    "notalar": ["Badem", "Kahve", "Sümbülteber", "Yasemin", "Tonka Fasulyesi", "Kakao", "Vanilya", "Sandal Ağacı"],
    "stokta_mi": false
  }
]
"""
# --- ADIM 2: FONKSİYONLAR ve MOTOR (v4.0) ---

# *** YEREL DOSYA YOLU ***
ERKEK_YOLU = "resimler/erkek.jpg"
KADIN_YOLU = "resimler/kadin.jpg"
NICHE_YOLU = "resimler/niche.jpg" 
STOK_YOK_YOLU = "resimler/stok-yok.jpg" # Stokta olmayanlar için özel resim

# Veritabanını yükle
try:
    veritabani_json = json.loads(parfum_veritabani_json)
    # Veriyi bir DataFrame'e dönüştürmek, filtrelemeyi hızlandırır
    db_df = pd.DataFrame(veritabani_json)
except json.JSONDecodeError as e:
    st.error("JSON Veritabanı Yüklenirken Kritik Hata Oluştu! Lütfen JSON yapısını kontrol edin.")
    st.exception(e)
    st.stop()
except Exception as e:
    st.error(f"Veri işlenirken hata: {e}")
    st.stop()

# Fonksiyon: Benzerlik motorunu hazırla
@st.cache_resource
def benzerlik_motorunu_hazirla(df):
    # 'notalar' listesini ' ' (boşluk) ile birleştirilmiş string'e dönüştür
    df['notalar_str'] = df['notalar'].apply(lambda x: ' '.join(x))
    
    dokumanlar = df['notalar_str'].tolist()
    vectorizer = CountVectorizer()
    notalar_matrix = vectorizer.fit_transform(dokumanlar)
    benzerlik_skorlari = cosine_similarity(notalar_matrix)
    
    # Benzerlik skorlarını DataFrame'e dönüştürmek, aramayı kolaylaştırır
    skor_df = pd.DataFrame(benzerlik_skorlari, index=df.index, columns=df.index)
    return skor_df, df # Güncellenmiş df'i (notalar_str ile) geri döndür

# Motoru çalıştır
skor_matrisi_df, db_df = benzerlik_motorunu_hazirla(db_df)

# Fonksiyon: Ana Arama ve Öneri Fonksiyonu (v4.0)
def akilli_arama_ve_oneri(arama_terimi, df, skor_df, top_n=3):
    arama_terimi_lower = arama_terimi.lower().strip()
    
    # 1. Parfümü 'kod' veya 'orijinal_ad' ile tam/kısmi olarak bul
    # Önce kodda ara
    sonuc = df[df['kod'].str.lower() == arama_terimi_lower]
    
    # Kodda bulunamazsa, ad içinde ara
    if sonuc.empty:
        sonuc = df[df['orijinal_ad'].str.lower().str.contains(arama_terimi_lower)]
        
    # Eğer birden fazla bulursa, ilkini al
    if not sonuc.empty:
        baz_parfum_index = sonuc.index[0]
        baz_parfum = df.loc[baz_parfum_index]
        
        # 2. Skorları al
        skorlar = skor_df[baz_parfum_index]
        
        # 3. Sadece 'Stokta Olan' parfümleri filtrele
        stoktaki_parfumler_df = df[df['stokta_mi'] == True]
        
        # 4. Stoktaki parfümlerin skorlarını al
        stoktaki_skorlar = skorlar.loc[stoktaki_parfumler_df.index]
        
        # 5. En yüksek skorlu 'stoktaki' parfümleri sırala
        # (Kendisi hariç)
        if baz_parfum['stokta_mi'] == True:
            stoktaki_skorlar = stoktaki_skorlar.drop(baz_parfum_index)
            
        en_benzer_stoktaki_indexler = stoktaki_skorlar.nlargest(top_n).index
        
        oneriler = df.loc[en_benzer_stoktaki_indexler]
        
        return baz_parfum, oneriler.to_dict('records')

    # 3. Eğer isimle bulunamazsa, anahtar kelime (nota) araması yap
    arama_terimleri = arama_terimi_lower.split()
    
    # Sadece STOKTA OLANLAR içinde nota araması yap
    stoktaki_df = df[df['stokta_mi'] == True].copy()
    
    def nota_icerir(satir, terimler):
        aranacak_metin = (
            str(satir['cinsiyet']).lower() + " " +
            str(satir['kategori']).lower() + " " +
            satir['notalar_str'].
