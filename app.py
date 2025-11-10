import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os # Bu satır artık gerekmese de, bir zararı yok

# --- ADIM 1: "YAPAY ZEKA"NIN BEYNİ (v2.5 - TEK GENEL RESİM) ---
# TÜM PARFÜMLERİN "resim_yolu" ALANI SİZİN SAĞLADIĞINIZ TEK BİR URL İLE DEĞİŞTİRİLDİ.
GENEL_RESIM_URL = "https://i.imgur.com/uQy5n8t.jpg" 

parfum_veritabani_json = f"""
[
  {{
    "kod": "002",
    "orijinal_ad": "Amouage Honour Man",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Baharatlı, Odunsu, Taze",
    "notalar": ["Pembe Biber", "Sardunya", "Elemi", "Muskat", "Tütsü", "Güve Otu", "Sedir", "Misk", "Tonka Fasulyesi", "Paçuli"]
  }},
  {{
    "kod": "008",
    "orijinal_ad": "Creed Aventus",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Şipre, Meyveli, Taze",
    "notalar": ["Ananas", "Huş Ağacı", "Bergamot", "Siyah Frenk Üzümü", "Meşe Yosunu", "Misk", "Ambergris"]
  }},
  {{
    "kod": "010",
    "orijinal_ad": "Ex Nihilo Fleur Narcotique",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Meyveli",
    "notalar": ["Liçi", "Şakayık", "Şeftali", "Portakal Çiçeği", "Misk", "Yasemin"]
  }},
  {{
    "kod": "012",
    "orijinal_ad": "Frederic Malle Portrait of a Lady",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Amber, Baharatlı",
    "notalar": ["Gül", "Karanfil", "Ahududu", "Siyah Frenk Üzümü", "Tarçın", "Paçuli", "Tütsü", "Sandal Ağacı", "Misk", "Amber", "Benzoin"]
  }},
  {{
    "kod": "013",
    "orijinal_ad": "Maison Francis Kurkdjian Baccarat Rouge 540",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Odunsu, Amber",
    "notalar": ["Safran", "Yasemin", "Amberwood", "Ambergris", "Reçine", "Sedir"]
  }},
  {{
    "kod": "021",
    "orijinal_ad": "Nasomatto Black Afgano",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Tütsü, Baharatlı",
    "notalar": ["Kenevir", "Yeşil Notalar", "Reçine", "Odunsu Notalar", "Tütün", "Kahve", "Ud", "Tütsü"]
  }},
  {{
    "kod": "024",
    "orijinal_ad": "Tom Ford Black Orchid",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Çiçeksi, Gurme",
    "notalar": ["Yasemin", "Gardenya", "Ylang Ylang", "Bergamot", "Frenk Üzümü", "Yumru", "Baharat", "Meyveli Notalar", "Orkide", "Vetiver", "Sandal Ağacı", "Paçuli", "Amber", "Tütsü", "Vanilya", "Çikolata"]
  }},
  {{
    "kod": "027",
    "orijinal_ad": "Xerjoff Erba Pura",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Narenciye, Meyveli, Misk",
    "notalar": ["Sicilya Portakalı", "Calabria Bergamotu", "Sicilya Limonu", "Tropikal Meyveler", "Beyaz Misk", "Amber", "Madagaskar Vanilyası"]
  }},
  {{
    "kod": "031",
    "orijinal_ad": "Memo Paris Marfa",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Odunsu, Misk",
    "notalar": ["Sümbülteber", "Agave", "Vanilya", "Portakal Çiçeği", "Sandal Ağacı", "Beyaz Misk"]
  }},
  {{
    "kod": "040",
    "orijinal_ad": "Parfums de Marly Delina",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Meyveli, Taze",
    "notalar": ["Liçi", "Rhubarb", "Bergamot", "Muskat", "Türk Gülü", "Şakayık", "Vanilya", "Kaşmir", "Sedir", "Vetiver", "Tütsü", "Misk"]
  }},
  {{
    "kod": "041",
    "orijinal_ad": "Zadig & Voltaire This is Her",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Vanilya, Gurme",
    "notalar": ["Yasemin", "Yumuşak Vanilya", "Kestane", "Sandal Ağacı"]
  }},
  {{
    "kod": "045",
    "orijinal_ad": "Gucci Intense Oud",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Ud, Amber, Oryantal",
    "notalar": ["Armut", "Ahududu", "Safran", "Bulgar Gülü", "Portakal Çiçeği", "Doğal Ud", "Paçuli"]
  }},
  {{
    "kod": "049",
    "orijinal_ad": "Xerjoff Casamorati Lira",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Gurme, Narenciyeli",
    "notalar": ["Karamel", "Vanilya", "Kan Portakalı", "Tarçın", "Lavanta", "Meyan Kökü"]
  }},
  {{
    "kod": "052",
    "orijinal_ad": "Tom Ford Lost Cherry",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Gurme, Meyveli",
    "notalar": ["Vişne", "Acı Badem", "Likör", "Tonka Fasulyesi", "Vanilya", "Gül", "Yasemin"]
  }},
  {{
    "kod": "055",
    "orijinal_ad": "Xerjoff More Than Words",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Odunsu, Baharatlı",
    "notalar": ["Ud", "Meyvemsi Notalar", "Amber", "Güve Otu", "Olibanum"]
  }},
  {{
    "kod": "068",
    "orijinal_ad": "Tom Ford Noir Extreme",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Baharatlı, Odunsu",
    "notalar": ["Kakule", "Muskat", "Safran", "Mandalina", "Neroli", "Gül", "Yasemin", "Damla Sakızı", "Vanilya", "Amber", "Odunsu Notalar", "Sandal Ağacı"]
  }},
  {{
    "kod": "078",
    "orijinal_ad": "Maison Francis Kurkdjian Baccarat Rouge 540 Extrait",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Odunsu, Baharatlı",
    "notalar": ["Safran", "Acı Badem", "Mısır Yasemini", "Sedir", "Ambergris", "Misk"]
  }},
  {{
    "kod": "079",
    "orijinal_ad": "Orto Parisi Megamare",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Aromatik, Akuatik (Deniz), Misk",
    "notalar": ["Bergamot", "Limon", "Yosun", "Calone", "Hedione", "Ambrox", "Sedir", "Misk"]
  }},
  {{
    "kod": "080",
    "orijinal_ad": "Marc-Antoine Barrois Ganymede",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Baharatlı, Mineral",
    "notalar": ["Mineral Notalar", "Safran", "Menekşe Yaprağı", "Mandalina", "Ölümsüz Otu"]
  }},
  {{
    "kod": "085",
    "orijinal_ad": "Initio Oud for Greatness",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Baharatlı, Oryantal",
    "notalar": ["Ud", "Safran", "Muskat", "Lavanta", "Paçuli"]
  }},
  {{
    "kod": "091",
    "orijinal_ad": "Nishane Hacivat",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Şipre, Meyveli",
    "notalar": ["Ananas", "Greyfurt", "Meşe Yosunu", "Bergamot", "Odunsu Notalar", "Paçuli"]
  }},
  {{
    "kod": "092",
    "orijinal_ad": "Tom Ford Ombre Leather",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Deri, Odunsu",
    "notalar": ["Deri", "Kakule", "Yasemin", "Amber", "Paçuli", "Yosun"]
  }},
  {{
    "kod": "099",
    "orijinal_ad": "Maison Francis Kurkdjian Oud Silk Mood",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Gül, Oud, Misk",
    "notalar": ["Gül", "Papatya", "Bergamot", "Hedione", "Guaiac Ağacı", "Oud", "Papirüs"]
  }},
  {{
    "kod": "102",
    "orijinal_ad": "Richard White Chocola",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Gurme, Vanilya, Çiçeksi",
    "notalar": ["Beyaz Çikolata", "Vanilya", "Badem", "Şeftali", "Fındık", "Orkide"]
  }},
  {{
    "kod": "106",
    "orijinal_ad": "Tom Ford Electric Cherry",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Meyveli, Çiçeksi, Misk",
    "notalar": ["Kiraz", "Zencefil", "Yasemin Sambac", "Ambrette", "Pembe Biber", "Misk", "Odunsu Notalar"]
  }},
  {{
    "kod": "114",
    "orijinal_ad": "Initio Musk Therapy",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Misk, Odunsu, Çiçeksi",
    "notalar": ["Bergamot", "Greyfurt", "Sedir Ağacı", "Gül", "Paçuli", "Sandal Ağacı", "Vanilya", "Amber", "Ambergris"]
  }},
  {{
    "kod": "116",
    "orijinal_ad": "Tom Ford Vanilla Sex",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Vanilya, Gurme",
    "notalar": ["Badem", "Yasemin", "Portakal Çiçeği", "Vanilya", "Sandal Ağacı", "Amber"]
  }},
  {{
    "kod": "117",
    "orijinal_ad": "Kilian Angels' Share",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Gurme, Amber, Baharatlı",
    "notalar": ["Konyak", "Tarçın", "Tonka Fasulyesi", "Meşe", "Pralin", "Vanilya", "Sandal Ağacı"]
  }},
  {{
    "kod": "120",
    "orijinal_ad": "Marc-Antoine Barrois Tilia",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Odunsu",
    "notalar": ["Lime", "Katırtırnağı", "Yasemin", "Vetiver", "Kediotu", "Sedir Ağacı", "Ambroxan"]
  }},
  {{
    "kod": "122",
    "orijinal_ad": "Parfums de Marly Layton",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Oryantal, Çiçeksi, Baharatlı",
    "notalar": ["Elma", "Bergamot", "Lavanta", "Yasemin", "Menekşe", "Gülhatmi", "Vanilya", "Biber", "Guaiac Ağacı", "Paçuli"]
  }},
  {{
    "kod": "123",
    "orijinal_ad": "Montale Arabians Tonka",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Oryantal, Odunsu, Tonka",
    "notalar": ["Safran", "Bergamot", "Ud", "Bulgar Gülü", "Tonka Fasulyesi", "Şeker Kamışı", "Amber", "Beyaz Misk", "Meşe Yosunu"]
  }},
  {{
    "kod": "124",
    "orijinal_ad": "Louis Vuitton Imagination",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Narenciye, Amber, Çay",
    "notalar": ["Ağaç Kavunu", "Bergamot", "Portakal", "Zencefil", "Neroli", "Tarçın", "Siyah Çay", "Ambroksan", "Olibanum", "Guaiac Ağacı"]
  }},
  {{
    "kod": "125",
    "orijinal_ad": "Amouage Guidance",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Baharatlı, Gourmand",
    "notalar": ["Armut", "Fındık Sütü", "Safran", "Gül", "Yasemin", "Osmanthus", "Sandal Ağacı", "Vanilya", "Deri", "Tütsü", "Ambergris"]
  }},
  {{
    "kod": "127",
    "orijinal_ad": "Kayali Vanilla",
    "cinsiyet": "Unisex",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Vanilya",
    "notalar": ["Vanilya", "Yasemin", "Orkide", "Esmer Şeker", "Tonka Fasulyesi", "Amber", "Misk", "Paçuli"]
  }},
  {{
    "kod": "128",
    "orijinal_ad": "Parfums de Marly Althair",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Vanilya, Baharatlı, Odunsu",
    "notalar": ["Portakal Çiçeği", "Bergamot", "Tarçın", "Bourbon Vanilya", "Elemi", "Guaiac Wood", "Ambrox", "Pralin", "Misk"]
  }},
  {{
    "kod": "134",
    "orijinal_ad": "Louis Vuitton L'Immensité",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Akuatik, Aromatik, Narenciye",
    "notalar": ["Greyfurt", "Zencefil", "Bergamot", "Su Notaları", "Adaçayı", "Biberiye", "Ambroxan", "Kehribar", "Labdanum"]
  }},
  {{
    "kod": "202",
    "orijinal_ad": "Dolce & Gabbana The One EDP",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Oryantal, Baharatlı, Odunsu",
    "notalar": ["Greyfurt", "Kişniş", "Fesleğen", "Zencefil", "Kakule", "Portakal Çiçeği", "Tütün", "Amber", "Sedir Ağacı"]
  }},
  {{
    "kod": "206",
    "orijinal_ad": "Donna Karan Be Delicious Green",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Taze, Çiçeksi, Meyveli",
    "notalar": ["Elma", "Salatalık", "Greyfurt", "Manolya", "Gül", "Sandal Ağacı", "Beyaz Amber"]
  }},
  {{
    "kod": "207",
    "orijinal_ad": "Giorgio Armani Acqua di Gio",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Aromatik, Akuatik (Deniz), Taze",
    "notalar": ["Deniz Notaları", "Limon", "Bergamot", "Mandalina", "Yasemin", "Beyaz Misk", "Sedir"]
  }},
  {{
    "kod": "208",
    "orijinal_ad": "Giorgio Armani Code Profumo",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Baharatlı, Odunsu",
    "notalar": ["Tonka Fasulyesi", "Kakule", "Odunsu Notalar"]
  }},
  {{
    "kod": "209",
    "orijinal_ad": "Giorgio Armani Si Parfum",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Şipre, Meyveli, Vanilya",
    "notalar": ["Siyah Frenk Üzümü", "Gül", "Vanilya", "Paçuli", "Frezya", "Mandalina"]
  }},
  {{
    "kod": "210",
    "orijinal_ad": "Giorgio Armani Si Intense",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Çiçeksi, Vanilya",
    "notalar": ["Siyah Frenk Üzümü", "Gül", "Davana", "Vanilya", "Siyah Çay", "Paçuli"]
  }},
  {{
    "kod": "211",
    "orijinal_ad": "Giorgio Armani Code for Women",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Oryantal",
    "notalar": ["Zambak", "Yasemin", "Taze Zencefil", "Portakal Çiçeği", "Vanilya", "Sandal Ağacı"]
  }},
  {{
    "kod": "215",
    "orijinal_ad": "Gucci by Flora",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Meyveli, Tatlı",
    "notalar": ["Gardenya", "Armut Çiçeği", "Esmer Şeker", "Kırmızı Meyveler", "Paçuli", "Yasemin"]
  }},
  {{
    "kod": "217",
    "orijinal_ad": "Guerlain Robe Noir",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Meyveli, Tatlı",
    "notalar": ["Vişne", "Gül", "Badem", "Siyah Frenk Üzümü", "Misk", "Paçuli"]
  }},
  {{
    "kod": "218",
    "orijinal_ad": "Hermes Terre de Hermes",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Baharatlı, Narenciye",
    "notalar": ["Portakal", "Greyfurt", "Vetiver", "Biber", "Sedir", "Paçuli"]
  }},
  {{
    "kod": "222",
    "orijinal_ad": "Lacoste L.12.12 Blanc - White",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Aromatik, Taze",
    "notalar": ["Greyfurt", "Kakule", "Sümbülteber", "Ylang-Ylang", "Süet", "Vetiver"]
  }},
  {{
    "kod": "224",
    "orijinal_ad": "Lacoste Pour Femme",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Odunsu, Pudralı",
    "notalar": ["Frezya", "Karabiber", "Yasemin", "Süet", "Sedir Ağacı", "Heliotrop"]
  }},
  {{
    "kod": "225",
    "orijinal_ad": "Lancome Tresor La Nuit",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Gurme, Vanilya",
    "notalar": ["Pralin", "Karamel", "Vanilya", "Orkide", "Gül", "Liçi", "Paçuli", "Kahve"]
  }},
  {{
    "kod": "226",
    "orijinal_ad": "Lancome La Vie Est Belle",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Gurme, Tatlı",
    "notalar": ["İris", "Pralin", "Vanilya", "Paçuli", "Portakal Çiçeği", "Siyah Frenk Üzümü"]
  }},
  {{
    "kod": "229",
    "orijinal_ad": "Moschino Love Love",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Odunsu, Narenciye",
    "notalar": ["Greyfurt", "Portakal", "Limon", "Şeker Kamışı", "Misk", "Sedir", "Kırmızı Frenk Üzümü"]
  }},
  {{
    "kod": "231",
    "orijinal_ad": "Paco Rabanne Invictus",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Akuatik (Deniz), Odunsu, Taze",
    "notalar": ["Deniz Notaları", "Greyfurt", "Defne Yaprağı", "Ambergris", "Guaiac Ağacı", "Meşe Yosunu"]
  }},
  {{
    "kod": "233",
    "orijinal_ad": "Paco Rabanne Olympea",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Çiçeksi, Gurme",
    "notalar": ["Tuzlu Vanilya", "Su Yasemini", "Mandalina", "Zambak", "Kaşmir Ağacı", "Ambergris"]
  }},
  {{
    "kod": "234",
    "orijinal_ad": "Paco Rabanne Lady Million",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Meyveli, Tatlı",
    "notalar": ["Bal", "Paçuli", "Portakal Çiçeği", "Ahududu", "Yasemin", "Amber"]
  }},
  {{
    "kod": "235",
    "orijinal_ad": "Thierry Mugler Alien",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Beyaz Çiçek, Amber",
    "notalar": ["Yasemin", "Kaşmir", "Beyaz Amber", "Odunsu Notalar"]
  }},
  {{
    "kod": "238",
    "orijinal_ad": "Versace Eros",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Aromatik, Fougère, Taze",
    "notalar": ["Nane", "Yeşil Elma", "Limon", "Tonka Fasulyesi", "Vanilya", "Amber", "Sedir"]
  }},
  {{
    "kod": "241",
    "orijinal_ad": "Versace Crystal Noir",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Baharatlı, Çiçeksi, Amber",
    "notalar": ["Kakule", "Karabiber", "Zencefil", "Gardenya", "Hindistan Cevizi", "Amber", "Misk"]
  }},
  {{
    "kod": "242",
    "orijinal_ad": "Yves Saint Laurent Black Opium",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Gurme, Vanilya",
    "notalar": ["Kahve", "Vanilya", "Portakal Çiçeği", "Armut", "Yasemin", "Misk", "Sedir"]
  }},
  {{
    "kod": "243",
    "orijinal_ad": "Carolina Herrera 212 VIP",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Vanilya, Rom, Gurme",
    "notalar": ["Rom", "Vanilya", "Çarkıfelek", "Tonka Fasulyesi", "Gardenya", "Misk"]
  }},
  {{
    "kod": "246",
    "orijinal_ad": "Bvlgari Aqva Pour Homme",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Akuatik (Deniz), Aromatik, Taze",
    "notalar": ["Deniz Yosunu", "Mandalina", "Pamuk Çiçeği", "Sedir", "Amber"]
  }},
  {{
    "kod": "248",
    "orijinal_ad": "Calvin Klein Euphoria",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Çiçeksi, Meyveli",
    "notalar": ["Nar", "Siyah Orkide", "Lotus Çiçeği", "Amber", "Misk", "Paçuli", "Maun"]
  }},
  {{
    "kod": "249",
    "orijinal_ad": "Carrolina Herrera 212 Sexy Magnetik",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Çiçeksi, Tatlı",
    "notalar": ["Pamuk Şekeri", "Pembe Biber", "Vanilya", "Misk", "Gardenya", "Sandal Ağacı", "Mandalina"]
  }},
  {{
    "kod": "251",
    "orijinal_ad": "Carolina Herrera 212 Sexy",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Oryantal, Çiçeksi, Tatlı",
    "notalar": ["Gül", "Biber", "Bergamot", "Gardenya", "Sardunya", "Pamuk Şekeri", "Vanilya", "Baharat"]
  }},
  {{
    "kod": "253",
    "orijinal_ad": "Chanel Bleu de Chanel",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Aromatik, Odunsu, Amber",
    "notalar": ["Limon", "Bergamot", "Nane", "Pelin Otu", "Lavanta", "Sardunya", "Ananas", "Sandal Ağacı", "Sedir", "Amberwood", "Tonka Fasulyesi"]
  }},
  {{
    "kod": "255",
    "orijinal_ad": "Christian Dior J'adore",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Meyveli",
    "notalar": ["Ylang-Ylang", "Yasemin", "Gül", "Şeftali", "Armut", "Misk", "Sedir"]
  }},
  {{
    "kod": "256",
    "orijinal_ad": "Christian Dior Addict",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Çiçeksi, Vanilya",
    "notalar": ["Vanilya", "Tonka Fasulyesi", "Yasemin", "Portakal Çiçeği", "Sandal Ağacı", "Bourbon Vanilyası"]
  }},
  {{
    "kod": "260",
    "orijinal_ad": "Christian Dior Homme Intense",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Çiçeksi, Misk",
    "notalar": ["İris", "Lavanta", "Sedir", "Vetiver", "Kakao", "Amber"]
  }},
  {{
    "kod": "261",
    "orijinal_ad": "Christian Dior Fahrenheit",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Deri, Aromatik, Odunsu",
    "notalar": ["Menekşe Yaprağı", "Deri", "Muskat", "Sedir", "Vetiver", "Lavanta"]
  }},
  {{
    "kod": "262",
    "orijinal_ad": "Chanel Coco Mademoiselle",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Şipre, Çiçeksi, Narenciye",
    "notalar": ["Narenciye", "Portakal", "Bergamot", "Yasemin", "Gül", "Liçi", "Amber", "Beyaz Misk", "Vetiver", "Paçuli"]
  }},
  {{
    "kod": "263",
    "orijinal_ad": "Chanel Chance Eau Tendre",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Meyveli",
    "notalar": ["Greyfurt", "Ayva", "Yasemin", "Gül", "Beyaz Misk", "Hafif Odunsu Notalar", "Amber"]
  }},
  {{
    "kod": "264",
    "orijinal_ad": "Chanel Chance Eau de Parfum",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Baharatlı, Amber",
    "notalar": ["Pembe Biber", "Yasemin", "Ambersi Paçuli", "Beyaz Misk", "Vanilya"]
  }},
  {{
    "kod": "265",
    "orijinal_ad": "Chanel No. 5",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Aldehit, Sabunsu",
    "notalar": ["Aldehitler", "Ylang-Ylang", "Neroli", "Gül", "Yasemin", "Sandal Ağacı", "Vanilya", "Amber"]
  }},
  {{
    "kod": "267",
    "orijinal_ad": "Chloé Eau de Parfum",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Gül, Pudralı",
    "notalar": ["Şakayık", "Liçi", "Gül", "Manolya", "Sedir", "Amber"]
  }},
  {{
    "kod": "268",
    "orijinal_ad": "Chanel Egoiste",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Baharatlı, Sandal Ağacı",
    "notalar": ["Sandal Ağacı", "Gül", "Tarçın", "Vanilya", "Tütün", "Limon"]
  }},
  {{
    "kod": "270",
    "orijinal_ad": "Emporio Armani Stronger With You",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Aromatik, Baharatlı",
    "notalar": ["Nane", "Menekşe Yaprağı", "Pembe Biber", "Kakule", "Tarçın", "Lavanta", "Ananas", "Kavun", "Adaçayı", "Amber", "Sedir", "Kestane", "Vanilya"]
  }},
  {{
    "kod": "271",
    "orijinal_ad": "YSL Libre",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Odunsu, Misk",
    "notalar": ["Mandalina Yağı", "Tahıl Yağı", "Fransız Lavanta Yağı", "Kuşüzümü", "Lavanta Yağı", "Zambak", "Yasemin", "Portakal Çiçeği", "Vanilya Özü", "Sedir Ağacı Yağı", "Amber", "Misk"]
  }},
  {{
    "kod": "274",
    "orijinal_ad": "Burberry Classic",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Meyveli, Çiçeksi, Odunsu",
    "notalar": ["Yeşil Elma", "Bergamot", "Şeftali", "Kayısı", "Erik", "Yasemin", "Sandal Ağacı", "Sedir", "Misk", "Vanilya"]
  }},
  {{
    "kod": "275",
    "orijinal_ad": "Burberry Classic Men",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Aromatik",
    "notalar": ["Bergamot", "Taze Nane", "Lavanta", "Dağ Kekiği", "Itır Çiçeği", "Sandal Ağacı", "Amber", "Sedir"]
  }},
  {{
    "kod": "276",
    "orijinal_ad": "Chloé Love",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Baharatlı",
    "notalar": ["Mor Salkımlı Sümbüller", "Leylaklar", "Portakal Çiçeği", "Sıcak Baharatlar"]
  }},
  {{
    "kod": "278",
    "orijinal_ad": "Paco Rabanne Black XS for Him",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Oryantal, Odunsu, Tatlı",
    "notalar": ["Turunçgiller", "Limon", "Adaçayı", "Kadife Çiçeği", "Pralin", "Tarçın", "Tolu Balsamı", "Siyah Kakule", "Paçuli", "Siyah Kehribar", "Abanoz Ağacı", "Palisander Gül Ağacı"]
  }},
  {{
    "kod": "281",
    "orijinal_ad": "Giorgio Armani Sì Passione",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Meyveli, Tatlı",
    "notalar": ["Ananas", "Gül", "Armut", "Vanilya", "Sedir", "Amberwood"]
  }},
  {{
    "kod": "282",
    "orijinal_ad": "Gucci Guilty Pour Homme",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Aromatik, Taze",
    "notalar": ["Limon", "Lavanta", "Neroli", "Sedir", "Paçuli", "Amber"]
  }},
  {{
    "kod": "284",
    "orijinal_ad": "Givenchy Insensé Ultramarine",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Akuatik (Deniz), Taze, Meyveli",
    "notalar": ["Kırmızı Meyveler", "Deniz Notaları", "Nane", "Manolya", "Vetiver", "Tütün"]
  }},
  {{
    "kod": "285",
    "orijinal_ad": "Bvlgari Man in Black",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Baharatlı, Deri",
    "notalar": ["Baharatlar", "Rom", "Tütün", "Deri", "İris", "Sümbülteber", "Tonka Fasulyesi", "Guaiac Ağacı", "Benzoin"]
  }},
  {{
    "kod": "286",
    "orijinal_ad": "Narciso Rodriguez For Her",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Misk, Çiçeksi, Odunsu",
    "notalar": ["Vişne", "Erik", "Frezya", "Orkide", "İris", "Vanilya", "Misk", "Amber"]
  }},
  {{
    "kod": "288",
    "orijinal_ad": "Jean Paul Gaultier Le Male",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Aromatik, Odunsu, Tatlı",
    "notalar": ["Kakule", "Lavanta", "İris", "Vanilya", "Doğu Notaları", "Odunsu Notalar"]
  }},
  {{
    "kod": "289",
    "orijinal_ad": "Carolina Herrera 212 Men",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Aromatik, Baharatlı",
    "notalar": ["Narenciye Yaprakları", "Kesik Çim", "Baharat Yaprakları", "Taze Biber", "Zencefil", "Gardenya", "Sandal Ağacı", "Gayak Ağacı", "Tütsülenmiş Beyaz Misk"]
  }},
  {{
    "kod": "291",
    "orijinal_ad": "Rochas Femme",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Şipre, Meyveli, Baharatlı",
    "notalar": ["Erik", "Şeftali", "Tarçın", "Karanfil", "Gül", "Meşe Yosunu", "Amber", "Misk"]
  }},
  {{
    "kod": "292",
    "orijinal_ad": "Victoria's Secret Bombshell",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Meyveli, Çiçeksi",
    "notalar": ["Çarkıfelek Meyvesi", "Greyfurt", "Ananas", "Mandalina", "Çilek", "Şakayık", "Vanilya Orkidesi", "Kırmızı Meyveler", "Yasemin", "Müge Çiçeği", "Misk", "Odunsu Notalar", "Meşe Yosunu"]
  }},
  {{
    "kod": "293",
    "orijinal_ad": "Victoria's Secret Sexy Little Things",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Meyveli, Tatlı",
    "notalar": ["Armut", "Liçi", "Kırmızı Elma", "Mandalina", "Gardenya", "Yasemin", "Frezya", "Manolya", "Vanilya", "Pralin", "Amber", "Misk", "Sandal Ağacı", "Benzoin"]
  }},
  {{
    "kod": "298",
    "orijinal_ad": "Lancôme Idôle",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Şipre, Misk",
    "notalar": ["Armut", "Bergamot", "Isparta Gülü", "Yasemin Çiçeği", "Beyaz Şipre", "Beyaz Misk", "Vanilya"]
  }},
  {{
    "kod": "299",
    "orijinal_ad": "Narciso Rodriguez Poudrée",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Pudralı, Misk, Odunsu",
    "notalar": ["Şehvetli Çiçek Buketi", "Beyaz Yasemin Yaprakları", "Bulgar Gülü", "Pudramsı Misk", "Vetiver", "Sedir Ağacı"]
  }},
  {{
    "kod": "301",
    "orijinal_ad": "YSL L'Homme",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Baharatlı, Narenciye",
    "notalar": ["Beyaz Biber", "Limon", "Ağaç Kavunu", "Bergamot", "Meyvemsi Davana Notaları", "Likör", "Portakal Çiçeği", "Islak Otsu Notalar", "Sedir", "Aselbent", "Amber"]
  }},
  {{
    "kod": "304",
    "orijinal_ad": "Issey Miyake L'Eau d'Issey Pour Homme",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Akuatik, Narenciye",
    "notalar": ["Yuzu", "Limon", "Mine Çiçeği", "Mandalina", "Selvi", "Calone", "Kişniş", "Tarhun", "Adaçayı", "Mavi Lotus", "Muskat", "Müge Çiçeği", "Geranyum", "Safran", "Tarçın", "Vetiver", "Tütün"]
  }},
  {{
    "kod": "305",
    "orijinal_ad": "Jean Paul Gaultier Scandal Pour Homme",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Aromatik, Odunsu, Karamel",
    "notalar": ["Adaçayı", "Mandalina", "Karamel", "Tonka Fasulyesi", "Vetiver"]
  }},
  {{
    "kod": "306",
    "orijinal_ad": "Jean Paul Gaultier Ultra Male",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Oryantal, Fougere, Meyveli",
    "notalar": ["Armut", "Siyah Lavanta", "Nane", "Bergamot", "Kimyon", "Tarçın", "Adaçayı", "Siyah Vanilya", "Amber", "Odunsu Notalar"]
  }},
  {{
    "kod": "308",
    "orijinal_ad": "Diesel Fuel for Life Homme",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Aromatik, Odunsu, Fougère",
    "notalar": ["Anason", "Greyfurt", "Ahududu", "Lavanta", "Guaiac Ağacı", "Vetiver"]
  }},
  {{
    "kod": "309",
    "orijinal_ad": "Viktor&Rolf Spicebomb",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Oryantal, Baharatlı",
    "notalar": ["Bergamot", "Greyfurt", "Tarçın", "Pembe Biber", "Lavanta", "Elemi", "Vetiver", "Tütün", "Deri"]
  }},
  {{
    "kod": "310",
    "orijinal_ad": "Paco Rabanne 1 Million Lucky",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Taze, Tatlı",
    "notalar": ["Ozonik Notalar", "Erik", "Bergamot", "Greyfurt", "Portakal Çiçeği", "Bal", "Yasemin", "Kaşmir Ahşap", "Sedir", "Fındık", "Amber Ahşap", "Vetiver", "Paçuli", "Meşe Yosunu"]
  }},
  {{
    "kod": "313",
    "orijinal_ad": "Jean Paul Gaultier Scandal",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Şipre, Çiçeksi, Bal",
    "notalar": ["Mandalina", "Kan Portakalı", "Şeftali", "Portakal Çiçeği", "Yasemin", "Gardenya", "Bal", "Meyankökü", "Karamel", "Balmumu", "Paçuli"]
  }},
  {{
    "kod": "314",
    "orijinal_ad": "Giorgio Armani My Way",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Odunsu",
    "notalar": ["Sümbülteber", "Yasemin", "Bergamot", "Portakal Çiçeği", "Vanilya", "Beyaz Misk", "Sedir Ağacı"]
  }},
  {{
    "kod": "315",
    "orijinal_ad": "Roberto Cavalli Eau de Parfum",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Çiçeksi, Baharatlı",
    "notalar": ["Pembe Biber", "Yeşil Mandalina", "Portakal Çiçeği Özü", "Mirabelle Eriği", "Kavrulmuş Tonka Tanesi", "Laos Benzoini"]
  }},
  {{
    "kod": "316",
    "orijinal_ad": "Givenchy Very Irrésistible",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Gül, Aromatik",
    "notalar": ["Anason", "Verbena", "Gül", "Şakayık", "Vanilya", "Paçuli"]
  }},
  {{
    "kod": "317",
    "orijinal_ad": "Hugo Boss Bottled Intense",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Baharatlı, Meyveli",
    "notalar": ["Elma", "Portakal Çiçeği", "Tarçın", "Karanfil", "Sardunya", "Vanilya", "Sandal Ağacı", "Sedir Ağacı", "Güve Otu"]
  }},
  {{
    "kod": "318",
    "orijinal_ad": "Givenchy L'Interdit Parfum",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Beyaz Çiçek, Odunsu, Amber",
    "notalar": ["Armut", "Sümbülteber", "Yasemin", "Portakal Çiçeği", "Vetiver", "Paçuli", "Vanilya"]
  }},
  {{
    "kod": "319",
    "orijinal_ad": "Versace Dylan Blue",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Akuatik, Aromatik, Odunsu",
    "notalar": ["Kalabriyen Bergamot", "Greyfurt", "İncir Yaprağı", "Su Notaları", "Menekşe Yaprakları", "Kara Biber", "Papirus Odunu", "Ambrox", "Paçuli Özü", "Mineral Misk", "Tonka Fasulyesi", "Safran", "Projen Tütsüsü"]
  }},
  {{
    "kod": "321",
    "orijinal_ad": "Prada Paradoxe",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Amber",
    "notalar": ["Armut", "Neroli", "Bergamot", "Yosun", "Yasemin", "Neroli Özü", "Ambrofix", "Serenolide", "Amber", "Bourbon Vanilya", "Vanilya"]
  }},
  {{
    "kod": "323",
    "orijinal_ad": "Christian Dior Miss Dior Blooming Bouquet",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Misk",
    "notalar": ["Gül", "Şakayık", "Bergamot", "Beyaz Misk"]
  }},
  {{
    "kod": "326",
    "orijinal_ad": "Giorgio Armani Acqua di Gio Profumo",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Aromatik, Akuatik, Tütsü",
    "notalar": ["Sucul Notalar", "Bergamot", "Biberiye", "Adaçayı", "Sardunya", "Tütsü", "Paçuli"]
  }},
  {{
    "kod": "327",
    "orijinal_ad": "Jean Paul Gaultier Le Male Elixir",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Woody, Amber, Aromatik",
    "notalar": ["Tonka Fasulyesi", "Lavanta", "Benzoin"]
  }},
  {{
    "kod": "328",
    "orijinal_ad": "YSL Myself",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Odunsu, Çiçeksi",
    "notalar": ["Kalabria Bergamotu", "Tunus Portakal Çiçeği", "Endonezya Paçulisi", "Ambrofix"]
  }},
  {{
    "kod": "329",
    "orijinal_ad": "Yves Saint Laurent Y Eau de Parfum",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Aromatik, Baharatlı, Taze",
    "notalar": ["Zencefil", "Adaçayı", "Elma", "Lavanta", "Greyfurt", "Amberwood", "Tütsü"]
  }},
  {{
    "kod": "331",
    "orijinal_ad": "Dior Sauvage Elixir",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Baharatlı, Lavanta, Odunsu",
    "notalar": ["Tarçın", "Muskat", "Kakule", "Greyfurt", "Lavanta", "Meyan Kökü", "Sandal Ağacı", "Kehribar", "Paçuli", "Haiti Vetiveri"]
  }},
  {{
    "kod": "332",
    "orijinal_ad": "Armani Stronger With You Absolutely",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Baharatlı, Odunsu",
    "notalar": ["Bergamot", "Amber", "Likör", "Meyveli Notalar", "Kestane", "Sedir Ağacı"]
  }},
  {{
    "kod": "335",
    "orijinal_ad": "Burberry Goddess",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Oryantal, Vanilya",
    "notalar": ["Ahududu", "Lavanta", "Vanilya Çiçeği", "Süet", "Kakao", "Zencefil", "Vanilyalı Havyar"]
  }},
  {{
    "kod": "336",
    "orijinal_ad": "Carolina Herrera Good Girl Blush",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Amber, Vanilya",
    "notalar": ["Bergamot", "Ylang Ylang", "Portakal Çiçeği", "Şakayık", "Gardenya", "Gül Suyu", "Tonka Fasulyesi", "Amber", "Vanilya"]
  }},
  {{
    "kod": "338",
    "orijinal_ad": "Azzaro The Most Wanted Parfum",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Baharatlı, Odunsu, Citrus",
    "notalar": ["Zencefil", "Odunsu Notalar", "Vanilya"]
  }},
  {{
    "kod": "340",
    "orijinal_ad": "Valentino Uomo Born in Roma Intense",
    "cinsiyet": "Erkek",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Vanilya, Aromatik",
    "notalar": ["Vanilya", "Vetiver", "Adaçayı", "Lavanta"]
  }},
  {{
    "kod": "342",
    "orijinal_ad": "Jean Paul Gaultier La Belle",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Oryantal, Vanilya, Meyveli",
    "notalar": ["Armut", "Bergamot", "Vanilya Orkidesi", "Tonka Fasulyesi", "Vetiver", "Amber"]
  }},
  {{
    "kod": "343",
    "orijinal_ad": "Jean Paul Gaultier Divine",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Çiçeksi, Misk, Aquatik",
    "notalar": ["Calypsone", "Kırmızı Meyveler", "Bergamot", "Zambak", "Yasemin", "Ylang-Ylang", "Beze", "Misk", "Paçuli"]
  }},
  {{
    "kod": "345",
    "orijinal_ad": "Victoria's Secret Tease",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Meyveli, Çiçeksi, Tatlı",
    "notalar": ["Armut", "Mandalina", "Liçi", "Kırmızı Elma", "Gardenya", "Bezelye", "Yasemin", "Frezya", "Manolya", "Vanilya", "Benzoin", "Misk", "Pralin", "Kehribar", "Sandal Ağacı"]
  }},
  {{
    "kod": "346",
    "orijinal_ad": "YSL Libre Intense",
    "cinsiyet": "Kadın",
    "resim_yolu": "{GENEL_RESIM_URL}",
    "kategori": "Amber, Fougère, Vanilya",
    "notalar": ["Lavanta", "Vanilya", "Orkide", "Tonka Fasulyesi", "Amber", "Vetiver"]
  }}
]
"""
# --- ADIM 2: VERİTABANINI VE MOTORU YÜKLEME ---

# Veritabanını yükle
try:
    veritabani = json.loads(parfum_veritabani_json)
except json.JSONDecodeError as e:
    st.error("JSON Veritabanı Yüklenirken Kritik Hata Oluştu!")
    st.exception(e)
    st.stop()

# Fonksiyon: Nota ile arama
def nota_ile_parfum_bul(arama_terimi, db):
    sonuclar = []
    arama_terimleri = arama_terimi.lower().split()
    
    for parfum in db:
        aranacak_metin = (
            parfum['cinsiyet'].lower() + " " +
            parfum['kategori'].lower() + " " +
            " ".join(parfum['notalar']).lower()
        )
        if all(terim in aranacak_metin for terim in arama_terimleri):
            sonuclar.append(parfum)
            
    return sonuclar

# Fonksiyon: Benzerlik motorunu hazırla ve çalıştır
@st.cache_resource
def benzerlik_motorunu_hazirla(db):
    dokumanlar = [" ".join(p['notalar']) for p in db]
    vectorizer = CountVectorizer()
    notalar_matrix = vectorizer.fit_transform(dokumanlar)
    benzerlik_skorlari = cosine_similarity(notalar_matrix)
    return benzerlik_skorlari

# Motoru çalıştır
benzerlik_skor_matrisi = benzerlik_motorunu_hazirla(veritabani)

# Fonksiyon: Benzerlik önermesi
def benzer_parfumleri_getir(kod_veya_ad, db, skor_matrisi, top_n=3):
    kod_veya_ad_lower = kod_veya_ad.lower().strip()
    bulunan_index = -1
    bulunan_parfum = None

    for i, parfum in enumerate(db):
        if parfum['kod'].lower() == kod_veya_ad_lower:
            bulunan_index = i
            bulunan_parfum = parfum
            break
    
    if bulunan_index == -1:
        for i, parfum in enumerate(db):
            if kod_veya_ad_lower in parfum['orijinal_ad'].lower():
                bulunan_index = i
                bulunan_parfum = parfum
                break
                
    if bulunan_index == -1:
        return None, []

    skorlar = list(enumerate(skor_matrisi[bulunan_index]))
    skorlar = sorted(skorlar, key=lambda x: x[1], reverse=True)
    en_benzer_indexler = [i[0] for i in skorlar[1:top_n+1]]
    benzer_parfumler = [db[i] for i in en_benzer_indexler]
    return bulunan_parfum, benzer_parfumler

# --- ADIM 3: ARAYÜZÜ (WEB SİTESİ) OLUŞTURMA (v2.5 - TEK URL) ---

st.set_page_config(page_title="Lorinna Koku Rehberi", layout="wide", page_icon="✨")
st.title("✨ Lorinna Koku Rehberi (v2.5)")
st.write(f"Stoktaki **{len(veritabani)}** adet parfüm arasından mükemmel kokuyu bulun.")
st.markdown("---")

col1, col2 = st.columns(2)

# --- SÜTUN 1: NOTA, KATEGORİ VEYA CİNSİYETE GÖRE ARAMA ---
with col1:
    st.header("1. Nota veya Gruba Göre Bul")
    st.write("Arama kutusuna birden fazla kelime yazabilirsiniz:")
    st.caption("Örnek aramalar: 'odunsu erkek', 'unisex vanilya', 'çiçeksi kadın', 'ananas'")
    
    nota_terimi = st.text_input("Aranacak Nota veya Grup:", key="nota_arama")
    
    if st.button("Parfümleri Bul", key="nota_buton"):
        if nota_terimi:
            sonuclar = nota_ile_parfum_bul(nota_terimi, veritabani)
            if not sonuclar:
                st.warning(f"'{nota_terimi}' içeren parfüm bulunamadı.")
            else:
                st.success(f"'{nota_terimi}' içeren {len(sonuclar)} adet parfüm bulundu:")
                
                for p in sonuclar:
                    col_img, col_txt = st.columns([1, 4]) 
                    with col_img:
                        # DÜZELTME: Artık os.path.exists kontrolü yok, doğrudan URL'yi göster
                        st.image(p['resim_yolu'], width=100)
                    with col_txt:
                        st.markdown(f"**{p['kod']} - {p['orijinal_ad']} ({p['cinsiyet']})**")
                        st.caption(f"Kategori: *{p['kategori']}*")
                    st.divider() 
        else:
            st.error("Lütfen aranacak bir terim girin.")

# --- SÜTUN 2: BENZER KOKU ÖNERİSİ ---
with col2:
    st.header("2. Benzer Koku Öner")
    st.write("Parfümün kodunu veya adının bir kısmını yazın (Örn: 'aventus', 'baccarat', '008')")
    
    isim_terimi = st.text_input("Beğenilen Parfümün Kodu veya Adı:", key="isim_arama")
    
    if st.button("Benzer Öneriler Getir", key="isim_buton"):
        if isim_terimi:
            baz_parfum, benzer_oneriler = benzer_parfumleri_getir(isim_terimi, veritabani, benzerlik_skor_matrisi, top_n=3)
            
            if baz_parfum:
                st.success(f"Baz Alınan Parfüm:")
                col_img_baz, col_txt_baz = st.columns([1, 4])
                with col_img_baz:
                    # DÜZELTME: Artık os.path.exists kontrolü yok, doğrudan URL'yi göster
                    st.image(baz_parfum['resim_yolu'], width=100)
                with col_txt_baz:
                    st.markdown(f"**{baz_parfum['kod']} - {baz_parfum['orijinal_ad']} ({baz_parfum['cinsiyet']})**")
                
                st.markdown("---")
                st.write(f"Bu parfüme en çok benzeyen ilk 3 öneri:")
                
                for p in benzer_oneriler:
                    col_img, col_txt = st.columns([1, 4])
                    with col_img:
                        # DÜZELTME: Artık os.path.exists kontrolü yok, doğrudan URL'yi göster
                        st.image(p['resim_yolu'], width=100)
                    with col_txt:
                        st.markdown(f"**{p['kod']} - {p['orijinal_ad']} ({p['cinsiyet']})**")
                        st.caption(f"Öne çıkan ortak notalar: {', '.join(p['notalar'][:4])}...")
                    st.divider()
            else:
                st.warning(f"'{isim_terimi}' kodlu veya isimli parfüm bulunamadı.")
        else:
            st.error("Lütfen aranacak bir parfüm girin.")

# --- KODUN SONU ---
