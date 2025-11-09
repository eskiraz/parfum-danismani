import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# --- ADIM 1: "YAPAY ZEKA"NIN BEYNİ (76 PARFÜMLÜK TAM VERİTABANI) ---
# Tüm 76 parfüm buraya gömüldü
parfum_veritabani_json = """
[
  {
    "kod": "LRN.09.008",
    "orijinal_ad": "Creed Aventus",
    "kategori": "Şipre, Meyveli, Taze",
    "notalar": ["Ananas", "Huş Ağacı", "Bergamot", "Siyah Frenk Üzümü", "Meşe Yosunu", "Misk", "Ambergris"]
  },
  {
    "kod": "LRN.09.010",
    "orijinal_ad": "Ex Nihilo Fleur Narcotique",
    "kategori": "Çiçeksi, Meyveli",
    "notalar": ["Liçi", "Şakayık", "Şeftali", "Portakal Çi çi", "Misk", "Yasemin"]
  },
  {
    "kod": "LRN.09.031",
    "orijinal_ad": "Memo Paris Marfa",
    "kategori": "Çiçeksi, Odunsu, Misk",
    "notalar": ["Sümbülteber", "Agave", "Vanilya", "Portakal Çiçeği", "Sandal Ağacı", "Beyaz Misk"]
  },
  {
    "kod": "LRN.09.049",
    "orijinal_ad": "Xerjoff Casamorati Lira",
    "kategori": "Amber, Gurme, Narenciyeli",
    "notalar": ["Karamel", "Vanilya", "Kan Portakalı", "Tarçın", "Lavanta", "Meyan Kökü"]
  },
  {
    "kod": "LRN.09.052",
    "orijinal_ad": "Tom Ford Lost Cherry",
    "kategori": "Amber, Gurme, Meyveli",
    "notalar": ["Vişne", "Acı Badem", "Likör", "Tonka Fasulyesi", "Vanilya", "Gül", "Yasemin"]
  },
  {
    "kod": "LRN.09.055",
    "orijinal_ad": "Xerjoff More Than Words",
    "kategori": "Amber, Odunsu, Baharatlı",
    "notalar": ["Ud", "Meyvemsi Notalar", "Amber", "Güve Otu", "Olibanum"]
  },
  {
    "kod": "LRN.09.078",
    "orijinal_ad": "Maison Francis Kurkdjian Baccarat Rouge 540 Extrait",
    "kategori": "Amber, Odunsu, Baharatlı",
    "notalar": ["Safran", "Acı Badem", "Mısır Yasemini", "Sedir", "Ambergris", "Misk"]
  },
  {
    "kod": "LRN.09.080",
    "orijinal_ad": "Marc-Antoine Barrois Ganymede",
    "kategori": "Odunsu, Baharatlı, Mineral",
    "notalar": ["Mineral Notalar", "Safran", "Menekşe Yaprağı", "Mandalina", "Ölümsüz Otu"]
  },
  {
    "kod": "LRN.09.085",
    "orijinal_ad": "Initio Oud for Greatness",
    "kategori": "Odunsu, Baharatlı, Oryantal",
    "notalar": ["Ud", "Safran", "Muskat", "Lavanta", "Paçuli"]
  },
  {
    "kod": "LRN.09.091",
    "orijinal_ad": "Nishane Hacivat",
    "kategori": "Şipre, Meyveli",
    "notalar": ["Ananas", "Greyfurt", "Meşe Yosunu", "Bergamot", "Odunsu Notalar", "Paçuli"]
  },
  {
    "kod": "LRN.09.092",
    "orijinal_ad": "Tom Ford Ombre Leather",
    "kategori": "Deri, Odunsu",
    "notalar": ["Deri", "Kakule", "Yasemin", "Amber", "Paçuli", "Yosun"]
  },
  {
    "kod": "LRN.09.099",
    "orijinal_ad": "Maison Francis Kurkdjian Oud Silk Mood",
    "kategori": "Amber, Çiçeksi, Odunsu",
    "notalar": ["Ud", "Bulgar Gülü", "Papatya", "Papirüs", "Guaiac Ağacı"]
  },
  {
    "kod": "LRN.09.102",
    "orijinal_ad": "Richard White Chocola",
    "kategori": "Gurme, Vanilya, Çiçeksi",
    "notalar": ["Beyaz Çikolata", "Vanilya", "Badem", "Şeftali", "Fındık", "Orkide"]
  },
  {
    "kod": "LRN.09.106",
    "orijinal_ad": "Tom Ford Electric Cherry",
    "kategori": "Çiçeksi, Meyveli",
    "notalar": ["Vişne", "Zencefil", "Yasemin", "Pembe Biber", "Misk"]
  },
  {
    "kod": "LRN.09.116",
    "orijinal_ad": "Tom Ford Vanilla Sex",
    "kategori": "Amber, Vanilya, Gurme",
    "notalar": ["Vanilya", "Acı Badem", "Sandal Ağacı", "Tonka Fasulyesi", "Çiçeksi Notalar"]
  },
  {
    "kod": "LRN.09.122",
    "orijinal_ad": "Parfums de Marly Layton",
    "kategori": "Amber, Çiçeksi, Baharatlı",
    "notalar": ["Elma", "Vanilya", "Lavanta", "Kakule", "Sandal Ağacı", "Bergamot"]
  },
  {
    "kod": "LRN.09.123",
    "orijinal_ad": "Montale Arabians Tonka",
    "kategori": "Amber, Gurme, Odunsu",
    "notalar": ["Tonka Fasulyesi", "Şeker Kamışı", "Safran", "Ud", "Gül", "Amber"]
  },
  {
    "kod": "LRN.09.127",
    "orijinal_ad": "Kayali Vanilla 28",
    "kategori": "Amber, Vanilya, Gurme",
    "notalar": ["Vanilya Orkidelesi", "Kahverengi Şeker", "Tonka Fasulyesi", "Amber", "Paçuli"]
  },
  {
    "kod": "LRN.09.128",
    "orijinal_ad": "Parfums de Marly Althair",
    "kategori": "Amber, Vanilya, Gurme",
    "notalar": ["Vanilya", "Tarçın", "Pralin", "Kakule", "Portakal Çiçeği", "Misk"]
  },
  {
    "kod": "LRN.09.206",
    "orijinal_ad": "Donna Karan Be Delicious Green",
    "kategori": "Taze, Çiçeksi, Meyveli",
    "notalar": ["Elma", "Salatalık", "Greyfurt", "Manolya", "Gül", "Sandal Ağacı", "Beyaz Amber"]
  },
  {
    "kod": "LRN.09.207",
    "orijinal_ad": "Giorgio Armani Acqua di Gio",
    "kategori": "Aromatik, Akuatik (Deniz), Taze",
    "notalar": ["Deniz Notaları", "Limon", "Bergamot", "Mandalina", "Yasemin", "Beyaz Misk", "Sedir"]
  },
  {
    "kod": "LRN.09.209",
    "orijinal_ad": "Giorgio Armani Si Parfum",
    "kategori": "Şipre, Meyveli, Vanilya",
    "notalar": ["Siyah Frenk Üzümü", "Gül", "Vanilya", "Paçuli", "Frezya", "Mandalina"]
  },
  {
    "kod": "LRN.09.215",
    "orijinal_ad": "Gucci by Flora",
    "kategori": "Çiçeksi, Meyveli, Tatlı",
    "notalar": ["Gardenya", "Armut Çiçeği", "Esmer Şeker", "Kırmızı Meyveler", "Paçuli", "Yasemin"]
  },
  {
    "kod": "LRN.09.217",
    "orijinal_ad": "Guerlain Robe Noir",
    "kategori": "Çiçeksi, Meyveli, Tatlı",
    "notalar": ["Vişne", "Gül", "Badem", "Siyah Frenk Üzümü", "Misk", "Paçuli"]
  },
  {
    "kod": "LRN.09.218",
    "orijinal_ad": "Hermes Terre de Hermes",
    "kategori": "Odunsu, Baharatlı, Narenciye",
    "notalar": ["Portakal", "Greyfurt", "Vetiver", "Biber", "Sedir", "Paçuli"]
  },
  {
    "kod": "LRN.09.222",
    "orijinal_ad": "Lacoste L.12.12 Blanc - White",
    "kategori": "Odunsu, Aromatik, Taze",
    "notalar": ["Greyfurt", "Kakule", "Sümbülteber", "Ylang-Ylang", "Süet", "Vetiver"]
  },
  {
    "kod": "LRN.09.224",
    "orijinal_ad": "Lacoste Pour Femme",
    "kategori": "Çiçeksi, Odunsu, Pudralı",
    "notalar": ["Frezya", "Karabiber", "Yasemin", "Süet", "Sedir Ağacı", "Heliotrop"]
  },
  {
    "kod": "LRN.09.225",
    "orijinal_ad": "Lancome Tresor La Nuit",
    "kategori": "Amber, Gurme, Vanilya",
    "notalar": ["Pralin", "Karamel", "Vanilya", "Orkide", "Gül", "Liçi", "Paçuli", "Kahve"]
  },
  {
    "kod": "LRN.09.226",
    "orijinal_ad": "Lancome La Vie Est Belle",
    "kategori": "Çiçeksi, Gurme, Tatlı",
    "notalar": ["İris", "Pralin", "Vanilya", "Paçuli", "Portakal Çiçeği", "Siyah Frenk Üzümü"]
  },
  {
    "kod": "LRN.09.229",
    "orijinal_ad": "Moschino Love Love",
    "kategori": "Çiçeksi, Odunsu, Narenciye",
    "notalar": ["Greyfurt", "Portakal", "Limon", "Şeker Kamışı", "Misk", "Sedir", "Kırmızı Frenk Üzümü"]
  },
  {
    "kod": "LRN.09.231",
    "orijinal_ad": "Paco Rabanne Invictus",
    "kategori": "Akuatik (Deniz), Odunsu, Taze",
    "notalar": ["Deniz Notaları", "Greyfurt", "Defne Yaprağı", "Ambergris", "Guaiac Ağacı", "Meşe Yosunu"]
  },
  {
    "kod": "LRN.09.233",
    "orijinal_ad": "Paco Rabanne Olympea",
    "kategori": "Amber, Çiçeksi, Gurme",
    "notalar": ["Tuzlu Vanilya", "Su Yasemini", "Mandalina", "Zambak", "Kaşmir Ağacı", "Ambergris"]
  },
  {
    "kod": "LRN.09.234",
    "orijinal_ad": "Paco Rabanne Lady Million",
    "kategori": "Çiçeksi, Meyveli, Tatlı",
    "notalar": ["Bal", "Paçuli", "Portakal Çiçeği", "Ahududu", "Yasemin", "Amber"]
  },
  {
    "kod": "LRN.09.238",
    "orijinal_ad": "Versace Eros",
    "kategori": "Aromatik, Fougère, Taze",
    "notalar": ["Nane", "Yeşil Elma", "Limon", "Tonka Fasulyesi", "Vanilya", "Amber", "Sedir"]
  },
  {
    "kod": "LRN.09.242",
    "orijinal_ad": "Yves Saint Laurent Black Opium",
    "kategori": "Amber, Gurme, Vanilya",
    "notalar": ["Kahve", "Vanilya", "Portakal Çiçeği", "Armut", "Yasemin", "Misk", "Sedir"]
  },
  {
    "kod": "LRN.09.248",
    "orijinal_ad": "Calvin Klein Euphoria",
    "kategori": "Amber, Çiçeksi, Meyveli",
    "notalar": ["Nar", "Siyah Orkide", "Lotus Çiçeği", "Amber", "Misk", "Paçuli", "Maun"]
  },
  {
    "kod": "LRN.09.249",
    "orijinal_ad": "Carrolina Herrera 212 Sexy Magnetik",
    "kategori": "Amber, Çiçeksi, Tatlı",
    "notalar": ["Pamuk Şekeri", "Pembe Biber", "Vanilya", "Misk", "Gardenya", "Sandal Ağacı", "Mandalina"]
  },
  {
    "kod": "LRN.09.251",
    "orijinal_ad": "Carrolina Herrera 212 Sexy",
    "kategori": "Amber, Çiçeksi, Tatlı",
    "notalar": ["Pamuk Şekeri", "Pembe Biber", "Vanilya", "Misk", "Gardenya", "Sandal Ağacı", "Mandalina"]
  },
  {
    "kod": "LRN.09.253",
    "orijinal_ad": "Chanel Bleu de Chanel",
    "kategori": "Aromatik, Odunsu, Taze",
    "notalar": ["Limon", "Bergamot", "Nane", "Zencefil", "Sandal Ağacı", "Sedir", "Amberwood"]
  },
  {
    "kod": "LRN.09.262",
    "orijinal_ad": "Chanel Mademoiselle",
    "kategori": "Amber, Çiçeksi, Şipre",
    "notalar": ["Portakal", "Bergamot", "Yasemin", "Gül", "Paçuli", "Beyaz Misk", "Vetiver"]
  },
  {
    "kod": "LRN.09.263",
    "orijinal_ad": "Chanel Chance Eau Tendre",
    "kategori": "Çiçeksi, Meyveli, Taze",
    "notalar": ["Ayva", "Greyfurt", "Yasemin", "Gül", "Beyaz Misk", "Amber"]
  },
  {
    "kod": "LRN.09.264",
    "orijinal_ad": "Chanel Chance Parfum",
    "kategori": "Şipre, Çiçeksi, Baharatlı",
    "notalar": ["Pembe Biber", "Yasemin", "Paçuli", "Amber", "Beyaz Misk", "Vanilya", "İris"]
  },
  {
    "kod": "LRN.09.270",
    "orijinal_ad": "Emporio Armani Stronger With You",
    "kategori": "Aromatik, Gurme, Vanilya",
    "notalar": ["Kestane", "Vanilya", "Kardamon (Kakule)", "Lavanta", "Pembe Biber", "Adaçayı"]
  },
  {
    "kod": "LRN.09.271",
    "orijinal_ad": "Yves Saint Laurent Libre",
    "kategori": "Amber, Fougère, Çiçeksi",
    "notalar": ["Lavanta", "Portakal Çiçeği", "Mandalina", "Vanilya", "Gri Amber", "Misk"]
  },
  {
    "kod": "LRN.09.274",
    "orijinal_ad": "Burberry Classic (Women)",
    "kategori": "Çiçeksi, Meyveli, Odunsu",
    "notalar": ["Şeftali", "Kayısı", "Siyah Frenk Üzümü", "Yasemin", "Sandal Ağacı", "Misk", "Vanilya"]
  },
  {
    "kod": "LRN.09.275",
    "orijinal_ad": "Burberry Classic Men",
    "kategori": "Aromatik, Odunsu, Taze",
    "notalar": ["Lavanta", "Nane", "Bergamot", "Kekik", "Sandal Ağacı", "Sedir", "Amber", "Misk"]
  },
  {
    "kod": "LRN.09.276",
    "orijinal_ad": "Chloe Love (Story)",
    "kategori": "Çiçeksi, Sabunsu, Taze",
    "notalar": ["Portakal Çiçeği", "Neroli", "Yasemin", "Misk", "Sedir Ağacı", "Armut"]
  },
  {
    "kod": "LRN.09.278",
    "orijinal_ad": "Paco Rabanne Black XS Men",
    "kategori": "Amber, Odunsu, Tatlı",
    "notalar": ["Pralin", "Tarçın", "Siyah Kakule", "Limon", "Adaçayı", "Paçuli", "Siyah Amber"]
  },
  {
    "kod": "LRN.09.285",
    "orijinal_ad": "Bvlgari Man in Black",
    "kategori": "Amber, Baharatlı, Deri",
    "notalar": ["Baharatlar", "Rom", "Tütün", "Deri", "İris", "Tonka Fasulyesi", "Guaiac Ağacı"]
  },
  {
    "kod": "LRN.09.286",
    "orijinal_ad": "Narciso Rodriguez for Her",
    "kategori": "Çiçeksi, Odunsu, Misk",
    "notalar": ["Misk", "Gül", "Şeftali", "Amber", "Sandal Ağacı", "Paçuli"]
  },
  {
    "kod": "LRN.09.288",
    "orijinal_ad": "Jean Paul Gaultier Le Male",
    "kategori": "Amber, Fougère, Aromatik",
    "notalar": ["Lavanta", "Vanilya", "Nane", "Kakule", "Tarçın", "Tonka Fasulyesi", "Sandal Ağacı"]
  },
  {
    "kod": "LRN.09.289",
    "orijinal_ad": "Carolina Herrera 212 Men",
    "kategori": "Odunsu, Misk, Taze Baharatlı",
    "notalar": ["Yeşil Notalar", "Zencefil", "Greyfurt", "Bergamot", "Baharatlar", "Misk", "Sandal Ağacı"]
  },
  {
    "kod": "LRN.09.292",
    "orijinal_ad": "Victoria Secret Bombshell",
    "kategori": "Çiçeksi, Meyveli, Taze",
    "notalar": ["Çarkıfelek", "Ananas", "Greyfurt", "Çilek", "Şakayık", "Vanilya Orkidesi", "Misk"]
  },
  {
    "kod": "LRN.09.293",
    "orijinal_ad": "Victoria Secret Sexy Little (Noir Tease)",
    "kategori": "Çiçeksi, Meyveli, Gurme",
    "notalar": ["Vanilya", "Pralin", "Armut", "Gardenya", "Amber", "Liçi", "Misk"]
  },
  {
    "kod": "LRN.09.298",
    "orijinal_ad": "Lancome Idole Icone (L'Intense)",
    "kategori": "Şipre, Çiçeksi, Odunsu",
    "notalar": ["Gül", "Yasemin", "Misk", "Vanilya", "Paçuli", "Sedir Ağacı", "Acı Portakal"]
  },
  {
    "kod": "LRN.09.299",
    "orijinal_ad": "Narciso Rodriguez Poudree",
    "kategori": "Çiçeksi, Odunsu, Pudralı",
    "notalar": ["Pudralı Notalar", "Misk", "Yasemin", "Gül", "Sedir", "Vetiver", "Kumarin"]
  },
  {
    "kod": "LRN.09.301",
    "orijinal_ad": "Yves Saint Laurent L'Homme",
    "kategori": "Odunsu, Çiçeksi, Misk",
    "notalar": ["Zencefil", "Bergamot", "Limon", "Baharatlar", "Beyaz Biber", "Vetiver", "Sedir"]
  },
  {
    "kod": "LRN.09.304",
    "orijinal_ad": "Issey Miyake Pour Homme",
    "kategori": "Odunsu, Akuatik (Deniz), Narenciye",
    "notalar": ["Yuzu", "Limon", "Bergamot", "Lotus Çiçeği", "Muskat", "Sedir", "Vetiver", "Misk"]
  },
  {
    "kod": "LRN.09.305",
    "orijinal_ad": "Jean Paul Gaultier Scandal US Man",
    "kategori": "Amber, Odunsu, Gurme",
    "notalar": ["Karamel", "Tonka Fasulyesi", "Adaçayı", "Mandalina", "Vetiver"]
  },
  {
    "kod": "LRN.09.306",
    "orijinal_ad": "Jean Paul Gaultier Ultra Male",
    "kategori": "Amber, Fougère, Tatlı",
    "notalar": ["Armut", "Vanilya", "Lavanta", "Tarçın", "Nane", "Amber"]
  },
  {
    "kod": "LRN.09.309",
    "orijinal_ad": "Victor Rolf Spice Bomb",
    "kategori": "Odunsu, Baharatlı, Tütün",
    "notalar": ["Tarçın", "Tütün", "Pembe Biber", "Deri", "Safran", "Bergamot"]
  },
  {
    "kod": "LRN.09.310",
    "orijinal_ad": "Paco Rabane One Million Lucky Man",
    "kategori": "Odunsu, Gurme, Meyveli",
    "notalar": ["Fındık", "Bal", "Erik", "Sedir Ağacı", "Kaşmir", "Greyfurt", "Amberwood"]
  },
  {
    "kod": "LRN.09.313",
    "orijinal_ad": "Jean Paul Gaultier Scandal",
    "kategori": "Şipre, Çiçeksi, Gurme",
    "notalar": ["Bal", "Gardenya", "Kan Portakalı", "Paçuli", "Karamel", "Yasemin"]
  },
  {
    "kod": "LRN.09.314",
    "orijinal_ad": "Giorgio Armani My Way",
    "kategori": "Çiçeksi, Beyaz Çiçek",
    "notalar": ["Sümbülteber", "Portakal Çiçeği", "Bergamot", "Vanilya", "Beyaz Misk", "Sedir"]
  },
  {
    "kod": "LRN.09.315",
    "orijinal_ad": "Roberto Cavalli Eau de Parfum",
    "kategori": "Amber, Çiçeksi, Tatlı",
    "notalar": ["Portakal Çiçeği", "Vanilya", "Benzoin", "Tonka Fasulyesi", "Pembe Biber"]
  },
  {
    "kod": "LRN.09.317",
    "orijinal_ad": "Hugo Boss Intens",
    "kategori": "Odunsu, Baharatlı, Elma",
    "notalar": ["Elma", "Tarçın", "Karanfil", "Sandal Ağacı", "Vanilya", "Bergamot", "Sedir"]
  },
  {
    "kod": "LRN.09.319",
    "orijinal_ad": "Versace Dylan Blue",
    "kategori": "Çiçeksi, Meyveli, Taze",
    "notalar": ["Granny Smith Elma", "Frenk Üzümü Sorbet", "Şakayık", "Gül", "Misk", "Paçuli"]
  },
  {
    "kod": "LRN.09.321",
    "orijinal_ad": "Prada Paradoxe",
    "kategori": "Amber, Çiçeksi, Beyaz Çiçek",
    "notalar": ["Portakal Çiçeği", "Neroli", "Yasemin", "Amber", "Vanilya", "Misk", "Armut"]
  },
  {
    "kod": "LRN.09.323",
    "orijinal_ad": "Cristian Dior Miss Dior Bloming Bouquet",
    "kategori": "Çiçeksi, Taze, Gül",
    "notalar": ["Şakayık", "Gül", "Beyaz Misk", "Bergamot", "Kayısı", "Şeftali"]
  },
  {
    "kod": "LRN.09.326",
    "orijinal_ad": "Giorgio Armani Gio Profumo",
    "kategori": "Aromatik, Akuatik (Deniz), Baharatlı",
    "notalar": ["Deniz Notaları", "Tütsü", "Bergamot", "Biberiye", "Adaçayı", "Paçuli"]
  },
  {
    "kod": "LRN.09.327",
    "orijinal_ad": "Jean Paul Gaultier Le Male Elixir",
    "kategori": "Amber, Fougère, Aromatik",
    "notalar": ["Vanilya", "Bal", "Tütün", "Tonka Fasulyesi", "Lavanta", "Nane", "Benzoin"]
  },
  {
    "kod": "LRN.09.328",
    "orijinal_ad": "Yves Saint Laurent Myself Man",
    "kategori": "Aromatik, Çiçeksi, Taze",
    "notalar": ["Portakal Çiçeği", "Bergamot", "Ambrofix", "Paçuli"]
  },
  {
    "kod": "LRN.09.331",
    "orijinal_ad": "DIOR SAUVAGE ELIXIR",
    "kategori": "Aromatik, Baharatlı, Odunsu",
    "notalar": ["Lavanta", "Tarçın", "Muskat", "Kakule", "Meyan Kökü", "Sandal Ağacı", "Amber"]
  },
  {
    "kod": "LRN.09.332",
    "orijinal_ad": "ARMANI STRONGER WITH YOU ABSOLUTELY",
    "kategori": "Amber, Gurme, Baharatlı",
    "notalar": ["Rom", "Kestane", "Vanilya", "Lavanta", "Paçuli", "Sedir"]
  },
  {
    "kod": "LRN.09.335",
    "orJinal_ad": "BURBERRY GODDESS",
    "kategori": "Aromatik, Vanilya, Gurme",
    "notalar": ["Vanilya", "Lavanta", "Kakao", "Zencefil", "Ginseng"]
  },
  {
    "kod": "LRN.09.336",
    "orijinal_ad": "CAROLINA HERRERA GOOD GIRL BLUSH",
    "kategori": "Şipre, Çiçeksi, Taze",
    "notalar": ["Şakayık", "Gül Suyu", "Vanilya", "Bergamot", "Ylang-Ylang", "Acı Badem"]
  },
  {
    "kod": "LRN.09.338",
    "orijinal_ad": "AZZARO THE MOST WANTED",
    "kategori": "Amber, Baharatlı, Gurme",
    "notalar": ["Karamel (Toffee)", "Kakule", "Amberwood", "Odunsu Notalar"]
  }
]
"""
# --- ADIM 2: VERİTABANINI VE MOTORU YÜKLEME ---
# (Bu fonksiyonlar Colab'daki ile aynı, dokunmuyoruz)

# Veritabanını yükle
try:
    veritabani = json.loads(parfum_veritabani_json)
except json.JSONDecodeError as e:
    st.error(f"Veritabanı (JSON) yüklenirken bir hata oluştu: {e}")
    st.stop() # Hata varsa uygulamayı durdur

# Fonksiyon: Nota ile arama
def nota_ile_parfum_bul(arama_terimi, db):
    sonuclar = []
    arama_terimi = arama_terimi.lower()
    for parfum in db:
        tum_notalar_ve_kategoriler = parfum['kategori'].lower() + " " + " ".join(parfum['notalar']).lower()
        if arama_terimi in tum_notalar_ve_kategoriler:
            sonuclar.append(parfum)
    return sonuclar

# Fonksiyon: Benzerlik motorunu hazırla ve çalıştır
# Streamlit'in önbellekleme (cache) özelliğini kullanıyoruz.
# Bu sayede 76 parfümün benzerlik hesabı her tıklamada değil, sadece 1 kez yapılır.
@st.cache_resource
def benzerlik_motorunu_hazirla(db):
    dokumanlar = [" ".join(p['notalar']) for p in db]
    vectorizer = CountVectorizer()
    notalar_matrix = vectorizer.fit_transform(dokumanlar)
    benzerlik_skorlari = cosine_similarity(notalar_matrix)
    return benzerlik_skorlari

# Motoru çalıştır
benzerlik_skor_matrisi = benzerlik_motorunu_hazirla(veritabani)

# Fonksiyon: Benzerlik önermesi (Hem kod hem isimle)
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
        return None, [] # Hiçbir şey bulunamadıysa

    # Benzerlik skorlarını al
    skorlar = list(enumerate(skor_matrisi[bulunan_index]))
    skorlar = sorted(skorlar, key=lambda x: x[1], reverse=True)
    
    # Kendisi hariç (skorlar[1:]) en benzer 'top_n' taneyi al
    en_benzer_indexler = [i[0] for i in skorlar[1:top_n+1]]
    
    benzer_parfumler = [db[i] for i in en_benzer_indexler]
    return bulunan_parfum, benzer_parfumler # Baz alınan parfümü ve önerileri döndür

# --- ADIM 3: ARAYÜZÜ (WEB SİTESİ) OLUŞTURMA ---

# Sayfa Başlığı
st.set_page_config(page_title="Lorinna Parfüm Danışmanı", layout="wide")
st.title("🤖 Lorinna Yapay Zeka Parfüm Danışmanı")
st.write(f"Şu anda veritabanında {len(veritabani)} adet parfüm yüklü.")

# Arayüzü iki sütuna böl
col1, col2 = st.columns(2)

# --- SÜTUN 1: NOTA VEYA KATEGORİYE GÖRE ARAMA ---
with col1:
    st.header("1. Nota veya Kategoriye Göre Bul")
    st.write("Müşterinin istediği bir nota veya koku tipini yazın (Örn: 'çiçeksi', 'vanilya', 'pudralı', 'ananas')")
    
    # Metin giriş kutusu
    nota_terimi = st.text_input("Aranacak Nota veya Kategori:", key="nota_arama")
    
    # Arama butonu
    if st.button("Parfümleri Bul", key="nota_buton"):
        if nota_terimi:
            sonuclar = nota_ile_parfum_bul(nota_terimi, veritabani)
            if not sonuclar:
                st.warning(f"'{nota_terimi}' içeren parfüm bulunamadı.")
            else:
                st.success(f"'{nota_terimi}' içeren {len(sonuclar)} adet parfüm bulundu:")
                # Sonuçları güzel bir şekilde göster
                for p in sonuclar:
                    st.markdown(f"**{p['kod']} - {p['orijinal_ad']}** (Kategori: *{p['kategori']}*)")
        else:
            st.error("Lütfen aranacak bir terim girin.")

# --- SÜTUN 2: BENZER KOKU ÖNERİSİ ---
with col2:
    st.header("2. Benzer Koku Öner")
    st.write("Müşterinin beğendiği bir parfümün kodunu veya adını yazın (Örn: 'Aventus' veya 'LRN.09.049')")
    
    # Metin giriş kutusu
    isim_terimi = st.text_input("Beğenilen Parfümün Kodu veya Adı:", key="isim_arama")
    
    # Arama butonu
    if st.button("Benzer Öneriler Getir", key="isim_buton"):
        if isim_terimi:
            baz_parfum, benzer_oneriler = benzer_parfumleri_getir(isim_terimi, veritabani, benzerlik_skor_matrisi, top_n=3)
            
            if baz_parfum:
                st.success(f"Baz Alınan Parfüm: **{baz_parfum['kod']} - {baz_parfum['orijinal_ad']}**")
                st.write(f"Bu parfüme en çok benzeyen ilk 3 öneri:")
                
                # Sonuçları güzel bir şekilde göster
                for p in benzer_oneriler:
                    st.markdown(f"**{p['kod']} - {p['orijinal_ad']}**")
                    st.caption(f"Öne çıkan ortak notalar: {', '.join(p['notalar'][:4])}...")
            else:
                st.warning(f"'{isim_terimi}' kodlu veya isimli parfüm bulunamadı.")
        else:
            st.error("Lütfen aranacak bir parfüm girin.")
