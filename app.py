import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# --- ADIM 1: "YAPAY ZEKA"NIN BEYNİ (96 PARFÜMLÜK TAM VERİTABANI - v1.2) ---
parfum_veritabani_json = """
[
  {
    "kod": "002",
    "orijinal_ad": "Amouage Honour Man",
    "kategori": "Baharatlı, Odunsu, Taze",
    "notalar": ["Pembe Biber", "Sardunya", "Elemi", "Muskat", "Tütsü", "Güve Otu", "Sedir", "Misk", "Tonka Fasulyesi", "Paçuli"]
  },
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
    "notalar": ["Liçi", "Şakayık", "Şeftali", "Portakal Çiçeği", "Misk", "Yasemin"]
  },
  {
    "kod": "012",
    "orijinal_ad": "Frederic Malle Portrait of a Lady",
    "kategori": "Çiçeksi, Amber, Baharatlı",
    "notalar": ["Gül", "Karanfil", "Ahududu", "Siyah Frenk Üzümü", "Tarçın", "Paçuli", "Tütsü", "Sandal Ağacı", "Misk", "Amber", "Benzoin"]
  },
  {
    "kod": "013",
    "orijinal_ad": "Maison Francis Kurkdjian Baccarat Rouge 540",
    "kategori": "Çiçeksi, Odunsu, Amber",
    "notalar": ["Safran", "Yasemin", "Amberwood", "Ambergris", "Reçine", "Sedir"]
  },
  {
    "kod": "021",
    "orijinal_ad": "Nasomatto Black Afgano",
    "kategori": "Odunsu, Tütsü, Baharatlı",
    "notalar": ["Kenevir", "Yeşil Notalar", "Reçine", "Odunsu Notalar", "Tütün", "Kahve", "Ud", "Tütsü"]
  },
  {
    "kod": "024",
    "orijinal_ad": "Tom Ford Black Orchid",
    "kategori": "Amber, Çiçeksi, Gurme",
    "notalar": ["Yasemin", "Gardenya", "Ylang Ylang", "Bergamot", "Frenk Üzümü", "Yumru", "Baharat", "Meyveli Notalar", "Orkide", "Vetiver", "Sandal Ağacı", "Paçuli", "Amber", "Tütsü", "Vanilya", "Çikolata"]
  },
  {
    "kod": "027",
    "orijinal_ad": "Xerjoff Erba Pura",
    "kategori": "Narenciye, Meyveli, Misk",
    "notalar": ["Sicilya Portakalı", "Calabria Bergamotu", "Sicilya Limonu", "Tropikal Meyveler", "Beyaz Misk", "Amber", "Madagaskar Vanilyası"]
  },
  {
    "kod": "031",
    "orijinal_ad": "Memo Paris Marfa",
    "kategori": "Çiçeksi, Odunsu, Misk",
    "notalar": ["Sümbülteber", "Agave", "Vanilya", "Portakal Çiçeği", "Sandal Ağacı", "Beyaz Misk"]
  },
  {
    "kod": "040",
    "orijinal_ad": "Parfums de Marly Delina",
    "kategori": "Çiçeksi, Meyveli, Taze",
    "notalar": ["Liçi", "Rhubarb", "Bergamot", "Muskat", "Türk Gülü", "Şakayık", "Vanilya", "Kaşmir", "Sedir", "Vetiver", "Tütsü", "Misk"]
  },
  {
    "kod": "041",
    "orijinal_ad": "Zadig & Voltaire This is Her",
    "kategori": "Odunsu, Vanilya, Gurme",
    "notalar": ["Yasemin", "Yumuşak Vanilya", "Kestane", "Sandal Ağacı"]
  },
  {
    "kod": "045",
    "orijinal_ad": "Gucci Intense Oud",
    "kategori": "Ud, Amber, Oryantal",
    "notalar": ["Armut", "Ahududu", "Safran", "Bulgar Gülü", "Portakal Çiçeği", "Doğal Ud", "Paçuli"]
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
    "kod": "068",
    "orijinal_ad": "Tom Ford Noir Extreme",
    "kategori": "Amber, Baharatlı, Odunsu",
    "notalar": ["Kakule", "Muskat", "Safran", "Mandalina", "Neroli", "Gül", "Yasemin", "Damla Sakızı", "Vanilya", "Amber", "Odunsu Notalar", "Sandal Ağacı"]
  },
  {
    "kod": "078",
    "orijinal_ad": "Maison Francis Kurkdjian Baccarat Rouge 540 Extrait",
    "kategori": "Amber, Odunsu, Baharatlı",
    "notalar": ["Safran", "Acı Badem", "Mısır Yasemini", "Sedir", "Ambergris", "Misk"]
  },
  {
    "kod": "079",
    "orijinal_ad": "Orto Parisi Megamare",
    "kategori": "Aromatik, Akuatik (Deniz), Misk",
    "notalar": ["Bergamot", "Limon", "Yosun", "Calone", "Hedione", "Ambrox", "Sedir", "Misk"]
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
    "kod": "114",
    "orijinal_ad": "Initio Musk Therapy",
    "kategori": "Misk, Odunsu, Çiçeksi",
    "notalar": ["Bergamot", "Greyfurt", "Sedir Ağacı", "Gül", "Paçuli", "Sandal Ağacı", "Vanilya", "Amber", "Ambergris"]
  },
  {
    "kod": "116",
    "orijinal_ad": "Tom Ford Vanilla Sex",
    "kategori": "Amber, Vanilya, Gurme",
    "notalar": ["Vanilya", "Acı Badem", "Sandal Ağacı", "Tonka Fasulyesi", "Çiçeksi Notalar"]
  },
  {
    "kod": "117",
    "orijinal_ad": "Kilian Angels' Share",
    "kategori": "Gurme, Amber, Baharatlı",
    "notalar": ["Konyak", "Tarçın", "Tonka Fasulyesi", "Meşe", "Pralin", "Vanilya", "Sandal Ağacı"]
  },
  {
    "kod": "120",
    "orijinal_ad": "Marc-Antoine Barrois Tilia",
    "kategori": "Aromatik, Çiçeksi, Odunsu",
    "notalar": ["Lime", "Katırtırnağı (bitki)", "Yasemin", "Vetiver", "Kediotu", "Sedir Ağacı", "Ambroxan"]
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
    "kod": "124",
    "orijinal_ad": "Louis Vuitton Imagination",
    "kategori": "Narenciye, Taze, Odunsu",
    "notalar": ["Limon", "Bergamot", "Yasemin", "Gül", "Frezya", "Sandal Ağacı", "Amber", "Vanilya"]
  },
  {
    "kod": "125",
    "orijinal_ad": "Amouage Guidance",
    "kategori": "Çiçeksi, Oryantal, Gurme",
    "notalar": ["Armut", "Tütsü", "Fındık", "Safran", "Gül", "Yasemin", "Sandal Ağacı", "Amber", "Vanilya"]
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
    "kod": "202",
    "orijinal_ad": "Dolce & Gabbana The One EDP",
    "kategori": "Oryantal, Baharatlı, Odunsu",
    "notalar": ["Greyfurt", "Kişniş", "Fesleğen", "Zencefil", "Kakule", "Portakal Çiçeği", "Tütün", "Amber", "Sedir Ağacı"]
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
    "kod": "208",
    "orijinal_ad": "Giorgio Armani Code Profumo",
    "kategori": "Amber, Baharatlı, Odunsu",
    "notalar": ["Tonka Fasulyesi", "Kakule", "Odunsu Notalar"]
  },
  {
    "kod": "209",
    "orijinal_ad": "Giorgio Armani Si Parfum",
    "kategori": "Şipre, Meyveli, Vanilya",
    "notalar": ["Siyah Frenk Üzümü", "Gül", "Vanilya", "Paçuli", "Frezya", "Mandalina"]
  },
  {
    "kod": "210",
    "orijinal_ad": "Giorgio Armani Si Intense",
    "kategori": "Amber, Çiçeksi, Vanilya",
    "notalar": ["Siyah Frenk Üzümü", "Gül", "Davana", "Vanilya", "Siyah Çay", "Paçuli"]
  },
  {
    "kod": "211",
    "orijinal_ad": "Giorgio Armani Code for Women",
    "kategori": "Çiçeksi, Oryantal",
    "notalar": ["Zambak", "Yasemin", "Taze Zencefil", "Portakal Çiçeği", "Vanilya", "Sandal Ağacı"]
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
    "notalar": ["Kestane", "Vanilya", "Kardamon (Kakule)", "Lavanta", "Pembe Biber", "Adaçayı"]
  },
  {
    "kod": "271",
    "orijinal_ad": "Yves Saint Laurent Libre",
    "kategori": "Amber, Fougère, Çiçeksi",
    "notalar": ["Lavanta", "Portakal Çiçeği", "Mandalina", "Vanilya", "Gri Amber", "Misk"]
  },
  {
    "kod": "274",
    "orijinal_ad": "Burberry Classic (Women)",
    "kategori": "Çiçeksi, Meyveli, Odunsu",
    "notalar": ["Şeftali", "Kayısı", "Siyah Frenk Üzümü", "Yasemin", "Sandal Ağacı", "Misk", "Vanilya"]
  },
  {
    "kod": "275",
    "orijinal_ad": "Burberry Classic Men",
    "kategori": "Aromatik, Odunsu, Taze",
    "notalar": ["Lavanta", "Nane", "Bergamot", "Kekik", "Sandal Ağacı", "Sedir", "Amber", "Misk"]
  },
  {
    "kod": "276",
    "orijinal_ad": "Chloe Love (Story)",
    "kategori": "Çiçeksi, Sabunsu, Taze",
    "notalar": ["Portakal Çiçeği", "Neroli", "Yasemin", "Misk", "Sedir Ağacı", "Armut"]
  },
  {
    "kod": "278",
    "orijinal_ad": "Paco Rabanne Black XS Men",
    "kategori": "Amber, Odunsu, Tatlı",
    "notalar": ["Pralin", "Tarçın", "Siyah Kakule", "Limon", "Adaçayı", "Paçuli", "Siyah Amber"]
  },
  {
    "kod": "285",
    "orijinal_ad": "Bvlgari Man in Black",
    "kategori": "Amber, Baharatlı, Deri",
    "notalar": ["Baharatlar", "Rom", "Tütün", "Deri", "İris", "Tonka Fasulyesi", "Guaiac Ağacı"]
  },
  {
    "kod": "286",
    "orijinal_ad": "Narciso Rodriguez for Her",
    "kategori": "Çiçeksi, Odunsu, Misk",
    "notalar": ["Misk", "Gül", "Şeftali", "Amber", "Sandal Ağacı", "Paçuli"]
  },
  {
    "kod": "288",
    "orijinal_ad": "Jean Paul Gaultier Le Male",
    "kategori": "Amber, Fougère, Aromatik",
    "notalar": ["Lavanta", "Vanilya", "Nane", "Kakule", "Tarçın", "Tonka Fasulyesi", "Sandal Ağacı"]
  },
  {
    "kod": "289",
    "orijinal_ad": "Carolina Herrera 212 Men",
    "kategori": "Odunsu, Misk, Taze Baharatlı",
    "notalar": ["Yeşil Notalar", "Zencefil", "Greyfurt", "Bergamot", "Baharatlar", "Misk", "Sandal Ağacı"]
  },
  {
    "kod": "292",
    "orijinal_ad": "Victoria Secret Bombshell",
    "kategori": "Çiçeksi, Meyveli, Taze",
    "notalar": ["Çarkıfelek", "Ananas", "Greyfurt", "Çilek", "Şakayık", "Vanilya Orkidesi", "Misk"]
  },
  {
    "kod": "293",
    "orijinal_ad": "Victoria Secret Sexy Little (Noir Tease)",
    "kategori": "Çiçeksi, Meyveli, Gurme",
    "notalar": ["Vanilya", "Pralin", "Armut", "Gardenya", "Amber", "Liçi", "Misk"]
  },
  {
    "kod": "298",
    "orijinal_ad": "Lancome Idole Icone (L'Intense)",
    "kategori": "Şipre, Çiçeksi, Odunsu",
    "notalar": ["Gül", "Yasemin", "Misk", "Vanilya", "Paçuli", "Sedir Ağacı", "Acı Portakal"]
  },
  {
    "kod": "299",
    "orijinal_ad": "Narciso Rodriguez Poudree",
    "kategori": "Çiçeksi, Odunsu, Pudralı",
    "notalar": ["Pudralı Notalar", "Misk", "Yasemin", "Gül", "Sedir", "Vetiver", "Kumarin"]
  },
  {
    "kod": "301",
    "orijinal_ad": "Yves Saint Laurent L'Homme",
    "kategori": "Odunsu, Çiçeksi, Misk",
    "notalar": ["Zencefil", "Bergamot", "Limon", "Baharatlar", "Beyaz Biber", "Vetiver", "Sedir"]
  },
  {
    "kod": "304",
    "orijinal_ad": "Issey Miyake Pour Homme",
    "kategori": "Odunsu, Akuatik (Deniz), Narenciye",
    "notalar": ["Yuzu", "Limon", "Bergamot", "Lotus Çiçeği", "Muskat", "Sedir", "Vetiver", "Misk"]
  },
  {
    "kod": "305",
    "orijinal_ad": "Jean Paul Gaultier Scandal US Man",
    "kategori": "Amber, Odunsu, Gurme",
    "notalar": ["Karamel", "Tonka Fasulyesi", "Adaçayı", "Mandalina", "Vetiver"]
  },
  {
    "kod": "306",
    "orijinal_ad": "Jean Paul Gaultier Ultra Male",
    "kategori": "Amber, Fougère, Tatlı",
    "notalar": ["Armut", "Vanilya", "Lavanta", "Tarçın", "Nane", "Amber"]
  },
  {
    "kod": "309",
    "orijinal_ad": "Victor Rolf Spice Bomb",
    "kategori": "Odunsu, Baharatlı, Tütün",
    "notalar": ["Tarçın", "Tütün", "Pembe Biber", "Deri", "Safran", "Bergamot"]
  },
  {
    "kod": "310",
    "orijinal_ad": "Paco Rabane One Million Lucky Man",
    "kategori": "Odunsu, Gurme, Meyveli",
    "notalar": ["Fındık", "Bal", "Erik", "Sedir Ağacı", "Kaşmir", "Greyfurt", "Amberwood"]
  },
  {
    "kod": "313",
    "orijinal_ad": "Jean Paul Gaultier Scandal",
    "kategori": "Şipre, Çiçeksi, Gurme",
    "notalar": ["Bal", "Gardenya", "Kan Portakalı", "Paçuli", "Karamel", "Yasemin"]
  },
  {
    "kod": "314",
    "orijinal_ad": "Giorgio Armani My Way",
    "kategori": "Çiçeksi, Beyaz Çiçek",
    "notalar": ["Sümbülteber", "Portakal Çiçeği", "Bergamot", "Vanilya", "Beyaz Misk", "Sedir"]
  },
  {
    "kod": "315",
    "orijinal_ad": "Roberto Cavalli Eau de Parfum",
    "kategori": "Amber, Çiçeksi, Tatlı",
    "notalar": ["Portakal Çiçeği", "Vanilya", "Benzoin", "Tonka Fasulyesi", "Pembe Biber"]
  },
  {
    "kod": "317",
    "orijinal_ad": "Hugo Boss Intens",
    "kategori": "Odunsu, Baharatlı, Elma",
    "notalar": ["Elma", "Tarçın", "Karanfil", "Sandal Ağacı", "Vanilya", "Bergamot", "Sedir"]
  },
  {
    "kod": "319",
    "orijinal_ad": "Versace Dylan Blue",
    "kategori": "Çiçeksi, Meyveli, Taze",
    "notalar": ["Granny Smith Elma", "Frenk Üzümü Sorbet", "Şakayık", "Gül", "Misk", "Paçuli"]
  },
  {
    "kod": "321",
    "orijinal_ad": "Prada Paradoxe",
    "kategori": "Amber, Çiçeksi, Beyaz Çiçek",
    "notalar": ["Portakal Çiçeği", "Neroli", "Yasemin", "Amber", "Vanilya", "Misk", "Armut"]
  },
  {
    "kod": "323",
    "orijinal_ad": "Cristian Dior Miss Dior Bloming Bouquet",
    "kategori": "Çiçeksi, Taze, Gül",
    "notalar": ["Şakayık", "Gül", "Beyaz Misk", "Bergamot", "Kayısı", "Şeftali"]
  },
  {
    "kod": "326",
    "orijinal_ad": "Giorgio Armani Gio Profumo",
    "kategori": "Aromatik, Akuatik (Deniz), Baharatlı",
    "notalar": ["Deniz Notaları", "Tütsü", "Bergamot", "Biberiye", "Adaçayı", "Paçuli"]
  },
  {
    "kod": "327",
    "orijinal_ad": "Jean Paul Gaultier Le Male Elixir",
    "kategori": "Amber, Fougère, Aromatik",
    "notalar": ["Vanilya", "Bal", "Tütün", "Tonka Fasulyesi", "Lavanta", "Nane", "Benzoin"]
  },
  {
    "kod": "328",
    "orijinal_ad": "Yves Saint Laurent Myself Man",
    "kategori": "Aromatik, Çiçeksi, Taze",
    "notalar": ["Portakal Çiçeği", "Bergamot", "Ambrofix", "Paçuli"]
  },
  {
    "kod": "331",
    "orijinal_ad": "DIOR SAUVAGE ELIXIR",
    "kategori": "Aromatik, Baharatlı, Odunsu",
    "notalar": ["Lavanta", "Tarçın", "Muskat", "Kakule", "Meyan Kökü", "Sandal Ağacı", "Amber"]
  },
  {
    "kod": "332",
    "orijinal_ad": "ARMANI STRONGER WITH YOU ABSOLUTELY",
    "kategori": "Amber, Gurme, Baharatlı",
    "notalar": ["Rom", "Kestane", "Vanilya", "Lavanta", "Paçuli", "Sedir"]
  },
  {
    "kod": "335",
    "orijinal_ad": "BURBERRY GODDESS",
    "kategori": "Aromatik, Vanilya, Gurme",
    "notalar": ["Vanilya", "Lavanta", "Kakao", "Zencefil", "Ginseng"]
  },
  {
    "kod": "336",
    "orijinal_ad": "CAROLINA HERRERA GOOD GIRL BLUSH",
    "kategori": "Şipre, Çiçeksi, Taze",
    "notalar": ["Şakayık", "Gül Suyu", "Vanilya", "Bergamot", "Ylang-Ylang", "Acı Badem"]
  },
  {
    "kod": "338",
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
st.title("🤖 Lorinna Yapay Zeka Parfüm Danışmanı (v1.2)")
st.write(f"Şu anda veritabanında **{len(veritabani)}** adet parfüm yüklü.")

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
    st.write("Müşterinin beğendiği bir parfümün kodunu veya adını yazın (Örn: 'Aventus' veya '049')")
    
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

# --- KODUN SONU ---
