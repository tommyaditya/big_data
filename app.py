import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Prediksi Harga Laptop", layout="centered")
st.title("💻 Prediksi Kategori Harga Laptop")

# Load & preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("lapdata.csv", encoding="ISO-8859-1")
    df.columns = df.columns.str.strip()

    # Preprocessing dasar
    df['Price'] = df['Price'].str.replace(',', '').astype(float)
    df['Ram'] = df['Ram'].str.extract('(\d+)').astype(float)
    df['Size'] = df['Size'].str.extract('(\d+\.?\d*)').astype(float)

    # Ekstrak generasi prosesor (e.g. 11th Gen)
    df['Processor_Gen'] = df['Processor'].str.extract(r'(\d+)(?:th|st|nd|rd)\s+Gen').astype(float)

    # Fallback jika gagal: ambil angka pertama dari string prosesor
    fallback_gen = df['Processor'].str.extract(r'(\d+)').iloc[:, 0].astype(float)
    df['Processor_Gen'] = df['Processor_Gen'].fillna(fallback_gen)

    # Encode kolom Memory dan OS
    le_mem = LabelEncoder()
    le_os = LabelEncoder()
    df['Memory'] = le_mem.fit_transform(df['Memory'])
    df['OS'] = le_os.fit_transform(df['OS'])

    # Buat kategori harga
    df['PriceCategory'] = pd.qcut(df['Price'], q=3, labels=['Murah', 'Sedang', 'Mahal'])

    return df

# Muat data
df = load_data()

# Train model
X = df[['Processor_Gen', 'Ram', 'Memory', 'Size']]
y = df['PriceCategory']
model = RandomForestClassifier()
model.fit(X, y)

# UI
st.subheader("🧾 Masukkan Spesifikasi Laptop")

proc_gen = st.slider("Generasi Prosesor (misal: 11 untuk 11th Gen)", 1, 15, 11)
ram = st.slider("RAM (GB)", int(df['Ram'].min()), int(df['Ram'].max()), 16)
mem = st.slider("Kode Memory", int(df['Memory'].min()), int(df['Memory'].max()), 0)
size = st.slider("Ukuran Layar (inch)", float(df['Size'].min()), float(df['Size'].max()), 15.6)

if st.button("🔮 Prediksi"):
    pred = model.predict([[proc_gen, ram, mem, size]])
    st.success(f"📌 Prediksi kategori harga laptop: **{pred[0]}**")
