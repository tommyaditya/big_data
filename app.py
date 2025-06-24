import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prediksi Harga Laptop", layout="centered")
st.title("💻 Prediksi Kategori Harga Laptop")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("lapdata.csv", encoding="ISO-8859-1")
    df['Price'] = df['Price'].str.replace(',', '').astype(float)
    df['Ram'] = df['Ram'].str.extract('(\\d+)').astype(float)
    df['Size'] = df['Size'].str.extract('(\\d+\\.?\\d*)').astype(float)
    
    le = LabelEncoder()
    df['Processor'] = le.fit_transform(df['Processor'])
    df['Memory'] = le.fit_transform(df['Memory'])
    df['OS'] = le.fit_transform(df['OS'])
    
    df['PriceCategory'] = pd.qcut(df['Price'], q=3, labels=['Murah', 'Sedang', 'Mahal'])
    return df

df = load_data()

# Train model
X = df[['Processor', 'Ram', 'Memory', 'Size']]
y = df['PriceCategory']
model = RandomForestClassifier()
model.fit(X, y)

st.subheader("🔧 Masukkan Spesifikasi Laptop")

proc = st.slider("Kode Prosesor (0–{})".format(df['Processor'].max()), int(df['Processor'].min()), int(df['Processor'].max()), 0)
ram = st.slider("RAM (GB)", int(df['Ram'].min()), int(df['Ram'].max()), 16)
mem = st.slider("Kode Memory (0–{})".format(df['Memory'].max()), int(df['Memory'].min()), int(df['Memory'].max()), 0)
size = st.slider("Ukuran Layar (inch)", float(df['Size'].min()), float(df['Size'].max()), 15.6)

if st.button("🔮 Prediksi"):
    prediction = model.predict([[proc, ram, mem, size]])
    st.success(f"Hasil prediksi kategori harga: **{prediction[0]}**")
