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
    df.columns = df.columns.str.strip()  # Bersihkan nama kolom dari spasi/tab

    # Cleaning data
    df['Price'] = df['Price'].str.replace(',', '').astype(float)
    df['Ram'] = df['Ram'].str.extract('(\\d+)').astype(float)
    df['Size'] = df['Size'].str.extract('(\\d+\\.?\\d*)').astype(float)

    # Encode categorical
    le_processor = LabelEncoder()
    le_memory = LabelEncoder()
    le_os = LabelEncoder()

    df['Processor_Label'] = le_processor.fit_transform(df['Processor'])
    df['Memory'] = le_memory.fit_transform(df['Memory'])
    df['OS'] = le_os.fit_transform(df['OS'])

    df['PriceCategory'] = pd.qcut(df['Price'], q=3, labels=['Murah', 'Sedang', 'Mahal'])

    # Simpan encoder mapping
    processor_mapping = {name: idx for idx, name in enumerate(le_processor.classes_)}

    return df, processor_mapping

df, processor_mapping = load_data()

# Train model
X = df[['Processor_Label', 'Ram', 'Memory', 'Size']]
y = df['PriceCategory']
model = RandomForestClassifier()
model.fit(X, y)

# Tampilkan mapping prosesor → kode
st.subheader("🔑 Mapping Prosesor")
st.write("Gunakan kode berikut saat memasukkan spesifikasi:")
st.table(pd.DataFrame(list(processor_mapping.items()), columns=["Nama Prosesor", "Kode"]))

# Form input
st.subheader("🧾 Masukkan Spesifikasi Laptop")
proc = st.number_input("Kode Prosesor", min_value=0, max_value=int(df['Processor_Label'].max()), value=0)
ram = st.slider("RAM (GB)", int(df['Ram'].min()), int(df['Ram'].max()), 16)
mem = st.slider("Kode Memory", int(df['Memory'].min()), int(df['Memory'].max()), 0)
size = st.slider("Ukuran Layar (inch)", float(df['Size'].min()), float(df['Size'].max()), 15.6)

# Prediksi
if st.button("🔮 Prediksi"):
    pred = model.predict([[proc, ram, mem, size]])
    st.success(f"📌 Prediksi kategori harga laptop: **{pred[0]}**")
