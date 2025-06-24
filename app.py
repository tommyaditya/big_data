import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

st.title("💻 Analisis Harga Laptop")

# Upload CSV
uploaded_file = st.file_uploader("Upload dataset CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')

    # Preprocessing
    df['Price'] = df['Price'].str.replace(',', '').astype(float)
    df['Ram'] = df['Ram'].str.extract('(\\d+)').astype(float)
    df['Size'] = df['Size'].str.extract('(\\d+\\.?\\d*)').astype(float)

    le = LabelEncoder()
    df['Processor'] = le.fit_transform(df['Processor'])
    df['Memory'] = le.fit_transform(df['Memory'])
    df['OS'] = le.fit_transform(df['OS'])

    df['PriceCategory'] = pd.qcut(df['Price'], q=3, labels=['Murah', 'Sedang', 'Mahal'])

    st.subheader("📊 Data Overview")
    st.dataframe(df.head())

    # Visualisasi
    st.subheader("📈 Visualisasi Harga")
    fig1, ax1 = plt.subplots()
    sns.boxplot(x='PriceCategory', y='Price', data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("📈 Distribusi Tipe Prosesor")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.countplot(x='Processor', data=df, ax=ax2)
    st.pyplot(fig2)

    st.subheader("📈 Korelasi Fitur")
    fig3, ax3 = plt.subplots()
    sns.heatmap(df[['Processor', 'Ram', 'Memory', 'Size', 'Price']].corr(), annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    # Modeling
    X = df[['Processor', 'Ram', 'Memory', 'Size']]
    y = df['PriceCategory']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    st.subheader("🎯 Prediksi Kategori Harga")
    proc = st.slider("Processor (kode)", int(df['Processor'].min()), int(df['Processor'].max()))
    ram = st.slider("RAM (GB)", int(df['Ram'].min()), int(df['Ram'].max()))
    mem = st.slider("Memory (kode)", int(df['Memory'].min()), int(df['Memory'].max()))
    size = st.slider("Ukuran Layar (inch)", float(df['Size'].min()), float(df['Size'].max()))

    pred = model.predict([[proc, ram, mem, size]])
    st.write(f"📌 Prediksi kategori harga: **{pred[0]}**")