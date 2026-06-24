
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Titanic Data Analysis", layout="wide")

st.title("🚢 Titanic Streamlit Dashboard")

st.write("Upload file CSV Titanic atau gunakan sample data yang tersedia.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("titanic_sample.csv")

st.subheader("Preview Data")
st.dataframe(df.head())

st.subheader("Informasi Dataset")
st.write(df.describe(include="all"))

if "Survived" in df.columns:
    st.subheader("Distribusi Survival")

    fig, ax = plt.subplots()
    df["Survived"].value_counts().plot(kind="bar", ax=ax)
    ax.set_xlabel("Survived")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

if "Age" in df.columns:
    st.subheader("Distribusi Umur")

    fig2, ax2 = plt.subplots()
    df["Age"].dropna().plot(kind="hist", bins=10, ax=ax2)
    ax2.set_xlabel("Age")
    st.pyplot(fig2)

st.success("Dashboard berhasil dijalankan di Streamlit 🚀")
