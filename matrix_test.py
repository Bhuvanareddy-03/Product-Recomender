import streamlit as st
import pandas as pd

st.title("🔍 Matrix Test")

uploaded_file = st.file_uploader("Upload your ratings CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = ['userId', 'productId', 'rating', 'timestamp']
    df.drop(columns=['timestamp'], inplace=True)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df.dropna(inplace=True)

    matrix = df.pivot_table(index='userId', columns='productId', values='rating').fillna(0)
    st.write("Matrix shape:", matrix.shape)

    if matrix.shape[0] > 1:
        selected_user = st.selectbox("Select a User ID", matrix.index)
        st.write(f"You selected user: {selected_user}")
