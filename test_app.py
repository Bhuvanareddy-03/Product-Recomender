import streamlit as st

st.title("🔧 Streamlit Test App")

name = st.text_input("Enter your name")
if st.button("Say hello"):
    st.write(f"Hello, {name}!")
