import streamlit as st

st.set_page_config(page_title="My App", layout="wide")

st.title("Welcome to Insyt-Labs!")
st.sidebar.success("Select a page above.")

# Centered image (replace 'logo.png' with your image path or URL)
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("BSC_Logo.jpeg", use_container_width=True)
    st.caption("Built by Aryan Nangia")