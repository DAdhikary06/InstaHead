import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="InstaHead",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.logo('logo.png',size='large')

Headline_page = st.Page("pages/1_Headline_Generator.py", title="Headline Generation",icon="📰")
Analytics_page = st.Page("pages/2_Analytics_Page.py", title="Model Analytics", icon="📊")


pg = st.navigation([Headline_page,Analytics_page])

pg.run()