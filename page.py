import streamlit as st
from st_pages import add_page_title, get_nav_from_toml

st.set_page_config(layout="wide")

# sections = st.sidebar.toggle("Sections", value=True, key="use_sections")

nav = get_nav_from_toml( "pages_sections.toml")

st.logo("logo.png")

pg = st.navigation(nav)

add_page_title(pg)

pg.run()

# streamlit run page.py