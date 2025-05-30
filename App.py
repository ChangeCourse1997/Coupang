import streamlit as st
import os 


st.title('Lazada scraper')

# Display the image in the Streamlit app
st.write('Done by Chun Shen')
if st.button("Go to Stats page"):
    st.switch_page("pages/1-Scrape.py")

