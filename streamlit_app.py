import streamlit as st
import streamlit.components.v1 as components
import pickle
import pandas as pd

with st.sidebar.expander("Project Goals"):
    """
    1. Tyt chtoto написано
    
    """

title = st.text_input('Ввести описание')
st.write('И оно окажется тут ->', title)