import streamlit as st
import urllib.request
from gensim.utils import simple_preprocess
import fasttext

with st.sidebar.expander("Project Goals"):
    """
    1. Tyt chtoto написано
    
    """

title = st.text_input('Ввести описание')
st.write('И оно окажется тут ->', title)

url = 'https://github.com/Sekai-no-uragawa/aihack/releases/download/aihack/FastText_baseline.fasttext_model'
filename = url.split('/')[-1]

urllib.request.urlretrieve(url, filename)



model = fasttext.load_model(filename)
text_preproc = ' '.join(simple_preprocess(title))
model.predict(text_preproc)[0][0][9:]