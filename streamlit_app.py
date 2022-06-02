import streamlit as st
import urllib.request
import fasttext

from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import pymorphy2
import re

nltk.download('punkt')
nltk.download('stopwords')
morph = pymorphy2.MorphAnalyzer()

with st.sidebar.expander("Info"):
    """
    Team fit_predict

    Another text
    """

title = st.text_input('Ввести описание')
st.write('И оно окажется тут:')
st.write(title)
st.write('А вот и предсказание (топ-5 по вероятности)')


url = 'https://github.com/Sekai-no-uragawa/aihack/releases/download/v1.0.1/FastText_top5.fasttext_model'
filename = url.split('/')[-1]

urllib.request.urlretrieve(url, filename)

def preprocessing(x):
    text = x
    if text != None:
        tock_dirt = word_tokenize(text, language="russian")
        morph_lst = []
        tock = []
        for word in tock_dirt:
            word = re.sub("[^A-Za-zА-Яа-я]", " ", word)
            for i in word.split():
                if i != []:
                    if i not in stopwords.words("russian"):
                        morph_lst.append(morph.parse(i)[0].normal_form)
        return morph_lst
    else:
        return []

text_preproc = ' '.join(preprocessing(title))

model = fasttext.load_model(filename)
ans = model.predict(text_preproc, k=5)[0]

if ans:
    st.write('1.', ans[0][9:])
    st.write('2.', ans[1][9:])
    st.write('3.', ans[2][9:])
    st.write('4.', ans[3][9:])
    st.write('5.', ans[4][9:])