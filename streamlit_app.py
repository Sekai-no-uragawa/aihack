import streamlit as st
import urllib.request
import fasttext

from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import pymorphy2
import re


@st.cache
def get_nltk():
    nltk.download('punkt')
    nltk.download('stopwords')

@st.cache
def load_model():
    url = 'https://github.com/Sekai-no-uragawa/aihack/releases/download/v1.0.1/FastText_top5.fasttext_model'
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)
    return filename

def preprocessing(x):
    get_nltk()
    morph = pymorphy2.MorphAnalyzer()
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

def main():
    st.set_page_config(
        page_title="ТН ВЭД ЕАЭС",
        page_icon="📦",
    )
    
    with st.sidebar.expander("Info"):
        """
        Developed by team fit_predict

        2022
        """
    
    title = st.text_area(
            'Ввести описание товара: ',
        )
    st.write('И оно окажется тут:')
    st.write(title)
    pred_button = st.button('Get Predict!')    

    if pred_button:
        filename = load_model()
        model = fasttext.load_model(filename)
        text_preproc = ' '.join(preprocessing(title))
        ans = model.predict(text_preproc, k=5)[0]

        st.write('А вот и предсказание (топ-5 по вероятности)')
        if ans:
            st.write('1.', ans[0][9:])
            st.write('2.', ans[1][9:])
            st.write('3.', ans[2][9:])
            st.write('4.', ans[3][9:])
            st.write('5.', ans[4][9:])

if __name__ == '__main__':
    main()