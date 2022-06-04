import streamlit as st
import urllib.request
import fasttext
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import pymorphy2
import re

st.set_page_config(
    page_title="ТН ВЭД ЕАЭС",
    page_icon="📦",
    layout="wide",
)

def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
_max_width_()

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

@st.cache
def load_classifier():
    return pd.read_excel('tnved-CIS-02.xls')

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
    
    
    
    with st.sidebar.expander("Info"):
        """
        Developed by team fit_predict

        2022
        """
    c10, c20, c30 = st.columns([1, 4, 1])
    with c20:
        st.title("Определение кода товара ТН ВЭД ЕАЭС")
        st.write('Супер проект ...')
    
    c1, c2 = st.columns([3, 3])
    with c1:
        title = st.text_area(
                'Ввести описание товара: ',
            )

    with c2:
        pred_button = st.button('Get Predict!')    
    
    
    uploaded_file = st.file_uploader(
    "",
    key="1",
    help="Drop file here",
    )

    if uploaded_file is not None:
        file_container = st.expander("Check your uploaded .csv")
        shows = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)
        file_container.write(shows)

    if pred_button:
        filename = load_model()
        model = fasttext.load_model(filename)
        text_preproc = ' '.join(preprocessing(title))
        ans = model.predict(text_preproc, k=5)[0]

        st.write('А вот и предсказание (топ-5 по вероятности)')
        
        cat = [i[9:].replace('_', '') for i in ans]
        
        classifier = load_classifier()
        description = classifier[classifier.TNVED.isin(cat)].FULL_TEXT.tolist()
        while len(description) < 5:
            description.append('unknown')
        
        st.write('1.', ans[0][9:], '- описание:', description[0])
        st.write('2.', ans[1][9:], '- описание:', description[1])
        st.write('3.', ans[2][9:], '- описание:', description[2])
        st.write('4.', ans[3][9:], '- описание:', description[3])
        st.write('5.', ans[4][9:], '- описание:', description[4])

if __name__ == '__main__':
    main()  