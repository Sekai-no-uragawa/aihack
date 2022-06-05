import streamlit as st
import urllib.request
import fasttext
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import pymorphy2
import re
import base64
import uuid
import json
from download_description import description_predict_from_file
from PIL import Image


st.set_page_config(
    page_title="ТН ВЭД ЕАЭС",
    page_icon="📦",
    layout="wide",
)

def download_button(object_to_download, download_filename, button_text):
    """
    Generates a link to download the given object_to_download.
    From: https://discuss.streamlit.io/t/a-download-button-with-custom-css/4220
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """
    # if pickle_it:
    #    try:
    #        object_to_download = pickle.dumps(object_to_download)
    #    except pickle.PicklingError as e:
    #        st.write(e)
    #        return None

    # if:
    if isinstance(object_to_download, bytes):
        pass

    elif isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False, encoding="CP1252")
    # Try JSON encode for everything else
    else:
        object_to_download = json.dumps(object_to_download)

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()
    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = (
        custom_css
        + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br><br>'
    )
    # dl_link = f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}"><input type="button" kind="primary" value="{button_text}"></a><br></br>'

    st.markdown(dl_link, unsafe_allow_html=True)

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
    url = 'https://github.com/Sekai-no-uragawa/aihack/releases/download/v1.0.1/train_all_data_default_set.fasttext_model'
    filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, filename)
    return filename

@st.cache
def load_classifier():
    return pd.read_excel('data/tnved-CIS-02.xls')

@st.cache
def load_code_text():
    df_code = pd.read_csv('data/code_text.csv', dtype={'code': 'str'})
    dict_code = df_code.set_index('code').T.to_dict('list')
    return dict_code

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
    
    # Sidebar -- Image/Title
    icon = Image.open("data/1024px-Emblema_fts_2020.png")
    st.sidebar.image(
        icon, use_column_width=True, caption='AiHacks 2022'
    )

    st.sidebar.markdown(
        '''
        Автоматическая система, основанная на алгоритмах машинного обучения (Fasttext)

        Данная система позволяет:
        1. Сократить время для принятия решения декларантом
        2. Уменьшить вероятность ошибки человека.

        Возможно применение в двух сценариях - как для помощи декларанту, так и использование сотрудниками Таможенной Службы для проверки поступающих деклараций.

        Developed by team **fit_predict**

        2022 г.
        '''
    )

    c10, c20, c30 = st.columns([1, 4, 1])
    with c20:
        st.title("Определение кода товара ТН ВЭД ЕАЭС")
        st.write('''
        Вы можете ввести описание вашего товара в поле слева и получить предполагаемый код из классификатора ВЭД.

        А также загрузить файл в формате .csv в окно справа и получить на выходе таблицу "Описание : код"
        ''')
    
    _, c1, c2, _= st.columns([1, 2, 2, 1])
    
    with c1:
        title = st.text_area(
                'Ввести описание товара: ',
            )
        pred_button = st.button('Получить код')

    c100, c200, c300 = st.columns([1, 4, 1])
    if pred_button:
        with c200:
            filename = load_model()
            model = fasttext.load_model(filename)
            text_preproc = ' '.join(preprocessing(title))
            ans = model.predict(text_preproc, k=5)

            st.subheader('Возможные классификационные коды, в порядке убывания уверенности модели')
            for_print = []
            for label, prob in zip(*ans):
                for_print.append([label[9:], round(prob,3)])
            dict_code = load_code_text()
            df = pd.DataFrame(for_print, columns=['Код', 'Точность'])
            df['Описание категории'] = df['Код'].map(dict_code)
            st.dataframe(df)    
    
    with c2:
        uploaded_file = st.file_uploader(
            "",
            key="1",
            help="Drop file here",
        )
    
    c100, c200, c300 = st.columns([2, 4, 2])
    if uploaded_file is not None:
        with c200:
            file_container = st.expander("Просмотрите данные, что Вы загрузили в .csv")
            shows = pd.read_csv(uploaded_file, sep=';')
            uploaded_file.seek(0)
            file_container.write(shows)
            
            filename = load_model()
            model = fasttext.load_model(filename)
            classifier = load_classifier()

            data_to_download = description_predict_from_file(shows, model, preprocessing)
            # df_xlsx = to_excel(data_to_download)
            # st.download_button(label='📥 Download Current Result',
            #                                 data=df_xlsx ,
            #                                 file_name= 'df_test.xlsx')
            CSVButton = download_button(
                data_to_download,
                "File.csv",
                "Download to CSV",
            )
    else:
        with c2:
            st.info(
                f"""
                    👆 Загрузите .csv файл. Файл для примера: [for_test.csv](https://drive.google.com/file/d/1luxXwS7hRtCHDZLWG5pN2ybx1qrV2cxm/view?usp=sharing)
                    """
            )
            st.stop()
         
    
    
            # cat = [i[9:].replace('_', '') for i in ans]
            
            # classifier = load_classifier()
            # description = classifier[classifier.TNVED.isin(cat)].FULL_TEXT.tolist()
            # while len(description) < 5:
            #     description.append('unknown')
            
            # st.write('1.', ans[0][9:], '- описание:', description[0])
            # st.write('2.', ans[1][9:], '- описание:', description[1])
            # st.write('3.', ans[2][9:], '- описание:', description[2])
            # st.write('4.', ans[3][9:], '- описание:', description[3])
            # st.write('5.', ans[4][9:], '- описание:', description[4])

if __name__ == '__main__':
    main()  