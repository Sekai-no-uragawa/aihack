## Решение кейса "Исскуственный интеллект проводит таможенный контроль" от команды fit_predict. 

**4-е место в соревновании**

[Ссылка на решение, прототип веб-сервиса по определению кода ТНВЭД по описанию товара](https://share.streamlit.io/sekai-no-uragawa/aihack/main)

# Содержание

- [Задача](#task1)
- [Как мы решили кейс](#task2)
    - [Скриншоты](#task2_1)
    - [Что может наше решение](#task2_2)
    - [Что мы хотели бы добавить](#task2_3)
- [Соответствие техническим критериям](#task4)
    - [Проверка кода на запускаемость](#task4_1)
    - [Какую выбрали модель/метод](#task4_2)
    - [Какую метрику выбрали для проверки точности](#task4_3)
    - [Масштабируемость нашего решения](#task4_4)
    - [На основе каких библиотек/ПО построено решение](#task4_5)

# Задача <a class="anchor" id="task1"></a>
Необходимо создать интеллектуальную систему для помощи таможенному инспектору в принятии решения по определению кода ТНВЭД. Система должна определять код товара по описанию на основе датасета переданного для анализа текстового описания товаров.

# Как мы решили кейс <a class="anchor" id="task2"></a>
Интеллектуальная система реализована в виде алгоритма, который имеет веб-интерфейс и может определять код товара по его описанию.

## Скриншоты решения <a class="anchor" id="task2_1"></a>
![screenshot](screenshot.png 'Скриншот')   

## Что может наше решение <a class="anchor" id="task2_2"></a>
Реализованный веб-сервис имеет несколько основных функций:
1. Принимает описание товара и определяет соответствующий описанию код;
2. Принимает файл со списком описаний, определяет код для каждого описания и передает пользователю файл в котором для каждого описания определен код;
3. Принимает код и выдает пользователю описание этого кода;
4. Принимает код и описание и сообщает пользователю правильно ли определена пара код-описание. 
5. Принимает код и описание и в случае если пара код-описание не совпадают по определению модели, то предлагает наиболее вероятные коды ТНВЭД для данного описания.

## Что мы хотели бы добавить <a class="anchor" id="task2_3"></a>
Дальнейшее развитие нашего сервиса мы видим в интеграции с ЛК декларанта и проверяющего в системе таможенной службы. Пользователь должен иметь возможность при подаче декларации, не переходя в сторонние сервисы, проверить правильность введенного кода и получить рекомендации по его изменению. С другой стороны проверяющий должен получать от системы подсказки соответствует ли введенное описание предоставленному декларантом коду.

# Соответствие техническим критериям <a class="anchor" id="task4"></a>

## Проверка кода на запускаемость <a class="anchor" id="task4_1"></a>
Веб-сервис представлен в рабочем виде, исходный код представлен в репозитории. Код модели подготовлен в colab. Для открытия и запуска кода модели необходимо:
1. Открыть файл по ссылке ниже;
2. Нажать в открывшемся окне "открыть в приложении colab", в верхней части экрана.

- **[clear_data.py](https://drive.google.com/file/d/1vMxMdw1QUYDBPi5CGuJmuatCuvrUjTWO/view?usp=sharing)** - ноутбук для очистки данных от повторяющихся значений, знаков препинания и лишних символов.
- **[preprocessing_data.ipynb](https://drive.google.com/file/d/1ny7R-A4mXfHOYRCHjWHAPx-BPZP-UqVd/view?usp=sharing)** - ноутбук для препроцессинга данных, токенизации и лемматизации.
- **[train_model.ipynb](https://drive.google.com/file/d/1Cv4xctl9MTV83WChOGFpj1E7OpzTh9QI/view?usp=sharing)** - основной файл, где обучается модель.

## Какую выбрали модель/метод <a class="anchor" id="task4_2"></a>
Для построения модели нами изначально было выбрано 3 алгоритма для проведения экспериментов и нахождения лучшего из них. Были протестированы такие алгоритмы как FastText, SVD, CatBoost. Мы перебрали некоторое количество параметров и обучили каждую модель не по одному разу. В итоге, нами была выбрана модель на основе алгоритма FastText, так как она показала лучшие результаты по метрике F1.

FastText работает с векторным представлением слов и содержит предобученные векторные представления этих слов. Этот алгоритм работает несколько быстрее чем алгоритмы SVD и CatBoost. Две основные особенности которые использует алгоритм - негативное сэмплирование и subword-модель.

Негативное сэмплирование - это подбор в цепочках слов таких сочетаний которые не являются соседями по контексту исходя из данных. Например, "шерстяной свитер" - это положительный сэмпл, а "шерстяная лопата" отрицательный.

Subword-модель разбивает каждое слово на цепочки (n-граммы) от 3 до 6 символов, объединяет это в один список и добавляет в конце слово целиком. Это позволяет модели отсеять слова которые могут начинаться с одинаковых символов, но при этом иметь разный смысл. Также подобный подход помогают алгоритму работать со словами которые он раньше не встречал.

## Какую метрику выбрали для проверки точности <a class="anchor" id="task4_3"></a>
Была выбрана F-мера, на итоговой модели мы получили значение метрики 0.91 (91%).

## Масштабируемость нашего решения <a class="anchor" id="task4_4"></a>
Основная проблема в масштабируемости - это данные. Их где-то нужно хранить, выгружать и передавать модели. Вычислительные мощности не помогут масштабироваться если нет достаточного количества данных. Мы подготовили решение в виде эскизной базы данных, которую мы разработали для модели.  

Мы подготовили схему базы данных, запросы для создания БД и выгрузки из нее нужной для модели информации. Также нами был подготовлен python-скрипт который позволяет выгрузить данные из БД и передать их модели. Более подробно описано в папке этого репозитория sql_database, либо можно перейти по [этой ссылке](https://github.com/Sekai-no-uragawa/aihack/tree/main/sql_database).

## На основе каких библиотек/ПО построено решение  <a class="anchor" id="task4_5"></a>
Все использованные библиотеки предоставляются с открытым исходным кодом. Список библиотек можно найти в файле [requirements.txt](https://github.com/Sekai-no-uragawa/aihack/blob/main/requirements.txt)
