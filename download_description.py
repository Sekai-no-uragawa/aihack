def description_predict_from_file(df, model, preprocessing):
    '''docstring
    Функция получает датафрейм с описанием товара, обрабатывает каждое описание и выдает по нему
    предполагаемый код ВЭД. На выходе функция выдает файл эксель с загруженным описанием
    кодами ВЭД которые модель посчитала подходящими.
        Input: file format excel.
        Output: file format excel. 
    '''
    # проверим количество колонок в датасете
    if len(df.columns) != 1: #  выкинем ошибку, если передан не тот файл
        print('Был получен неверный файл. В файле должна быть одна колонка с описанием товара.') 

    list_for_ans = []
    for description in df[df.columns[0]].to_list():
        print(description)
        text_preproc = ' '.join(preprocessing(description))
        answer, prob = model.predict(text_preproc, k=1)
        list_for_ans.append([description, answer[0][9:], round(float(prob), 3)])
    df = pd.DataFrame(list_for_ans, columns=['Описание','ТНВЭД','Точность'])
    
    return df