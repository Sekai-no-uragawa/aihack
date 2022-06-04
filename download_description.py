def description_predict_from_file(df, model, classifier, preprocessing, count_pred: int = 3):
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
        #  и что произойдет если мы вернем фразу?
        
    #  создадим словари с нужным количеством предсказаний
    result_pred = {str(n) + '_result': [] for n in range(1, count_pred + 1)}
    description_result = {str(n) + '_description': [] for n in range(1, count_pred + 1)}
    
    for description in df[df.columns[0]].to_list(): #  здесь проходимся по описаниям
        text_preproc = ' '.join(preprocessing(description))
        answer = model.predict(text_preproc, k=count_pred)[0]
        cat = [i[9:].replace('_', '') for i in ans]
        description = classifier[classifier.TNVED.isin(cat)].FULL_TEXT.tolist()
        while len(description) < count_pred:
            description.append('unknown')
        
        for n_result in range(count_pred):
            result_pred[result_pred[n_result]].append(answer[n_result])
            description_result[description_result[n_result]].append(description[n_result])
    
    for cnt in range(1, count_pred + 1):
        df[str(cnt) + 'result'] = pd.Series(result_pred[result_pred.keys()[count_pred - 1]])
        df[str(cnt) + 'description'] = pd.Series(result_pred[result_pred.keys()[count_pred - 1]])
    
    return df