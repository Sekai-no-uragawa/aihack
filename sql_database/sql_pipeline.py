import pandas as pd
import psycopg2


with open('query_data_for_model.sql') as filename:
    query = filename.read()

date = '"01.01.2015"'
query = query.replace('&variable', date)


def df_from_postgre(connection, query, columns):
    
    cursor = connection.cursor()
    try:
        cursor.execute(query)
    except:
        print('Перезапустите ячейку с connection, если не поможет то ошибка в запросе или сбой подключения')
        cursor.close()
        return 1
            
    tupple = cursor.fetchall()
    cursor.close()
    df = pd.DataFrame(tupple, columns = columns)
    
    return df

connection = psycopg2.connect(
                    host='yandex.ru',
                    port=6432,
                    dbname='hr',
                    sslmode='require',
                    user='HR',
                    password='HR'
)