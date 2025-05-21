
#
# В этом примере мы постараемся преобразовать наш пример из ноутбука Jupiter в DAG airflow
#

from airflow import DAG
from datetime import datetime
from airflow.hooks.base_hook import BaseHook
from airflow.operators.python_operator import PythonOperator
from datetime import timedelta

# здесь используемые функциональные блоки

def create_table(sqlLiteConn):
    import sqlite3
    sqlLite = BaseHook.get_connection(sqlLiteConn)
    sqlLiteStr = sqlLite.host
    con = sqlite3.connect(sqlLiteStr) 
    drop_table_sql = 'DROP TABLE IF EXISTS books;'
    create_table_sql = '''
    CREATE TABLE books (
    publishing_year int,
    book_name text,
    author text,
    language_code text,
    author_rating float,
    book_average_rating float,
    book_ratings_count int,
    genre text,
    publisher text,
    book_id int  
    )
    '''
    cur = con.cursor()
    cur.execute(drop_table_sql)
    cur.execute(create_table_sql)
    con.close()


def load_data(metabaseConn, sqlLiteConn):
    from metabasepy import Client, MetabaseTableParser
    import sqlite3
    sqlLite = BaseHook.get_connection(sqlLiteConn)
    metabase = BaseHook.get_connection(metabaseConn)
    sqlLiteStr = sqlLite.host
    metabaseHost = metabase.host #"http://sql.skillfactory.ru:3000"
    metabaseLogin = metabase.login #"demo10@skillfactory.ru"
    metabasePassword = metabase.password#"t9vQJlErQ9WcMi"
    con = sqlite3.connect(sqlLiteStr)

    cli = Client(
                username=metabaseLogin, # тот самый, который вам выдали
                password=metabasePassword, # и пароль к нему!
                base_url=metabaseHost
      )

    cli.authenticate()
    books_sql = '''
        SELECT publishing_year,
       book_name,
       author,
       language_code,
       author_rating,
       book_average_rating,
       book_ratings_count,
       genre,
       publisher,
       book_id
        FROM books
        '''
    query_response = cli.dataset.post(database_id=2, query=books_sql)
    data_table = MetabaseTableParser.get_table(metabase_response=query_response)
    cur = con.cursor()
    cur.executemany("INSERT INTO books VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data_table.rows)
    con.commit()
    con.close()


# собственно, структура и последовательность DAG


with DAG(
    dag_id='learning_dag',
    schedule_interval='0 0 * * *',
    start_date=datetime(2023, 3, 3),
    catchup=False,
    dagrun_timeout=timedelta(minutes=60),
    tags=['example', 'example2'],
) as dag:

    create_table = PythonOperator (
        task_id = 'create_table',
        python_callable = create_table,
        op_kwargs = {
            'sqlLiteConn': 'sqlLiteConn'
        }
    )

    load_data = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
        op_kwargs = {
            'sqlLiteConn': 'sqlLiteConn',
            'metabaseConn': 'metabaseConn'
        }
    )

    create_table >> load_data