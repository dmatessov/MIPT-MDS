
#
# В этом примере мы постараемся преобразовать наш пример из ноутбука Jupiter в DAG airflow
#

from airflow import DAG
from datetime import datetime
# from airflow.hooks.base_hook import BaseHook
# from airflow.operators.python_operator import PythonOperator
from datetime import timedelta


# здесь используемые функциональные блоки

def create_table():
    import sqlite3
    con = sqlite3.connect("/home/danil/Yandex.Disk/DOC/MIPT-MDS/Семестр 2/Data engineering/test.db") #разумеется, тут вам надо указать свой путь к файлу!
    sqlite_drop_table = "DROP TABLE IF EXISTS books;"
    sqlite_create_table = """CREATE TABLE books (
        publishing_year     INTEGER,
        book_name           TEXT,
        author              TEXT,
        language_code       TEXT,
        author_rating       TEXT,
        book_average_rating NUMERIC,
        book_ratings_count  INTEGER,
        genre               TEXT,
        publisher           TEXT,
        book_id             INTEGER UNIQUE
        );"""
    cur = con.cursor()
    cur.execute(sqlite_drop_table)
    cur.execute(sqlite_create_table)
    con.close()


def load_data():
    from metabasepy import Client, MetabaseTableParser
    import sqlite3
    cli = Client(
                username="demo@skillfactory.ru", # тот самый, который вам выдали
                password="k0SZ4kEUasr0Fb", # и пароль к нему!
                base_url="http://sql.skillfactory.ru:3000"
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

    con = sqlite3.connect("/home/danil/Yandex.Disk/DOC/MIPT-MDS/Семестр 2/Data engineering/test.db")
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
        python_callable = create_table
    )

    load_data = PythonOperator(
        task_id='load_data',
        python_callable=load_data
    )

    create_table >> load_data