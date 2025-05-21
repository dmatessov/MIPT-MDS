import pandas as pd
import sqlite3

from dash import Dash, html, dash_table, dcc
import pandas as pd
import plotly.express as px


con = sqlite3.connect("./transaction_bd.db")
sql_query_f = """
SELECT
	DATE(TX_DATETIME) AS TX_DATE,
	SUM(TX_AMOUNT) AS SUM_TX_AMOUNT
FROM final_transactions
GROUP BY DATE(TX_DATETIME)
ORDER BY DATE(TX_DATETIME);"""
df = pd.read_sql(sql_query_f, con)


# Initialize the app
app = Dash(__name__)

# App layout
app.layout = html.Div([
    html.Div(children='(Балл - 0.2) Подготовить дашборд с помощью Dash по пункту 2.f, включив туда графики bar и histogram; вставить в конце ноутбука скрин графиков из дашборда.'),
    #dash_table.DataTable(data=df.to_dict('records'), page_size=10),
    dcc.Graph(figure=px.bar(df, x='TX_DATE', y='SUM_TX_AMOUNT')),
    dcc.Graph(figure=px.histogram(df, x='TX_DATE', y='SUM_TX_AMOUNT', histfunc='sum'))
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)