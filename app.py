import random, math, operator, io, numpy as np, pandas as pd

# Importando o Pyplot para fazer os histogramas e boxplots
from matplotlib import pyplot as plt
# Importando uma implementacao do KNN
from sklearn.neighbors import KNeighborsRegressor
# Importando o imputer para fazer a manipulacao do dataframe
from sklearn.impute import SimpleImputer 
from numpy.random import permutation
from flask import Flask, request, render_template, url_for, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app_database.db'
db = SQLAlchemy(app)


@app.before_first_request
def create_database():
    db.create_all()
    print('[Flask-Server] >>> Database created.')


@app.route('/', methods=['POST', 'GET'])
def index():

    tabela = pd.read_csv('data.csv')

    if request.method == 'POST':
        try:
            tabela = normalize(tabela)
            df_html = tabela.to_html()
            max_values = tabela.max().to_frame(name='max_values')
            min_values = tabela.min().to_frame(name='min_values')
            type_values = tabela.dtypes.to_frame(name='type_values')

            max_html = max_values.to_html()
            min_html = min_values.to_html()
            types_html = type_values.to_html()
        except:
            return render_template('index.html')    

        return render_template('index.html', table_html=df_html, max_values=max_html, min_values=min_html, type_values=types_html)
    else:    
        try:
            df_html = tabela.to_html()
            max_values = tabela.max().to_frame(name='max_values')
            min_values = tabela.min().to_frame(name='min_values')
            type_values = tabela.dtypes.to_frame(name='type_values')

            max_html = max_values.to_html()
            min_html = min_values.to_html()
            types_html = type_values.to_html()
        except:
            return render_template('index.html')    

        return render_template('index.html', table_html=df_html, max_values=max_html, min_values=min_html, type_values=types_html)    


def normalize(tabela):
    # Para começar a normalizacao, trocamos todas as strings '?' por NaN (Not a Number)
    tabela['Bare_Nuclei'] = tabela['Bare_Nuclei'].replace('?', np.nan)

    # Criamos um imputer que tira a media da coluna Bare_Nuclei
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Aqui o imputer pega os valores da coluna para tirar a media
    imputer = imputer.fit(tabela.iloc[:, 5:6])

    # Aqui os valores sao inseridos de volta no dataframe
    tabela.iloc[:,5:6] = imputer.transform(tabela.iloc[:,5:6])

    # Para padronizar as colunas, mudamos seu tipo para int
    tabela['Bare_Nuclei'] = tabela.Bare_Nuclei.astype(int)

    return tabela


if __name__ == "__main__":
    app.run(debug=True)