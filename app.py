import random, math, operator, io, numpy as np, pandas as pd

# Importando o Pyplot para fazer os histogramas e boxplots
from matplotlib import pyplot as plt
# Importando uma implementacao do KNN
from sklearn.neighbors import KNeighborsRegressor
# Importando o imputer para fazer a manipulacao do dataframe
from sklearn.impute import SimpleImputer 
from numpy.random import permutation
from flask import Flask, request, render_template, url_for, redirect, g
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app_database.db'
db = SQLAlchemy(app)

tabela = pd.read_csv('data.csv')

@app.before_first_request
def create_database():
    db.create_all()
    print('[Flask-Server] >>> Database created.')


@app.route('/', methods=['POST', 'GET'])
def index():

    global tabela

    if request.method == 'POST':
        try:
            normalized = True
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

        return render_template('index.html', 
                table_html=df_html, 
                max_values=max_html, 
                min_values=min_html, 
                type_values=types_html, 
                normalized=normalized)
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

@app.route('/table_data', methods=['POST', 'GET'])
def tableData():
    return render_template('table_data.html')


@app.route('/knn', methods=['POST', 'GET'])
def runKNN():
    # Marginal_Adhesion, Single_Epi_Cell_Size, Bare_Nuclei, Bland_Chromatin, Normal_Nucleoli, Mitoses
    dados = tabela[['Marginal_Adhesion', 'Single_Epi_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']]

    # Separando dados dos registros que precisam ser descobertos
    evaluation_data = dados.head(139)

    # Separando dados dos registros que ja possuem classe definida
    training_data = dados.tail(560)

    # Aqui iremos mostrar a precisao do algoritmo,
    # para isso separamos um trainingSet e um testSet
    # dentro do dataframe que ja possui todas as classes definidas,
    # dessa forma podemos saber se o algoritmo acertou em sua previsao
    trainingSet = training_data.head(374).values
    testSet = training_data.tail(186).values

    # Mostrando o tamanho de cada base
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))

    # salvando uma lista das previsoes com base nos vizinhos
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> classe prevista=' + repr(result) + ', classe real=' + repr(testSet[x][-1]))
        accuracy = getAccuracy(testSet, predictions)
        print("Chegou aqui")
        print('Precisão: ' + repr(accuracy) + '%')

    return "KNN"


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


# Funcao que recupera a distancia Euclidiana
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)


#Funcao que procura os vizinhos mais proximos de uma instancia
def getNeighbors(trainingSet, testInstance, k):
    # Iniciando lista de distancias
	distances = []
    # Pegando tamanho da instancia de teste
	length = len(testInstance)-1
    # Varrendo o trainingSet
	for x in range(len(trainingSet)):
        # Calculando a distancia da instancia em relacao aos outros pares
		dist = euclideanDistance(testInstance, trainingSet[x], length)
        # Adicionando o valor calculado a lista de distancias
		distances.append((trainingSet[x], dist))
    # Sorteando a lista de distancias    
	distances.sort(key=operator.itemgetter(1))
    # Criando a lista de vizinhos
	neighbors = []
    # Varrendo a lista de vizinhos
	for x in range(k):
        # Adicionando o numero de vizinhos de acordo a ordem da lista
		neighbors.append(distances[x][0])
    # Retorna os vizinhos mais proximos    
	return neighbors


# Funcao que sumariza o resultado final entre os vizinhos selecionados
def getResponse(neighbors):
    # Criando lista de votos
	classVotes = {}
    # Varrendo lista de vizinhos
	for x in range(len(neighbors)):
        # Pegando a classe do vizinho
		response = neighbors[x][-1]
		if response in classVotes:
            # O numero de votos incrementa conforme a classe aparece
			classVotes[response] += 1
		else:
			classVotes[response] = 1
    # Ordenando a lista pela classe com mais votos        
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    # Retorna a classe mais votada
	return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] is predictions[x]:
			correct += 1

	return (correct/float(len(testSet))) * 100.0


if __name__ == "__main__":
    app.run(debug=True)