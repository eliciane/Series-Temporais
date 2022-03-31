##Passos Preliminares <BR>
**Importar bibliotecas e pacotes:** <BR>

import pandas as pd <BR>
import numpy as np <BR>
import matplotlib.pylab as plt <BR>
from matplotlib.pylab import rcParams <BR>
rcParams['figure.figsize'] = 15,6 <BR>
from statsmodels.tsa.stattools import adfuller <BR>
import statsmodels.api as sm <BR>
from statsmodels.tsa.statespace.tools import diff <BR>
import seaborn as sns <BR>
from datetime import datetime <BR>


**Funções usadas no programa** <BR>
*FUNÇÃO 1:* <br>
Teste Dickey-Fuller:<br>
O teste Dickey-Fuller Aumentado (ADF) tem o objetivo verificar a presença de raiz unitária em uma série de dados, considerando como hipótese nula (H0) a existencia de raiz unitária (não estacionaridade) e, como hipótese alternativa (H1), a presença de estacionaridade.O resultado deve ter o teste estatístico menor que o valor crítico , neste caso podemos rejeitar a hipótese nula e dizer que esta série temporal é de fato estacionária.
````commandline
def test_stationary(timeseries):
    # determinando as estatísticas para analisar se a série é estacionária
    rolmean = pd.Series(timeseries).rolling(window=12).mean()
    rolstd = pd.Series(timeseries).rolling(window=12).std()

    #plotar o gráfico das estatísticas
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')

    #desempenhar o teste Dickey-Fuller
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)
    # plt.figure(figsize=(20, 10))
    plt.show()
````
**Carregar o dataset**
```
#carregar o conjunto de dados e investigando os dados
data = pd.read_csv('csv/AirPassengers.csv')
print(data.head())
print (data.info())
```
##Preparar os dados
**Limpar e tratar os dados**

````commandline
#verificar se tem dados nulos
print(print.isnull().sum())

#passar argumentos para transformar a data em tipo datatyme e em índice
data['Month']=pd.to_datetime(data['Month'])
pd.to_datetime(['01/01/2019', '02/01/2019', '03/01/2019'], format='%d/%m/%Y')
print(data.info())
print(data.head(50))


#transformando em indice:
data.set_index('Month', inplace=True)

#passando para frequencia mensal (MS) e com a data sempre no primeiro dia do mês
data_mes = data.groupby(pd.Grouper(freq="MS")).sum()
print(data_mes.info())
print(data_mes[:31])

#checar o tipo de dados do indice
print(data_mes.index)

#converter a coluna em um ojeto Series para evitar referindo-se a nomes de colunas cada vez que usar os TS
ts = data_mes['#Passengers']
print(ts.head(10))

#técnicas de indexação
print(ts['1949-01-01'])
print(ts['1949-01-01':'1949-05-01'])

#plotar o grafico para ver o padrão da demanda
plt.plot(ts)
plt.show()

#fazer um box plot para visualizar outliers, a tendencia central e dispersão
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(ts)
g.set_title('Passageiros')
plt.show()
````

Chamar a função para testar se a demanda é estacionária <br>
Analisar os resultados, conforme definido no incio deste programa <br>

````commandline
#chamando a função
test_stationary(ts)
#o modelo não é estacionário. Então, fazer a diferença entre os lags para período de 12 meses de sazonalizade
````
Aplicar a transformação para reduzir a tendência. E tornar o modelo estacionário.
A transformação penalizaos valores maiores mais do que o menores valores menores. 
Podemos tomar o logaritmo, a raiz quadrada, a raiz cúbica, etc 

```
#aplicar a transformação logaritma
ts_log = np.log(ts)
print(ts_log)

#visualizar o gráfico para analisar se tem outliers
plt.figure(figsize=(5.5, 5.5))
g = sns.boxplot(ts_log)
g.set_title('Passageiros')
plt.show()

#fazer a diferenção e testar a estacionaridade 
Augmented Dickey-Fuller test (ADF Test)
diffts = diff(ts, k_diff=0, k_seasonal_diff=True, seasonal_periods=12)
print(diffts)
ad_fuller_result = adfuller(diffts)
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')
```
Observe que os dados estão estacionários com a diferença.

Vamos analisar a correção entre os lags para parametrizar os modelos.
````commandline
#analisar a correlçao entre os lags
fig = sm.graphics.tsa.plot_acf(ts_log.values.squeeze(), lags=40) # You can change the lags value if you want to see more lags
fig = sm.graphics.tsa.plot_pacf(ts_log.values.squeeze(), lags=40)
plt.show()
````
* de acordo com ACF e PACF acima, é possível indentificar AR(2) para modelo não sazonal.<br>
* E o MA(0) para o modelo sazonal para modelo não sazonal.<br>
* Mediante ACF, confirmamos o ciclo sazonal de 12 meses.
*#https://www.youtube.com/watch?v=l7jpmJLDmxQ

*Separar os dados em treino e teste*
````commandline
# Separar o datase em treino e teste:
print(len(ts_log))
train_size = int(len(ts_log)* 2/3)
print(train_size)
train_set = ts_log[:train_size]
print(train_set)
test_set = ts_log[train_size:]
print(test_set)
````
## Testar o modelo

````commandline
#fazendo a predição
model = sm.tsa.statespace.SARIMAX(train_set, order=(2, 1, 0), seasonal_order=(1,1,0,12))
results_ARIMA = model.fit(disp=-1)
print(results_ARIMA.summary())
results_ARIMA.plot_diagnostics(figsize=(15, 12))
plt.show()
````
**Voltar os dados na escala original e visualizar o gráfico da predição**
````commandline
#obter a predição
predictions_ARIMA = results_ARIMA.forecast(len(test_set))
predictions_ARIMA = pd.Series(predictions_ARIMA, index=test_set.index)

#O último passo é tomar o expoente
predictions_ARIMA_exp = np.exp(predictions_ARIMA)
print(predictions_ARIMA_exp)

#gerar plot da predição
plt.plot(ts)
plt.plot(predictions_ARIMA_exp)
plt.legend(('Data', 'Predictions'), fontsize=16)
start_date = datetime(1949,1,1)
end_date = datetime(1960,1,1)
plt.title('Passageiros', fontsize=20)
plt.ylabel('Quantidade', fontsize=16)
for year in range(start_date.year,end_date.year):
    plt.axvline(pd.to_datetime(str(year)+'-01-01'), color='k', linestyle='--', alpha=0.2)
plt.show()
````
###Validar a previsão nos dados de teste
````commandline
#encontrar o MAPE
from sklearn.metrics import mean_absolute_percentage_error
y_true = ts[96:]
y_pred = predictions_ARIMA_exp
print('Mean Absolute Percent Error:',mean_absolute_percentage_error(y_true, y_pred))

# encontrar o RMSE
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(test_set, predictions_ARIMA_exp))
print('Root Mean Squared Error:', rmse)
````