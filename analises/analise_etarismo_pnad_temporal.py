import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prophet import Prophet
import matplotlib.pyplot as plt

# Carregar amostra PNAD
df = pd.read_csv('C:/TCC/dados/pnad/preprocessados/pnad_amostra.csv', sep=';')

# Agregar número de profissionais TI por ano
serie = df[df['eh_ti'] == True].groupby('ano').size().reset_index(name='n_ti')
serie = serie.rename(columns={'ano': 'ds', 'n_ti': 'y'})

# Modelo Prophet
model = Prophet(yearly_seasonality=True)
model.fit(serie)

# Previsão para próximos anos
future = model.make_future_dataframe(periods=3, freq='Y')
forecast = model.predict(future)

# Visualizar
model.plot(forecast)
plt.title('Evolução de Profissionais TI')
plt.xlabel('Ano')
plt.ylabel('Quantidade')
plt.show()