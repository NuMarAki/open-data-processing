import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Carregar amostra PNAD
df = pd.read_csv('C:/TCC/dados/pnad/preprocessados/pnad_amostra.csv', sep=';')

# Selecionar features
features = ['idade', 'nivel_instrucao', 'sexo', 'uf']
df = df[features].dropna()
df = pd.get_dummies(df, columns=['sexo', 'uf', 'nivel_instrucao'])

# Padronizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# PCA para 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X_pca)

# Visualizar
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis')
plt.title('Segmentação de Perfis Profissionais')
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.show()