import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Carregar amostra PNAD
df = pd.read_csv('C:/TCC/dados/pnad/dados_pnad_consolidados.csv', sep=';')

# Filtrar colunas relevantes e remover valores estranhos
df = df[['idade', 'eh_ti', 'sexo', 'nivel_instrucao']].dropna()
df = df[(df['idade'] >= 15) & (df['idade'] <= 100)]  # manter apenas idades plausíveis

# Clustering por idade para profissionais TI
ti_df = df[df['eh_ti'] == True].copy()
kmeans = KMeans(n_clusters=4, random_state=42)
ti_df['grupo_idade'] = kmeans.fit_predict(ti_df[['idade']])

# Resumo dos clusters
print("\nResumo dos grupos de idade (clusters):\n")
for i in range(4):
    subset = ti_df[ti_df['grupo_idade'] == i]['idade']
    print(f"Grupo {i}:")
    print(f"  Média: {subset.mean():.1f} anos")
    print(f"  Mínimo: {subset.min()} anos")
    print(f"  Máximo: {subset.max()} anos")
    print(f"  Quantidade: {len(subset)} pessoas\n")

# Visualizar distribuição dos grupos
plt.hist(
    [ti_df[ti_df['grupo_idade'] == i]['idade'] for i in range(4)],
    bins=20,
    label=[f'Grupo {i}' for i in range(4)],
    stacked=True
)
plt.legend()
plt.title('Distribuição de Idade - Profissionais TI')
plt.xlabel('Idade')
plt.ylabel('Quantidade')
plt.show()
