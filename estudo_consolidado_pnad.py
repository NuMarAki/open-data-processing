import pandas as pd
import matplotlib.pyplot as plt

# Caminho do consolidado
arquivo = "C:/TCC/dados/pnad/dados_pnad_consolidados.csv"

# Nova lista refinada de CBOs de TI
codigos_ti = [1425, 2120, 2122, 2123, 2124, 2125, 3171, 3172, 3173]

# Carregar dados (apenas colunas relevantes)
df = pd.read_csv(
    arquivo,
    sep=';',
    usecols=['ano','Trimestre','UF','idade','sexo','nivel_instrucao','cbo_ocupacao','eh_ti']
)

print(f"Total de registros: {len(df):,}")

# Corrigir idades inválidas
df = df[(df['idade'] >= 15) & (df['idade'] <= 100)]

# Criar coluna 'eh_ti_manual' com base nos códigos refinados
df['eh_ti_manual'] = df['cbo_ocupacao'].dropna().astype(str).str[:4]
df['eh_ti_manual'] = df['eh_ti_manual'].where(lambda x: x.str.isnumeric())
df['eh_ti_manual'] = df['eh_ti_manual'].astype(float).astype('Int64')
df['eh_ti_manual'] = df['eh_ti_manual'].isin(codigos_ti)

# Conferir totais
print(f"Marcados como TI (coluna eh_ti): {df['eh_ti'].sum():,}")
print(f"Marcados como TI (pela lista manual): {df['eh_ti_manual'].sum():,}")

# Filtrar profissionais de TI para análise de idade
ti_df = df[df['eh_ti_manual'] == True].copy()

# Clustering por idade
from sklearn.cluster import KMeans
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

# Frequência de CBOs
cbo_base = df[df['eh_ti']==True]['cbo_ocupacao'].value_counts()
cbo_manual = df[df['eh_ti_manual']==True]['cbo_ocupacao'].value_counts()

print("\n### CBOs marcados como TI na base original (eh_ti)")
print(cbo_base.head(20))

print("\n### CBOs marcados como TI pela lista manual")
print(cbo_manual.head(20))

# Diferenças
cbo_base_set = set(cbo_base.index)
cbo_manual_set = set(cbo_manual.index)

print("\n### CBOs na base original mas não na lista manual:")
print(cbo_base_set - cbo_manual_set)

print("\n### CBOs na lista manual mas não na base original:")
print(cbo_manual_set - cbo_base_set)

# Idades suspeitas (<15 ou >100)
idade_suspeita = df[(df['idade'] < 15) | (df['idade'] > 100)]
print(f"\nRegistros com idade suspeita (<15 ou >100): {len(idade_suspeita):,}")

# Mostrar exemplos apenas se existir
if len(idade_suspeita) > 0:
    print(idade_suspeita[['idade','cbo_ocupacao','eh_ti','eh_ti_manual']].sample(min(len(idade_suspeita),10), random_state=42))
else:
    print("Nenhuma idade suspeita encontrada.")

# Histograma de distribuição de idade
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