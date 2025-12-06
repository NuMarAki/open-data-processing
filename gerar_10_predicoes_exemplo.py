import pandas as pd
import joblib
import numpy as np

# Carregar o modelo treinado (ajuste o caminho se necessário)
modelo_path = r"modelos\salario_6sm\rf_ti_salario_6sm.joblib"  # Para alvo="salario_6sm"
# Ou para alvo="ocupado": r"modelos\ocupado\rf_ti_ocupado.joblib"

pipe = joblib.load(modelo_path)
print("Modelo carregado com sucesso.")

# Dados reais fornecidos (preenchendo missing com estimativas razoáveis)
# Nota: anos_estudo estimado com base em nivel_instrucao (ex.: 13 ~16-18 anos)
# horas_trabalhadas_semana ~40 (padrão), tipo_area 1 (urbana), trimestre 1 (exemplo)
dados_real = pd.DataFrame({
    'idade': [41],
    'anos_estudo': [17],  # Estimativa para nivel_instrucao 13 (superior)
    'horas_trabalhadas_semana': [40],  # Estimativa padrão
    'sexo': [2],  # Mulher
    'cor_raca': [1],  # Branca
    'uf': [53],  # DF
    'nivel_instrucao': [7],  # Ensino Fundamental (baseado no prompt, mas curso 13 pode ser superior; ajustado para 7 se for o caso)
    'carteira_assinada': [1],  # Sim
    'tipo_area': [1],  # Urbana
    'ano': [2024],
    'trimestre': [1]  # Exemplo
})

print("Dados reais (com estimativas para features ausentes):")
print(dados_real)

# Fazer predição
probabilidades = pipe.predict_proba(dados_real)[:, 1]  # Probabilidade da classe positiva (1)
predicoes = pipe.predict(dados_real)  # Predição binária (0 ou 1)

# Resultados
resultados = dados_real.copy()
resultados['probabilidade'] = probabilidades
resultados['probabilidade_%'] = (probabilidades * 100).round(2)  # Probabilidade em %
resultados['predicao'] = predicoes
resultados['interpretacao'] = resultados['predicao'].map({1: 'Salário >= 6 SM', 0: 'Salário < 6 SM'})

print("\nPredição para os dados reais:")
print(resultados[['idade', 'anos_estudo', 'nivel_instrucao', 'probabilidade', 'probabilidade_%', 'predicao', 'interpretacao']])

# Contexto: Salário mínimo 2024 = 1412, 6 SM = 8472. Renda bruta 12000 > 8472, então esperado positivo.
print("\nContexto: Com renda bruta de 12000 em 2024, o salário mínimo é 1412, então 6 SM = 8472. Como 12000 > 8472, esperado 'Salário >= 6 SM'.")

# Salvar em CSV
resultados.to_csv('predicao_dado_real.csv', index=False, sep=';')
print("\nPredição salva em 'predicao_dado_real.csv'.")

# Criar 20 registros, com pelo menos 2 similares ao dado real que funcionou
# Dados similares: idade ~41, sexo 2, cor_raca 1, uf 53, carteira_assinada 1, nivel_instrucao 7, ano 2024
# Variações em outros campos
# Ajuste: nivel_instrucao de 0 a 7

dados_exemplo = pd.DataFrame({
    'idade': [41, 42, 40, 39, 43, 41, 40, 38, 44, 41, 25, 30, 35, 45, 50, 55, 60, 28, 32, 37],
    'anos_estudo': [17, 16, 18, 15, 19, 17, 16, 14, 20, 17, 12, 14, 16, 18, 20, 22, 15, 10, 13, 17],
    'horas_trabalhadas_semana': [40, 42, 38, 41, 39, 40, 43, 37, 44, 40, 35, 38, 40, 42, 45, 48, 36, 32, 39, 41],
    'sexo': [2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],  # Pelo menos 2 com 2
    'cor_raca': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
    'uf': [53, 53, 53, 53, 53, 35, 33, 43, 31, 53, 41, 35, 33, 43, 31, 41, 35, 33, 43, 31],
    'nivel_instrucao': [7, 7, 7, 7, 7, 5, 6, 4, 7, 7, 3, 4, 5, 6, 7, 7, 2, 1, 3, 4],  # Ajustado para 0-7
    'carteira_assinada': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
    'tipo_area': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1],
    'ano': [2024] * 20,
    'trimestre': [1, 1, 1, 1, 1, 2, 3, 4, 1, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3]
})

print("Dados de exemplo (20 registros, com 5 similares ao dado real, nivel_instrucao 0-7):")
print(dados_exemplo.head(10))

# Fazer predições
probabilidades = pipe.predict_proba(dados_exemplo)[:, 1]
predicoes = pipe.predict(dados_exemplo)

# Resultados
resultados = dados_exemplo.copy()
resultados['probabilidade'] = probabilidades
resultados['probabilidade_%'] = (probabilidades * 100).round(2)
resultados['predicao'] = predicoes
resultados['interpretacao'] = resultados['predicao'].map({1: 'Salário >= 6 SM', 0: 'Salário < 6 SM'})

positivos = (predicoes == 1).sum()
print(f"\nResumo: De 20 predições, {positivos} são positivas.")

print("\nPredições:")
print(resultados[['idade', 'sexo', 'uf', 'nivel_instrucao', 'probabilidade_%', 'predicao', 'interpretacao']])

# Salvar em CSV
resultados.to_csv('20_predicoes_similares_ajustadas.csv', index=False, sep=';')
print("\n20 predições salvas em '20_predicoes_similares_ajustadas.csv'.")