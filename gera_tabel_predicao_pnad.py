import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table

# Mapeamentos de-para (baseados em gera_*.py e PNAD)
IBGE_UF = {
    11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO',
    21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA',
    31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP',
    41: 'PR', 42: 'SC', 43: 'RS',
    50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'
}

sexo_map = {1: 'Homem', 2: 'Mulher'}

nivel_instrucao_map = {
    0: 'Sem instrução',
    1: 'Fundamental incompleto',
    2: 'Fundamental completo',
    3: 'Médio incompleto',
    4: 'Médio completo',
    5: 'Superior incompleto',
    6: 'Superior completo',
    7: 'Pós-graduação'
}

# Dados originais
idades = [41, 42, 40, 39, 43, 41, 40, 38, 44, 41, 25, 30, 35, 45, 50, 55, 60, 28, 32, 37]
sexos = [2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
ufs = [53, 53, 53, 53, 53, 35, 33, 43, 31, 53, 41, 35, 33, 43, 31, 41, 35, 33, 43, 31]
niveis = [7, 7, 7, 7, 7, 5, 6, 4, 7, 7, 3, 4, 5, 6, 7, 7, 2, 1, 3, 4]
probs = ['68,27', '65,70', '68,20', '29,96', '69,77', '29,84', '35,33', '19,33', '60,17', '68,27', '4,64', '13,50', '14,38', '25,32', '66,78', '31,11', '24,42', '5,71', '10,47', '23,35']
predicoes_numericas = [1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
interpretacoes = ['Salário >= 6 SM', 'Salário >= 6 SM', 'Salário >= 6 SM', 'Salário < 6 SM', 'Salário >= 6 SM', 'Salário < 6 SM', 'Salário < 6 SM', 'Salário < 6 SM', 'Salário >= 6 SM', 'Salário >= 6 SM', 'Salário < 6 SM', 'Salário < 6 SM', 'Salário < 6 SM', 'Salário < 6 SM', 'Salário >= 6 SM', 'Salário < 6 SM', 'Salário < 6 SM', 'Salário < 6 SM', 'Salário < 6 SM', 'Salário < 6 SM']

# Aplicar mapeamentos
sexos_mapped = [sexo_map[s] for s in sexos]
ufs_mapped = [IBGE_UF[u] for u in ufs]
niveis_mapped = [nivel_instrucao_map[n] for n in niveis]

# Dados da tabela com valores traduzidos
data = {
    'Idade': idades,
    'Sexo': sexos_mapped,
    'UF': ufs_mapped,
    'Nível de Instrução': niveis_mapped,
    'Probabilidade (%)': probs,
    'Predição': ['Verdadeiro' if p == 1 else 'Falso' for p in predicoes_numericas],  # Verdadeiro/Falso
    'Interpretação': interpretacoes
}

df = pd.DataFrame(data)

# Criar figura
fig, ax = plt.subplots(figsize=(14, 12))  # Ajustar tamanho para textos maiores
ax.axis('off')

# Criar tabela
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# Estilizar com tons de verde menos vibrantes e fonte serif
table.auto_set_font_size(False)
table.set_fontsize(8)  # Fonte ainda menor para caber textos longos
table.scale(1.0, 1.0)

# Definir fonte serif
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Cores em tons de verde menos vibrantes
for (i, j), cell in table.get_celld().items():
    if i == 0:
        cell.set_facecolor('#556B2F')  # Verde oliva escuro para cabeçalho
        cell.set_text_props(weight='bold', color='white')
    else:
        if df.iloc[i-1]['Predição'] == 'Verdadeiro':
            cell.set_facecolor('#8FBC8F')  # Verde acinzentado para positivas
        else:
            cell.set_facecolor('#F0F8FF')  # Azul muito claro para negativas
        cell.set_text_props(color='black')

# Salvar imagem
plt.savefig('tabela_predicoes_excel_verde_traduzida.png', dpi=300, bbox_inches='tight')
plt.show()

print("Imagem da tabela salva como 'tabela_predicoes_excel_verde_traduzida.png'.")

# Dados para Tabela 1
data_tabela1 = {
    'Item': ['Alvo', 'Escopo', 'Tamanho da amostra', 'Distribuição de classes (treino)', 'Distribuição de classes (teste)'],
    'Valor': ['salario_6sm', 'TI', '98.738', '0: 58.618    1: 15.435', '0: 19.540    1: 5.145']
}

df1 = pd.DataFrame(data_tabela1)

# Dados para Tabela 2
data_tabela2 = {
    'Métrica': ['Acurácia (Acc)', 'Acurácia Balanceada (BalAcc)', 'Precisão (Prec)', 'Recall (Rec)', 'F1-score (F1)', 'ROC AUC', 'PR AUC', 'Limiar ótimo (F1)'],
    'Valor não ponderado': ['0,8287', '0,6726', '0,6408', '0,4051', '0,4964', '0,8590', '0,5909', '0,2500 → F1=0,5846'],
    'Valor ponderado (peso_populacional)': ['0,8093', '–', '0,6762', '0,4145', '0,5139', '–', '–', '0,3000 → [W]F1=0,6225']
}

df2 = pd.DataFrame(data_tabela2)

# Criar figura com subplots para duas tabelas
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Tabelas para TCC', fontsize=14, fontweight='bold')

# Tabela 1
ax1.axis('off')
table1 = ax1.table(cellText=df1.values, colLabels=df1.columns, cellLoc='center', loc='center')
table1.auto_set_font_size(False)
table1.set_fontsize(10)
table1.scale(1.2, 1.2)
ax1.set_title('Tabela 1 – Características da amostra', fontsize=12, pad=20)

# Estilizar Tabela 1
for (i, j), cell in table1.get_celld().items():
    if i == 0:
        cell.set_facecolor('#556B2F')
        cell.set_text_props(weight='bold', color='white')
    else:
        cell.set_facecolor('#F0F8FF')
        cell.set_text_props(color='black')

# Tabela 2
ax2.axis('off')
table2 = ax2.table(cellText=df2.values, colLabels=df2.columns, cellLoc='center', loc='center')
table2.auto_set_font_size(False)
table2.set_fontsize(9)
table2.scale(1.2, 1.2)
ax2.set_title('Tabela 2 – Resultados do modelo (Random Forest)', fontsize=12, pad=20)

# Estilizar Tabela 2
for (i, j), cell in table2.get_celld().items():
    if i == 0:
        cell.set_facecolor('#556B2F')
        cell.set_text_props(weight='bold', color='white')
    else:
        cell.set_facecolor('#F0F8FF')
        cell.set_text_props(color='black')

# Ajustar layout
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Salvar imagem
plt.savefig('tabelas_tcc.png', dpi=300, bbox_inches='tight')
plt.show()

print("Imagem das tabelas salva como 'tabelas_tcc.png'.")