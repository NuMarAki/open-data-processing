import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FuncFormatter, MaxNLocator

# --- MAPEAMENTOS GLOBAIS (CORRIGIDOS E COMPLETOS) ---

# Mapeamento para anos até 2015
# CORREÇÃO: Adicionados códigos 9, 11 e 12 com base na documentação e investigação.
MAPEAMENTO_ESCOLARIDADE_PRE_2015 = {
    1: 'Creche / Pré-Escola',
    2: 'Alfabetização de Jovens/Adultos',
    3: 'Antigo primário (elementar)',
    4: 'Antigo ginásio (médio 1º ciclo)',
    5: 'Ensino Fundamental (1º Grau)',
    6: 'EJA - Ensino Fundamental',
    7: 'Antigo científico, clássico, etc. (médio 2º ciclo)',
    8: 'Regular do ensino médio ou do 2º grau',
    9: 'Educação de jovens e adultos (EJA) ou supletivo do ensino médio',
    10: 'Superior - Graduação',
    11: 'Mestrado',
    12: 'Doutorado'
}

# Mapeamento para anos após 2015
# CORREÇÃO: Mapeados os códigos 7, 12, 13, 14, 15 conforme a PNAD Contínua e nossa investigação.
MAPEAMENTO_ESCOLARIDADE_POS_2015 = {
    1: 'Creche',
    2: 'Pré-Escola',
    3: 'Classe de Alfabetização',
    4: 'Alfabetização de Jovens/Adultos',
    5: 'Antigo Primário',
    6: 'Antigo Ginásio',
    7: 'Ensino Fundamental',
    8: 'EJA - Ensino Fundamental',
    9: 'Antigo Científico/Clássico',
    10: 'Ensino Médio',
    11: 'EJA - Ensino Médio',
    12: 'Superior - Graduação',
    13: 'Especialização Superior',
    14: 'Mestrado',
    15: 'Doutorado'
}

MAPEAMENTO_SEXO = {1: 'Homens', 2: 'Mulheres'}

def configurar_ambiente():
    """Cria o diretório para salvar os gráficos, se não existir."""
    if not os.path.exists('graficos'):
        os.makedirs('graficos')
    print("Diretório 'graficos' pronto.")

def carregar_dados():
    """Carrega os dados da PNAD a partir do caminho especificado ou de uma amostra."""
    try:
        # Usa o caminho absoluto fornecido pelo usuário
        df = pd.read_csv(r'C:\TCC\dados\pnad\dados_pnad_consolidados.csv', sep=';')
        print("Dados completos carregados com sucesso.")
    except FileNotFoundError:
        print("Arquivo principal em 'C:\\TCC\\dados\\pnad' não encontrado. Carregando 'amostra_pnad.csv' local.")
        try:
            df = pd.read_csv('amostra_pnad.csv', sep=';')
            print("Dados da amostra carregados com sucesso.")
        except FileNotFoundError:
            print("Erro: Nenhum arquivo de dados encontrado. Verifique os caminhos.")
            return None

    # Garante que a coluna 'eh_ti' seja booleana
    if df['eh_ti'].dtype == 'object':
        df['eh_ti'] = df['eh_ti'].str.upper().map({'TRUE': True, 'FALSE': False})

    return df

def gerar_grafico_escolaridade(df, ano, segmento):
    """
    Gera um gráfico de barras agrupado por sexo para cada nível de escolaridade,
    para um ano e segmento (TI ou Não-TI) específicos.
    """
    # 1. Definir títulos e nomes de arquivo com base no segmento
    if segmento == 'TI':
        df_segmento = df[df['eh_ti'] == True].copy()
        titulo_segmento = "Profissionais de TI"
        nome_arquivo_segmento = "TI"
    else:
        df_segmento = df[df['eh_ti'] == False].copy()
        titulo_segmento = "Demais Profissionais (Não-TI)"
        nome_arquivo_segmento = "Nao_TI"
        
    print(f"\n--- Processando: Ano {ano} | Segmento: {titulo_segmento} ---")

    # 2. Filtrar dados pelo ano e escolher a coluna e mapeamento corretos
    df_ano = df_segmento[df_segmento['ano'] == ano].copy()
    
    if ano <= 2015:
        coluna_escolaridade = 'curso_mais_elevado_antes_2015'
        mapeamento = MAPEAMENTO_ESCOLARIDADE_PRE_2015
    else:
        coluna_escolaridade = 'curso_mais_elevado'
        mapeamento = MAPEAMENTO_ESCOLARIDADE_POS_2015
        
    #    Isso lida tanto com números inteiros quanto com decimais (ex: 150.00155468)
    df_ano['peso_populacional'] = pd.to_numeric(df_ano['peso_populacional'], errors='coerce')
    # 3. Preparar os dados para o gráfico
    df_ano.dropna(subset=[coluna_escolaridade, 'sexo', 'peso_populacional'], inplace=True)
    
    # 3. Converte pa (remove) a parte decimal.ra inteiro. Esta ação TRUNCA
    #    Ex: 150.00155468 se torna 150.
    df_ano['peso_populacional'] = df_ano['peso_populacional'].astype(int)

    df_ano['nivel_escolaridade'] = df_ano[coluna_escolaridade].map(mapeamento)
    df_ano['sexo_desc'] = df_ano['sexo'].map(MAPEAMENTO_SEXO)
    df_ano.dropna(subset=['nivel_escolaridade', 'sexo_desc'], inplace=True)
    
    if df_ano.empty:
        print(f"Não há dados suficientes para o segmento '{titulo_segmento}' no ano {ano}.")
        return
        
    # 4. Agrupar dados e calcular população estimada
    dados_agrupados = df_ano.groupby(['nivel_escolaridade', 'sexo_desc'])['peso_populacional'].sum().reset_index()
    dados_agrupados.rename(columns={'peso_populacional': 'populacao_estimada'}, inplace=True)

    # Garantir que todas as categorias de escolaridade e sexo estejam presentes (mesmo com valor 0)
    all_niveis = [label for key, label in sorted(mapeamento.items())]
    all_sexos = ['Homens', 'Mulheres']
    idx = pd.MultiIndex.from_product([all_niveis, all_sexos], names=['nivel_escolaridade', 'sexo_desc'])
    dados_agrupados = dados_agrupados.set_index(['nivel_escolaridade', 'sexo_desc']).reindex(idx, fill_value=0).reset_index()

    # Ordenar o eixo X de forma lógica
    dados_agrupados['nivel_escolaridade'] = pd.Categorical(dados_agrupados['nivel_escolaridade'], categories=all_niveis, ordered=True)
    dados_agrupados.sort_values('nivel_escolaridade', inplace=True)

    # 5. Gerar o Gráfico
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 10))

    sns.barplot(
        data=dados_agrupados,
        x='nivel_escolaridade',
        y='populacao_estimada',
        hue='sexo_desc',
        hue_order=['Homens', 'Mulheres'],
        palette={'Homens': '#005A32', 'Mulheres': '#A4D3B3'},
        ax=ax,
    )

    # Forçar exibição de todos os rótulos no eixo X
    ax.set_xticks(range(len(all_niveis)))
    ax.set_xticklabels(all_niveis, rotation=45, ha='right', fontsize=12)

    # Remover números sobre as barras para evitar sobreposição/legibilidade ruim.
    # Em vez disso, aumentar a quantidade de linhas de grade no eixo Y para facilitar leitura.
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    try:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True, prune='lower'))
    except Exception:
        pass
    
    # ===== Salvar tabela com os dados usados no gráfico (CSV, sep=';') =====
    try:
        tabela = dados_agrupados.pivot(index='nivel_escolaridade', columns='sexo_desc', values='populacao_estimada').fillna(0)
        tabela = tabela.reset_index()
        base_name = f"escolaridade_sexo_{nome_arquivo_segmento}_{ano}"
        out_dir = 'graficos'
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"{base_name}.csv")
        # salvar com separador ';'
        tabela.to_csv(csv_path, index=False, sep=';', encoding='utf-8')
        print(f"Tabela de dados salva em: {csv_path}")
    except Exception as e:
        print("Aviso: não foi possível salvar tabela CSV:", e)
    # =====================================================================

    # Ajustes finais de layout e formatação
    ax.set_title(f'Distribuição por Nível de Escolaridade e Sexo\n({titulo_segmento} - {ano})', fontsize=18, pad=20)
    ax.set_xlabel('Nível de Escolaridade', fontsize=14)
    ax.set_ylabel('População Estimada', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x):,}'.replace(',', '.')))
    ax.legend(title='Sexo', fontsize=12)
    plt.tight_layout()

    # Salvar o gráfico
    caminho_salvar = f'graficos/escolaridade_sexo_{nome_arquivo_segmento}_{ano}.png'
    # garantir sobrescrita: remover arquivo anterior se existir
    if os.path.exists(caminho_salvar):
        try:
            os.remove(caminho_salvar)
        except Exception:
            pass
    plt.savefig(caminho_salvar, dpi=300)
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_salvar}") 


def main():
    """Função principal para orquestrar a análise."""
    configurar_ambiente()
    df = carregar_dados()

    if df is not None:
        anos_analise = [2012, 2024]
        segmentos = ['TI', 'Não-TI']
        
        for ano in anos_analise:
            # Verifica se o ano está presente nos dados antes de processar
            if ano in df['ano'].unique():
                for segmento in segmentos:
                    gerar_grafico_escolaridade(df, ano, segmento)
            else:
                print(f"\nAVISO: O ano {ano} não está disponível no dataset e será ignorado.")

    print("\nProcessamento concluído.")

if __name__ == "__main__":
    main()