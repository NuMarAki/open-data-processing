import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matplotlib.ticker import FuncFormatter, MaxNLocator

# Paleta e configurações visuais do projeto
PALETTE = {
    'primary': '#005A32',    # verde escuro (TI)
    'primary_dark': '#004427',
    'secondary': '#2E7D52',  # verde médio (alternativa)
    'accent': '#37a546',     # verde vivo (destaque)
    'grid': '#EAEAEA'
}
plt.rcParams.update({
    'figure.dpi': 100,
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 13,
    'axes.titlesize': 16,
    'legend.fontsize': 11
})

def carregar_dados():
    """Carrega os dados da PNAD e prepara as colunas necessárias."""
    try:
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

    # Garante que colunas importantes sejam do tipo correto
    if df['eh_ti'].dtype == 'object':
        df['eh_ti'] = df['eh_ti'].str.upper().map({'TRUE': True, 'FALSE': False})
    df['rendimento_trabalho_principal'] = pd.to_numeric(df['rendimento_trabalho_principal'], errors='coerce')
    df['ocupado'] = pd.to_numeric(df['ocupado'], errors='coerce')
    df['idade'] = pd.to_numeric(df['idade'], errors='coerce')
    
    return df

def gerar_grafico_representatividade(df):
    """Gera o gráfico de série temporal da representatividade de TI."""
    print("\n--- Processando dados para [Gráfico 1: Representatividade de TI] ---")
    df_ocupados = df[(df['ocupado'] == 1) | (df['rendimento_trabalho_principal'] > 0)].copy()

    if df_ocupados.empty:
        print("Não há dados de profissionais ocupados para o Gráfico 1.")
        return

    # AJUSTE: Adicionado 'include_groups=False' para remover o FutureWarning
    dados_anuais = df_ocupados.groupby('ano').apply(lambda x: pd.Series({
        'total_ti': (x['eh_ti'] * x['peso_populacional']).sum(),
        'total_trabalhadores': x['peso_populacional'].sum()
    }), include_groups=False).reset_index()

    dados_anuais['percentual_ti'] = (dados_anuis := dados_anuais)['total_ti'] / dados_anuais['total_trabalhadores'].replace(0, np.nan) * 100
    print("Dados anuais de representatividade processados.")

    # Definir estilo: preferimos usar seaborn.set_style quando disponível.
    try:
        sns.set_style('whitegrid')
    except Exception:
        # fallback para nomes de estilo do matplotlib compatíveis com várias versões
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            plt.style.use('default')

    fig, ax = plt.subplots(figsize=(16, 9))
    sns.lineplot(
        data=dados_anuais, x='ano', y='percentual_ti', ax=ax,
        marker='o', markersize=7, linewidth=2.2, color=PALETTE['primary'], markeredgecolor='white'
    )
    ax.set_facecolor('white')
    ax.grid(color=PALETTE['grid'], linewidth=0.8)

    # antes de anotar, adicionar folga vertical para evitar que textos saiam da área
    ax.margins(y=0.12)  # adiciona espaço vertical acima dos pontos
    for index, row in dados_anuais.iterrows():
        ax.annotate(
            f"{row['percentual_ti']:.2f}%",
            xy=(row['ano'], row['percentual_ti']),
            xytext=(0, 6),                # deslocamento em pontos (para cima)
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=11, color=PALETTE['primary_dark'], weight='semibold',
            clip_on=False                 # garante que o texto seja visível mesmo fora do bbox do eixo
        )

    ax.set_title('Representatividade de Profissionais de TI no Mercado de Trabalho (Série Histórica)', fontsize=18, pad=20)
    ax.set_xlabel('Ano', fontsize=14)
    ax.set_ylabel('Percentual de Profissionais de TI (%)', fontsize=14)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}%'))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    caminho_salvar = f'graficos/serie_temporal_representatividade_ti.png'
    plt.savefig(caminho_salvar, dpi=300)
    plt.close(fig)
    print(f"Gráfico 1 salvo em: {caminho_salvar}")

def gerar_grafico_media_idade_ti(df):
    """Gera o gráfico de série temporal da idade média dos profissionais de TI."""
    print("\n--- Processando dados para [Gráfico 2: Média de Idade de TI] ---")
    df_ti = df[df['eh_ti'] == True].copy()
    df_ti.dropna(subset=['idade', 'peso_populacional'], inplace=True)

    if df_ti.empty:
        print("Não há dados de profissionais de TI para o Gráfico 2.")
        return

    # AJUSTE: Adicionado 'include_groups=False' para remover o FutureWarning
    dados_idade = df_ti.groupby('ano').apply(
        lambda x: np.average(x['idade'], weights=x['peso_populacional']),
        include_groups=False
    ).reset_index(name='media_idade')
    print("Dados anuais de idade média processados.")

    # definir estilo com fallback: prefere seaborn.set_style quando disponível
    try:
        sns.set_style('whitegrid')
    except Exception:
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            plt.style.use('default')

    fig, ax = plt.subplots(figsize=(16, 9))
    sns.lineplot(
        data=dados_idade, x='ano', y='media_idade', ax=ax,
        marker='s', markersize=7, linewidth=2.2, color=PALETTE['secondary'], markeredgecolor='white'
    )
    ax.set_facecolor('white')
    ax.grid(color=PALETTE['grid'], linewidth=0.8)
    ax.margins(y=0.10)
    for index, row in dados_idade.iterrows():
        ax.annotate(
            f"{row['media_idade']:.1f} anos",
            xy=(row['ano'], row['media_idade']),
            xytext=(0, 6),
            textcoords='offset points',
            ha='center', va='bottom',
            fontsize=11, color=PALETTE['primary_dark'], weight='semibold',
            clip_on=False
        )

    ax.set_title('Média de Idade dos Profissionais de TI (Série Histórica)', fontsize=18, pad=20)
    ax.set_xlabel('Ano', fontsize=14)
    ax.set_ylabel('Idade Média', fontsize=14)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    caminho_salvar = f'graficos/serie_temporal_media_idade_ti.png'
    plt.savefig(caminho_salvar, dpi=300)
    plt.close(fig)
    print(f"Gráfico 2 salvo em: {caminho_salvar}")

def main():
    """Função principal para orquestrar a análise."""
    configurar_ambiente()
    df = carregar_dados()

    if df is not None:
        gerar_grafico_representatividade(df)
        gerar_grafico_media_idade_ti(df)

    print("\nProcessamento concluído.")

if __name__ == "__main__":
    main()