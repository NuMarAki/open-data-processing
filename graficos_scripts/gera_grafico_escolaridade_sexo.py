import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matplotlib.ticker import FuncFormatter, MaxNLocator

# --- MAPEAMENTOS GLOBAIS ---
MAPEAMENTO_ESCOLARIDADE_PRE_2015 = {
    1: 'Creche / Pre-Escola',
    2: 'Alfabetizacao de Jovens/Adultos',
    3: 'Antigo primario (elementar)',
    4: 'Antigo ginasio (medio 1 ciclo)',
    5: 'Ensino Fundamental (1 Grau)',
    6: 'EJA - Ensino Fundamental',
    7: 'Antigo cientifico, classico, etc. (medio 2 ciclo)',
    8: 'Regular do ensino medio ou do 2 grau',
    9: 'Educacao de jovens e adultos (EJA) ou supletivo do ensino medio',
    10: 'Superior - Graduacao',
    11: 'Mestrado',
    12: 'Doutorado'
}

MAPEAMENTO_ESCOLARIDADE_POS_2015 = {
    1: 'Creche',
    2: 'Pre-Escola',
    3: 'Classe de Alfabetizacao',
    4: 'Alfabetizacao de Jovens/Adultos',
    5: 'Antigo Primario',
    6: 'Antigo Ginasio',
    7: 'Ensino Fundamental',
    8: 'EJA - Ensino Fundamental',
    9: 'Antigo Cientifico/Classico',
    10: 'Ensino Medio',
    11: 'EJA - Ensino Medio',
    12: 'Superior - Graduacao',
    13: 'Especializacao Superior',
    14: 'Mestrado',
    15: 'Doutorado'
}

MAPEAMENTO_SEXO = {1: 'Homens', 2: 'Mulheres'}

def configurar_ambiente():
    """Cria a pasta de graficos se nao existir."""
    os.makedirs('graficos', exist_ok=True)
    print("[OK] Ambiente configurado (pasta 'graficos' verificada)")

def carregar_dados_preprocessados_2anos(anos=[2012, 2024]):
    """Carrega apenas dados dos anos especificados para otimizar memoria."""
    import glob
    
    # Encontrar caminho dos preprocessados
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'dados', 'pnad', 'preprocessados'),
        'dados/pnad/preprocessados',
        'z:/TCC/Entrega/open-data-processing/dados/pnad/preprocessados',
    ]
    
    preprocessados_dir = None
    for path in candidate_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            preprocessados_dir = abs_path
            break
    
    if not preprocessados_dir:
        print("Erro: Pasta preprocessados nao encontrada.")
        return None
    
    print(f"Carregando dados de {preprocessados_dir}")
    
    # Procurar arquivos correspondentes aos anos desejados
    csv_files = glob.glob(os.path.join(preprocessados_dir, '*preprocessado.csv'))
    
    # Filtrar apenas arquivos dos anos especificados
    dfs = []
    for csv_file in sorted(csv_files):
        basename = os.path.basename(csv_file)
        # Extrai o ano do nome do arquivo (ex: PNADC_012012_preprocessado.csv -> 2012)
        try:
            # Formato: PNADC_QQAAAA_preprocessado.csv (QQ=trimestre, AAAA=ano)
            partes = basename.split('_')
            if len(partes) >= 2:
                # Terceiro e quarto caracteres sao o ano
                ano_str = partes[1][2:6]
                ano = int(ano_str)
                if ano in anos:
                    print(f"  Carregando: {basename}...")
                    df_temp = pd.read_csv(csv_file, sep=';')
                    dfs.append(df_temp)
        except Exception as e:
            print(f"  Aviso ao processar {basename}: {e}")
            pass
    
    if not dfs:
        print(f"Erro: Nenhum arquivo encontrado para os anos {anos}")
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"[OK] Dados consolidados: {len(df)} registros totais")
    
    return df

def gerar_grafico_escolaridade(df, ano, segmento):
    """Gera grafico de escolaridade por sexo."""
    # Filtrar por segmento
    if segmento == 'TI':
        df_segmento = df[df['eh_ti'] == True].copy()
        titulo_segmento = "Profissionais de TI"
        nome_arquivo_segmento = "TI"
    else:
        df_segmento = df[df['eh_ti'] == False].copy()
        titulo_segmento = "Demais Profissionais (Nao-TI)"
        nome_arquivo_segmento = "Nao_TI"
    
    print(f"\n--- Processando: Ano {ano} | Segmento: {titulo_segmento} ({len(df_segmento)} registros) ---")
    
    # Filtrar por ano
    df_ano = df_segmento[df_segmento['ano'] == ano].copy()
    
    if len(df_ano) == 0:
        print(f"  Aviso: Sem dados para {titulo_segmento} em {ano}")
        return
    
    # Escolher coluna de escolaridade
    if ano <= 2015:
        coluna_escolaridade = 'curso_mais_elevado_antes_2015'
        mapeamento = MAPEAMENTO_ESCOLARIDADE_PRE_2015
    else:
        coluna_escolaridade = 'curso_mais_elevado'
        mapeamento = MAPEAMENTO_ESCOLARIDADE_POS_2015
    
    # Preparar dados
    df_ano['peso_populacional'] = pd.to_numeric(df_ano['peso_populacional'], errors='coerce')
    df_ano.dropna(subset=[coluna_escolaridade, 'sexo', 'peso_populacional'], inplace=True)
    
    df_ano['nivel_escolaridade'] = df_ano[coluna_escolaridade].astype(int).map(mapeamento)
    df_ano['sexo_desc'] = df_ano['sexo'].astype(int).map(MAPEAMENTO_SEXO)
    df_ano.dropna(subset=['nivel_escolaridade', 'sexo_desc'], inplace=True)
    
    if len(df_ano) == 0:
        print(f"  Erro: Nenhum dado valido para {titulo_segmento} em {ano}")
        return
    
    # Agrupar dados
    dados_agrupados = df_ano.groupby(['nivel_escolaridade', 'sexo_desc'])['peso_populacional'].sum().reset_index()
    dados_agrupados.rename(columns={'peso_populacional': 'populacao_estimada'}, inplace=True)
    
    # Garantir todas as categorias
    all_niveis = [label for key, label in sorted(mapeamento.items())]
    all_sexos = ['Homens', 'Mulheres']
    idx = pd.MultiIndex.from_product([all_niveis, all_sexos], names=['nivel_escolaridade', 'sexo_desc'])
    dados_agrupados = dados_agrupados.set_index(['nivel_escolaridade', 'sexo_desc']).reindex(idx, fill_value=0).reset_index()
    
    dados_agrupados['nivel_escolaridade'] = pd.Categorical(dados_agrupados['nivel_escolaridade'], categories=all_niveis, ordered=True)
    dados_agrupados.sort_values('nivel_escolaridade', inplace=True)
    
    # Criar grafico
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
    
    ax.set_xticks(range(len(all_niveis)))
    ax.set_xticklabels(all_niveis, rotation=45, ha='right', fontsize=12)
    ax.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
    
    # Salvar dados
    try:
        tabela = dados_agrupados.pivot(index='nivel_escolaridade', columns='sexo_desc', values='populacao_estimada').fillna(0)
        tabela = tabela.reset_index()
        base_name = f"escolaridade_sexo_{nome_arquivo_segmento}_{ano}"
        csv_path = f"graficos/{base_name}.csv"
        tabela.to_csv(csv_path, index=False, sep=';', encoding='utf-8')
        print(f"  Tabela CSV salva: {csv_path}")
    except Exception as e:
        print(f"  Aviso na tabela CSV: {e}")
    
    # Formatar e salvar grafico
    ax.set_title(f'Distribuicao por Nivel de Escolaridade e Sexo ({titulo_segmento} - {ano})', fontsize=18, pad=20)
    ax.set_xlabel('Nivel de Escolaridade', fontsize=14)
    ax.set_ylabel('Populacao Estimada', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x):,}'.replace(',', '.')))
    ax.legend(title='Sexo', fontsize=12)
    plt.tight_layout()
    
    caminho_salvar = f'graficos/escolaridade_sexo_{nome_arquivo_segmento}_{ano}.png'
    if os.path.exists(caminho_salvar):
        try:
            os.remove(caminho_salvar)
        except:
            pass
    
    plt.savefig(caminho_salvar, dpi=300)
    plt.close(fig)
    print(f"  Grafico PNG salvo: {caminho_salvar}")

def main():
    """Funcao principal."""
    configurar_ambiente()
    df = carregar_dados_preprocessados_2anos(anos=[2012, 2024])
    
    if df is not None:
        anos_analise = [2012, 2024]
        segmentos = ['TI', 'Nao-TI']
        
        for ano in anos_analise:
            if ano in df['ano'].unique():
                for segmento in segmentos:
                    gerar_grafico_escolaridade(df, ano, segmento)
            else:
                print(f"\nAVISO: O ano {ano} nao esta disponivel no dataset.")
    
    print("\n[OK] Processamento concluido!")
    print("\nArquivos gerados em: graficos/")
    arquivos = os.listdir('graficos')
    if arquivos:
        for arq in sorted(arquivos):
            if 'escolaridade_sexo' in arq:
                print(f"  - {arq}")

if __name__ == "__main__":
    main()
