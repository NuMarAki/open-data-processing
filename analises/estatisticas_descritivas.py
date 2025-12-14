import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scripts.utils import configurar_ambiente

def carregar_dados():
    """Carrega os dados da PNAD consolidados."""
    try:
        df = pd.read_csv(r'C:\TCC\dados\pnad\dados_pnad_consolidados.csv', sep=';')
        print("Dados completos carregados com sucesso.")
    except FileNotFoundError:
        print("Arquivo principal não encontrado. Carregando 'amostra_pnad.csv' local.")
        try:
            df = pd.read_csv('amostra_pnad.csv', sep=';')
            print("Dados da amostra carregados com sucesso.")
        except FileNotFoundError:
            print("Erro: Nenhum arquivo de dados encontrado.")
            return None

    # Garantir tipos de dados consistentes
    if df['eh_ti'].dtype == 'object':
        df['eh_ti'] = df['eh_ti'].str.upper().map({'TRUE': True, 'FALSE': False})
    
    # Converter colunas numéricas
    num_cols = ['idade', 'anos_estudo', 'rendimento_trabalho_principal', 
                'rendimento_bruto_mensal', 'peso_populacional']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df

def gerar_estatisticas_gerais(df):
    """Gera estatísticas descritivas gerais sobre a população de TI vs não-TI."""
    print("\n--- Processando estatísticas descritivas gerais ---")
    
    # Filtrar para profissionais ocupados
    df_ocupados = df[(df['ocupado'] == 1) | (df['rendimento_trabalho_principal'] > 0)].copy()
    
    # Separar TI e não-TI
    df_ti = df_ocupados[df_ocupados['eh_ti'] == True]
    df_nao_ti = df_ocupados[df_ocupados['eh_ti'] == False]
    
    # Estatísticas gerais
    stats = []
    
    # Total de profissionais (ponderado por peso_populacional)
    total_ti = df_ti['peso_populacional'].sum()
    total_nao_ti = df_nao_ti['peso_populacional'].sum()
    total_geral = df_ocupados['peso_populacional'].sum()
    
    stats.append({
        'Estatística': 'Total de profissionais (ponderado)',
        'TI': total_ti,
        'Não-TI': total_nao_ti,
        'Total': total_geral,
        'Percentual TI': (total_ti/total_geral)*100 if total_geral > 0 else 0
    })
    
    # Distribuição por sexo
    sexos = {1: 'Masculino', 2: 'Feminino'}
    for sexo_id, sexo_nome in sexos.items():
        ti_sexo = df_ti[df_ti['sexo'] == sexo_id]['peso_populacional'].sum()
        nao_ti_sexo = df_nao_ti[df_nao_ti['sexo'] == sexo_id]['peso_populacional'].sum()
        total_sexo = ti_sexo + nao_ti_sexo
        
        stats.append({
            'Estatística': f'Total {sexo_nome} (ponderado)',
            'TI': ti_sexo,
            'Não-TI': nao_ti_sexo,
            'Total': total_sexo,
            'Percentual TI': (ti_sexo/total_ti)*100 if total_ti > 0 else 0
        })
    
    # Médias de idade e anos de estudo (ponderadas)
    for col, col_nome in [('idade', 'Idade'), ('anos_estudo', 'Anos de estudo')]:
        if col in df.columns:
            # Média ponderada para TI
            ti_media = np.average(
                df_ti[col].dropna(), 
                weights=df_ti.loc[df_ti[col].notna(), 'peso_populacional']
            ) if len(df_ti) > 0 else np.nan
            
            # Média ponderada para não-TI
            nao_ti_media = np.average(
                df_nao_ti[col].dropna(), 
                weights=df_nao_ti.loc[df_nao_ti[col].notna(), 'peso_populacional']
            ) if len(df_nao_ti) > 0 else np.nan
            
            # Média ponderada geral
            total_media = np.average(
                df_ocupados[col].dropna(), 
                weights=df_ocupados.loc[df_ocupados[col].notna(), 'peso_populacional']
            ) if len(df_ocupados) > 0 else np.nan
            
            stats.append({
                'Estatística': f'Média de {col_nome} (ponderada)',
                'TI': ti_media,
                'Não-TI': nao_ti_media,
                'Total': total_media,
                'Diferença (TI - Não-TI)': ti_media - nao_ti_media if not np.isnan(ti_media) and not np.isnan(nao_ti_media) else np.nan
            })
    
    # Estatísticas de rendimento (ponderadas)
    for col, col_nome in [('rendimento_trabalho_principal', 'Rendimento do trabalho principal')]:
        if col in df.columns:
            # Estatísticas para TI
            ti_rendimento = df_ti[col].dropna()
            ti_pesos = df_ti.loc[df_ti[col].notna(), 'peso_populacional']
            
            # Estatísticas para não-TI
            nao_ti_rendimento = df_nao_ti[col].dropna()
            nao_ti_pesos = df_nao_ti.loc[df_nao_ti[col].notna(), 'peso_populacional']
            
            # Estatísticas para total
            total_rendimento = df_ocupados[col].dropna()
            total_pesos = df_ocupados.loc[df_ocupados[col].notna(), 'peso_populacional']
            
            # Média ponderada
            ti_media = np.average(ti_rendimento, weights=ti_pesos) if len(ti_rendimento) > 0 else np.nan
            nao_ti_media = np.average(nao_ti_rendimento, weights=nao_ti_pesos) if len(nao_ti_rendimento) > 0 else np.nan
            total_media = np.average(total_rendimento, weights=total_pesos) if len(total_rendimento) > 0 else np.nan
            
            stats.append({
                'Estatística': f'Média de {col_nome} (ponderada)',
                'TI': ti_media,
                'Não-TI': nao_ti_media,
                'Total': total_media,
                'Razão TI/Não-TI': ti_media / nao_ti_media if not np.isnan(ti_media) and not np.isnan(nao_ti_media) and nao_ti_media > 0 else np.nan
            })
            
            # Mediana ponderada (calculada com percentil 50)
            ti_mediana = weighted_percentile(ti_rendimento, ti_pesos, 50) if len(ti_rendimento) > 0 else np.nan
            nao_ti_mediana = weighted_percentile(nao_ti_rendimento, nao_ti_pesos, 50) if len(nao_ti_rendimento) > 0 else np.nan
            total_mediana = weighted_percentile(total_rendimento, total_pesos, 50) if len(total_rendimento) > 0 else np.nan
            
            stats.append({
                'Estatística': f'Mediana de {col_nome} (ponderada)',
                'TI': ti_mediana,
                'Não-TI': nao_ti_mediana,
                'Total': total_mediana,
                'Razão TI/Não-TI': ti_mediana / nao_ti_mediana if not np.isnan(ti_mediana) and not np.isnan(nao_ti_mediana) and nao_ti_mediana > 0 else np.nan
            })
    
    # Converter para DataFrame e salvar
    df_stats = pd.DataFrame(stats)
    
    # Salvar como CSV
    caminho_salvar = 'estatisticas/estatisticas_gerais.csv'
    df_stats.to_csv(caminho_salvar, index=False, sep=';', encoding='utf-8')
    print(f"Estatísticas gerais salvas em: {caminho_salvar}")
    
    return df_stats

def gerar_estatisticas_por_ano(df):
    """Gera estatísticas descritivas por ano para TI vs não-TI."""
    print("\n--- Processando estatísticas descritivas por ano ---")
    
    # Filtrar para profissionais ocupados
    df_ocupados = df[(df['ocupado'] == 1) | (df['rendimento_trabalho_principal'] > 0)].copy()
    
    # Inicializar listas para cada estatística
    anos = sorted(df_ocupados['ano'].unique())
    resultados = []
    
    for ano in anos:
        df_ano = df_ocupados[df_ocupados['ano'] == ano]
        df_ti_ano = df_ano[df_ano['eh_ti'] == True]
        df_nao_ti_ano = df_ano[df_ano['eh_ti'] == False]
        
        # Total de profissionais
        total_ti = df_ti_ano['peso_populacional'].sum()
        total_nao_ti = df_nao_ti_ano['peso_populacional'].sum()
        
        # Percentual de TI
        perc_ti = (total_ti / (total_ti + total_nao_ti)) * 100 if (total_ti + total_nao_ti) > 0 else 0
        
        # Média de idade (ponderada)
        idade_ti = np.average(
            df_ti_ano['idade'].dropna(),
            weights=df_ti_ano.loc[df_ti_ano['idade'].notna(), 'peso_populacional']
        ) if 'idade' in df.columns and len(df_ti_ano) > 0 else np.nan
        
        idade_nao_ti = np.average(
            df_nao_ti_ano['idade'].dropna(),
            weights=df_nao_ti_ano.loc[df_nao_ti_ano['idade'].notna(), 'peso_populacional']
        ) if 'idade' in df.columns and len(df_nao_ti_ano) > 0 else np.nan
        
        # Média de anos de estudo (ponderada)
        estudo_ti = np.average(
            df_ti_ano['anos_estudo'].dropna(),
            weights=df_ti_ano.loc[df_ti_ano['anos_estudo'].notna(), 'peso_populacional']
        ) if 'anos_estudo' in df.columns and len(df_ti_ano) > 0 else np.nan
        
        estudo_nao_ti = np.average(
            df_nao_ti_ano['anos_estudo'].dropna(),
            weights=df_nao_ti_ano.loc[df_nao_ti_ano['anos_estudo'].notna(), 'peso_populacional']
        ) if 'anos_estudo' in df.columns and len(df_nao_ti_ano) > 0 else np.nan
        
        # Rendimento médio (ponderado)
        renda_ti = np.average(
            df_ti_ano['rendimento_trabalho_principal'].dropna(),
            weights=df_ti_ano.loc[df_ti_ano['rendimento_trabalho_principal'].notna(), 'peso_populacional']
        ) if 'rendimento_trabalho_principal' in df.columns and len(df_ti_ano) > 0 else np.nan
        
        renda_nao_ti = np.average(
            df_nao_ti_ano['rendimento_trabalho_principal'].dropna(),
            weights=df_nao_ti_ano.loc[df_nao_ti_ano['rendimento_trabalho_principal'].notna(), 'peso_populacional']
        ) if 'rendimento_trabalho_principal' in df.columns and len(df_nao_ti_ano) > 0 else np.nan
        
        # Razão de rendimento TI/não-TI
        razao_renda = renda_ti / renda_nao_ti if not np.isnan(renda_ti) and not np.isnan(renda_nao_ti) and renda_nao_ti > 0 else np.nan
        
        # Adicionar à lista de resultados
        resultados.append({
            'Ano': ano,
            'Total TI': total_ti,
            'Total Não-TI': total_nao_ti,
            'Percentual TI (%)': perc_ti,
            'Idade Média TI': idade_ti,
            'Idade Média Não-TI': idade_nao_ti,
            'Anos Estudo TI': estudo_ti,
            'Anos Estudo Não-TI': estudo_nao_ti,
            'Renda Média TI': renda_ti,
            'Renda Média Não-TI': renda_nao_ti,
            'Razão Renda (TI/Não-TI)': razao_renda
        })
    
    # Converter para DataFrame e salvar
    df_anual = pd.DataFrame(resultados)
    
    # Salvar como CSV
    caminho_salvar = 'estatisticas/estatisticas_por_ano.csv'
    df_anual.to_csv(caminho_salvar, index=False, sep=';', encoding='utf-8')
    print(f"Estatísticas por ano salvas em: {caminho_salvar}")
    
    # Gerar gráfico de evolução temporal das métricas
    if len(df_anual) > 1:
        gerar_graficos_evolucao(df_anual)
    
    return df_anual

def gerar_estatisticas_por_uf(df):
    """Gera estatísticas descritivas por UF para TI vs não-TI."""
    print("\n--- Processando estatísticas descritivas por UF ---")
    
    # Filtrar para profissionais ocupados
    df_ocupados = df[(df['ocupado'] == 1) | (df['rendimento_trabalho_principal'] > 0)].copy()
    
    # Mapeamento de código UF para sigla
    mapeamento_uf = {
        11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO',
        21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA',
        31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP', 41: 'PR', 42: 'SC', 43: 'RS',
        50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'
    }
    
    # Pegar o último ano disponível para análise por UF
    ultimo_ano = df_ocupados['ano'].max()
    df_ultimo_ano = df_ocupados[df_ocupados['ano'] == ultimo_ano]
    
    # Inicializar resultados
    resultados = []
    
    for uf in sorted(df_ultimo_ano['uf'].unique()):
        df_uf = df_ultimo_ano[df_ultimo_ano['uf'] == uf]
        df_ti_uf = df_uf[df_uf['eh_ti'] == True]
        df_nao_ti_uf = df_uf[df_uf['eh_ti'] == False]
        
        # Total de profissionais
        total_ti = df_ti_uf['peso_populacional'].sum()
        total_nao_ti = df_nao_ti_uf['peso_populacional'].sum()
        
        # Percentual de TI
        perc_ti = (total_ti / (total_ti + total_nao_ti)) * 100 if (total_ti + total_nao_ti) > 0 else 0
        
        # Renda média (ponderada)
        renda_ti = np.average(
            df_ti_uf['rendimento_trabalho_principal'].dropna(),
            weights=df_ti_uf.loc[df_ti_uf['rendimento_trabalho_principal'].notna(), 'peso_populacional']
        ) if 'rendimento_trabalho_principal' in df.columns and len(df_ti_uf) > 0 else np.nan
        
        renda_nao_ti = np.average(
            df_nao_ti_uf['rendimento_trabalho_principal'].dropna(),
            weights=df_nao_ti_uf.loc[df_nao_ti_uf['rendimento_trabalho_principal'].notna(), 'peso_populacional']
        ) if 'rendimento_trabalho_principal' in df.columns and len(df_nao_ti_uf) > 0 else np.nan
        
        # Razão de rendimento TI/não-TI
        razao_renda = renda_ti / renda_nao_ti if not np.isnan(renda_ti) and not np.isnan(renda_nao_ti) and renda_nao_ti > 0 else np.nan
        
        # Mapear código UF para sigla
        sigla_uf = mapeamento_uf.get(uf, str(uf))
        
        # Adicionar à lista de resultados
        resultados.append({
            'UF': sigla_uf,
            'Código UF': uf,
            'Total TI': total_ti,
            'Total Não-TI': total_nao_ti,
            'Percentual TI (%)': perc_ti,
            'Renda Média TI': renda_ti,
            'Renda Média Não-TI': renda_nao_ti,
            'Razão Renda (TI/Não-TI)': razao_renda
        })
    
    # Converter para DataFrame e salvar
    df_uf = pd.DataFrame(resultados)
    
    # Salvar como CSV
    caminho_salvar = f'estatisticas/estatisticas_por_uf_{ultimo_ano}.csv'
    df_uf.to_csv(caminho_salvar, index=False, sep=';', encoding='utf-8')
    print(f"Estatísticas por UF para {ultimo_ano} salvas em: {caminho_salvar}")
    
    return df_uf

def gerar_estatisticas_distribuicao_idade(df):
    """Gera estatísticas sobre a distribuição de idade para TI vs não-TI."""
    print("\n--- Processando estatísticas de distribuição de idade ---")
    
    # Filtrar para profissionais ocupados
    df_ocupados = df[(df['ocupado'] == 1) | (df['rendimento_trabalho_principal'] > 0)].copy()
    
    # Pegar o último ano disponível
    ultimo_ano = df_ocupados['ano'].max()
    df_ultimo_ano = df_ocupados[df_ocupados['ano'] == ultimo_ano]
    
    # Definir faixas etárias
    bins = [0, 24, 34, 44, 54, 64, 120]
    labels = ['<=24', '25-34', '35-44', '45-54', '55-64', '65+']
    df_ultimo_ano['faixa_idade'] = pd.cut(df_ultimo_ano['idade'], bins=bins, labels=labels, right=True)
    
    # Separar TI e não-TI
    df_ti = df_ultimo_ano[df_ultimo_ano['eh_ti'] == True]
    df_nao_ti = df_ultimo_ano[df_ultimo_ano['eh_ti'] == False]
    
    # Calcular distribuição para TI
    dist_ti = df_ti.groupby('faixa_idade')['peso_populacional'].sum()
    dist_ti = dist_ti / dist_ti.sum() * 100 if dist_ti.sum() > 0 else dist_ti
    
    # Calcular distribuição para não-TI
    dist_nao_ti = df_nao_ti.groupby('faixa_idade')['peso_populacional'].sum()
    dist_nao_ti = dist_nao_ti / dist_nao_ti.sum() * 100 if dist_nao_ti.sum() > 0 else dist_nao_ti
    
    # Calcular média e mediana de rendimento por faixa etária
    resultados = []
    
    for faixa in labels:
        # TI
        ti_faixa = df_ti[df_ti['faixa_idade'] == faixa]
        ti_pop = ti_faixa['peso_populacional'].sum()
        ti_renda_media = np.average(
            ti_faixa['rendimento_trabalho_principal'].dropna(),
            weights=ti_faixa.loc[ti_faixa['rendimento_trabalho_principal'].notna(), 'peso_populacional']
        ) if len(ti_faixa) > 0 else np.nan
        
        # Não-TI
        nao_ti_faixa = df_nao_ti[df_nao_ti['faixa_idade'] == faixa]
        nao_ti_pop = nao_ti_faixa['peso_populacional'].sum()
        nao_ti_renda_media = np.average(
            nao_ti_faixa['rendimento_trabalho_principal'].dropna(),
            weights=nao_ti_faixa.loc[nao_ti_faixa['rendimento_trabalho_principal'].notna(), 'peso_populacional']
        ) if len(nao_ti_faixa) > 0 else np.nan
        
        # Percentual da população
        perc_ti = dist_ti[faixa] if faixa in dist_ti.index else 0
        perc_nao_ti = dist_nao_ti[faixa] if faixa in dist_nao_ti.index else 0
        
        # Razão de rendimento
        razao_renda = ti_renda_media / nao_ti_renda_media if not np.isnan(ti_renda_media) and not np.isnan(nao_ti_renda_media) and nao_ti_renda_media > 0 else np.nan
        
        resultados.append({
            'Faixa Etária': faixa,
            'Pop. TI': ti_pop,
            'Pop. Não-TI': nao_ti_pop,
            'Perc. Pop. TI (%)': perc_ti,
            'Perc. Pop. Não-TI (%)': perc_nao_ti,
            'Renda Média TI': ti_renda_media,
            'Renda Média Não-TI': nao_ti_renda_media,
            'Razão Renda (TI/Não-TI)': razao_renda
        })
    
    # Converter para DataFrame e salvar
    df_idade = pd.DataFrame(resultados)
    
    # Salvar como CSV
    caminho_salvar = f'estatisticas/estatisticas_distribuicao_idade_{ultimo_ano}.csv'
    df_idade.to_csv(caminho_salvar, index=False, sep=';', encoding='utf-8')
    print(f"Estatísticas de distribuição de idade para {ultimo_ano} salvas em: {caminho_salvar}")
    
    # Gerar gráfico de distribuição de idade
    gerar_grafico_distribuicao_idade(df_idade, ultimo_ano)
    
    return df_idade

def gerar_estatisticas_escolaridade(df):
    """Gera estatísticas sobre a distribuição de escolaridade para TI vs não-TI."""
    print("\n--- Processando estatísticas de escolaridade ---")
    
    # Filtrar para profissionais ocupados
    df_ocupados = df[(df['ocupado'] == 1) | (df['rendimento_trabalho_principal'] > 0)].copy()
    
    # Pegar o último ano disponível
    ultimo_ano = df_ocupados['ano'].max()
    df_ultimo_ano = df_ocupados[df_ocupados['ano'] == ultimo_ano]
    
    # Determinar coluna de escolaridade a ser usada com base no ano
    if ultimo_ano <= 2015:
        coluna_escolaridade = 'curso_mais_elevado_antes_2015'
        mapeamento = {
            1: 'Creche / Pré-Escola',
            2: 'Alfabetização de Jovens/Adultos',
            3: 'Antigo primário (elementar)',
            4: 'Antigo ginásio (médio 1º ciclo)',
            5: 'Ensino Fundamental (1º Grau)',
            6: 'EJA - Ensino Fundamental',
            7: 'Antigo científico, clássico, etc. (médio 2º ciclo)',
            8: 'Regular do ensino médio ou do 2º grau',
            9: 'EJA ou supletivo do ensino médio',
            10: 'Superior - Graduação',
            11: 'Mestrado',
            12: 'Doutorado'
        }
    else:
        coluna_escolaridade = 'curso_mais_elevado'
        mapeamento = {
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
    
    # Mapear código para descrição
    df_ultimo_ano['nivel_escolaridade'] = df_ultimo_ano[coluna_escolaridade].map(mapeamento)
    
    # Separar TI e não-TI
    df_ti = df_ultimo_ano[df_ultimo_ano['eh_ti'] == True]
    df_nao_ti = df_ultimo_ano[df_ultimo_ano['eh_ti'] == False]
    
    # Calcular estatísticas por nível de escolaridade
    resultados = []
    
    # Agrupar por nível de escolaridade
    for nivel_id, nivel_nome in sorted(mapeamento.items()):
        # TI
        ti_nivel = df_ti[df_ti[coluna_escolaridade] == nivel_id]
        ti_pop = ti_nivel['peso_populacional'].sum()
        ti_renda_media = np.average(
            ti_nivel['rendimento_trabalho_principal'].dropna(),
            weights=ti_nivel.loc[ti_nivel['rendimento_trabalho_principal'].notna(), 'peso_populacional']
        ) if len(ti_nivel) > 0 else np.nan
        
        # Não-TI
        nao_ti_nivel = df_nao_ti[df_nao_ti[coluna_escolaridade] == nivel_id]
        nao_ti_pop = nao_ti_nivel['peso_populacional'].sum()
        nao_ti_renda_media = np.average(
            nao_ti_nivel['rendimento_trabalho_principal'].dropna(),
            weights=nao_ti_nivel.loc[nao_ti_nivel['rendimento_trabalho_principal'].notna(), 'peso_populacional']
        ) if len(nao_ti_nivel) > 0 else np.nan
        
        # Percentual da população
        perc_ti = (ti_pop / df_ti['peso_populacional'].sum()) * 100 if df_ti['peso_populacional'].sum() > 0 else 0
        perc_nao_ti = (nao_ti_pop / df_nao_ti['peso_populacional'].sum()) * 100 if df_nao_ti['peso_populacional'].sum() > 0 else 0
        
        # Razão de rendimento
        razao_renda = ti_renda_media / nao_ti_renda_media if not np.isnan(ti_renda_media) and not np.isnan(nao_ti_renda_media) and nao_ti_renda_media > 0 else np.nan
        
        resultados.append({
            'Código': nivel_id,
            'Nível de Escolaridade': nivel_nome,
            'Pop. TI': ti_pop,
            'Pop. Não-TI': nao_ti_pop,
            'Perc. Pop. TI (%)': perc_ti,
            'Perc. Pop. Não-TI (%)': perc_nao_ti,
            'Renda Média TI': ti_renda_media,
            'Renda Média Não-TI': nao_ti_renda_media,
            'Razão Renda (TI/Não-TI)': razao_renda
        })
    
    # Converter para DataFrame e salvar
    df_escolaridade = pd.DataFrame(resultados)
    
    # Salvar como CSV
    caminho_salvar = f'estatisticas/estatisticas_escolaridade_{ultimo_ano}.csv'
    df_escolaridade.to_csv(caminho_salvar, index=False, sep=';', encoding='utf-8')
    print(f"Estatísticas de escolaridade para {ultimo_ano} salvas em: {caminho_salvar}")
    
    return df_escolaridade

def weighted_percentile(data, weights, percentile):
    """Calcula o percentil ponderado de uma distribuição."""
    # Convert pandas Series to numpy arrays to avoid indexing issues
    data_array = data.values if hasattr(data, 'values') else np.array(data)
    weights_array = weights.values if hasattr(weights, 'values') else np.array(weights)
    
    # Ordene os dados e os pesos correspondentes
    sorted_indices = np.argsort(data_array)
    sorted_data = data_array[sorted_indices]
    sorted_weights = weights_array[sorted_indices]
    
    # Calcule os pesos cumulativos
    cumsum_weights = np.cumsum(sorted_weights)
    
    # Normalize para ter uma soma de 1
    cumsum_weights = cumsum_weights / cumsum_weights[-1] if len(cumsum_weights) > 0 else np.array([0])
    
    # Interpole para encontrar o percentil
    idx = np.searchsorted(cumsum_weights, percentile/100)
    
    if idx == 0:
        return sorted_data[0] if len(sorted_data) > 0 else np.nan
    elif idx >= len(sorted_data):
        return sorted_data[-1] if len(sorted_data) > 0 else np.nan
    else:
        # Interpolação linear entre os pontos mais próximos
        x0 = cumsum_weights[idx-1]
        x1 = cumsum_weights[idx]
        y0 = sorted_data[idx-1]
        y1 = sorted_data[idx]
        
        return y0 + (y1 - y0) * (percentile/100 - x0) / (x1 - x0)

def gerar_graficos_evolucao(df_anual):
    """Gera gráficos de evolução temporal das principais métricas."""
    # 1. Percentual de TI ao longo dos anos
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_anual, x='Ano', y='Percentual TI (%)', marker='o', linewidth=2)
    plt.title('Evolução do Percentual de Profissionais de TI', fontsize=14)
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Percentual (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('graficos/evolucao_percentual_ti.png', dpi=300)
    plt.close()
    
    # 2. Razão de Renda TI/Não-TI
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_anual, x='Ano', y='Razão Renda (TI/Não-TI)', marker='o', linewidth=2)
    plt.title('Evolução da Razão de Renda (TI/Não-TI)', fontsize=14)
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Razão de Renda', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.5)  # Linha de referência onde razão = 1
    plt.tight_layout()
    plt.savefig('graficos/evolucao_razao_renda.png', dpi=300)
    plt.close()
    
    # 3. Evolução da renda média de TI e Não-TI
    plt.figure(figsize=(12, 8))
    df_anual_melt = df_anual.melt(id_vars='Ano', value_vars=['Renda Média TI', 'Renda Média Não-TI'],
                                 var_name='Categoria', value_name='Renda Média')
    sns.lineplot(data=df_anual_melt, x='Ano', y='Renda Média', hue='Categoria', marker='o', linewidth=2)
    plt.title('Evolução da Renda Média por Categoria', fontsize=14)
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Renda Média (R$)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('graficos/evolucao_renda_media.png', dpi=300)
    plt.close()
    
    # 4. Evolução da idade média
    plt.figure(figsize=(12, 8))
    df_anual_melt = df_anual.melt(id_vars='Ano', value_vars=['Idade Média TI', 'Idade Média Não-TI'],
                                 var_name='Categoria', value_name='Idade Média')
    sns.lineplot(data=df_anual_melt, x='Ano', y='Idade Média', hue='Categoria', marker='o', linewidth=2)
    plt.title('Evolução da Idade Média por Categoria', fontsize=14)
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Idade Média', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('graficos/evolucao_idade_media.png', dpi=300)
    plt.close()

def gerar_grafico_distribuicao_idade(df_idade, ano):
    """Gera gráfico comparando a distribuição etária entre TI e não-TI."""
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(df_idade))
    width = 0.35
    
    # Crie duas barras lado a lado
    plt.bar(x - width/2, df_idade['Perc. Pop. TI (%)'], width, label='TI')
    plt.bar(x + width/2, df_idade['Perc. Pop. Não-TI (%)'], width, label='Não-TI')
    
    plt.title(f'Distribuição Etária: TI vs Não-TI ({ano})', fontsize=14)
    plt.xlabel('Faixa Etária', fontsize=12)
    plt.ylabel('Percentual da População (%)', fontsize=12)
    plt.xticks(x, df_idade['Faixa Etária'])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adicione os valores sobre as barras
    for i, v in enumerate(df_idade['Perc. Pop. TI (%)']):
        plt.text(i - width/2, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
    
    for i, v in enumerate(df_idade['Perc. Pop. Não-TI (%)']):
        plt.text(i + width/2, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'graficos/distribuicao_etaria_comparativo_{ano}.png', dpi=300)
    plt.close()

def main():
    """Função principal para coordenar a geração de estatísticas descritivas."""
    configurar_ambiente('estatisticas')
    configurar_ambiente('graficos')
    df = carregar_dados()
    
    if df is not None:
        # Gerar as estatísticas
        gerar_estatisticas_gerais(df)
        gerar_estatisticas_por_ano(df)
        gerar_estatisticas_por_uf(df)
        gerar_estatisticas_distribuicao_idade(df)
        gerar_estatisticas_escolaridade(df)
        
        print("\nProcessamento de estatísticas descritivas concluído!")
        print("Os resultados foram salvos na pasta 'estatisticas/'.")
        print("Gráficos adicionais foram salvos na pasta 'graficos/'.")

if __name__ == "__main__":
    main()