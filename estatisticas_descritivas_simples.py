import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate  # Import the tabulate function directly

def configurar_ambiente():
    """Cria o diretório para salvar as estatísticas descritivas."""
    dirs = ['estatisticas', 'graficos']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
    print("Diretórios 'estatisticas' e 'graficos' prontos.")

def carregar_dados():
    """Carrega os dados da PNAD consolidados."""
    try:
        df = pd.read_csv(r'C:\TCC\dados\pnad\dados_pnad_consolidados.csv', sep=';')
        print(f"Dados completos carregados com sucesso: {len(df):,} registros.")
    except FileNotFoundError:
        print("Arquivo principal não encontrado. Carregando 'amostra_pnad.csv' local.")
        try:
            df = pd.read_csv('amostra_pnad.csv', sep=';')
            print(f"Dados da amostra carregados com sucesso: {len(df):,} registros.")
        except FileNotFoundError:
            print("Erro: Nenhum arquivo de dados encontrado.")
            return None

    # Converter colunas numéricas
    num_cols = ['idade', 'anos_estudo', 'rendimento_trabalho_principal', 
                'rendimento_bruto_mensal', 'peso_populacional', 'ocupado']
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Garantir que eh_ti seja booleano
    if 'eh_ti' in df.columns:
        if df['eh_ti'].dtype == 'object':
            df['eh_ti'] = df['eh_ti'].str.upper().map({'TRUE': 1, 'FALSE': 0})
        df['eh_ti'] = df['eh_ti'].astype(int)
    
    return df

def estatisticas_ponderadas(series, weights, percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]):
    """Calcula estatísticas ponderadas para uma série"""
    data = series.dropna()
    weights_valid = weights.loc[data.index]
    
    if len(data) == 0 or weights_valid.sum() == 0:
        return {
            'count': 0,
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            **{f'{int(p*100)}%': np.nan for p in percentiles}
        }
    
    # Média ponderada
    mean = np.average(data, weights=weights_valid)
    
    # Desvio padrão ponderado
    variance = np.average((data - mean)**2, weights=weights_valid)
    std = np.sqrt(variance)
    
    # Mínimo e máximo
    min_val = data.min()
    max_val = data.max()
    
    # Quantis ponderados
    result = {
        'count': len(data),
        'count_weighted': weights_valid.sum(),
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val
    }
    
    # Calcular percentis ponderados
    indices_sorted = np.argsort(data)
    data_sorted = data.iloc[indices_sorted]
    weights_sorted = weights_valid.iloc[indices_sorted]
    
    cumsum = weights_sorted.cumsum()
    cumsum = cumsum / cumsum.iloc[-1]
    
    for p in percentiles:
        # Encontrar o índice onde a soma acumulada de pesos ultrapassa o percentil
        idx = np.searchsorted(cumsum, p, side='right')
        if idx == 0:
            result[f'{int(p*100)}%'] = data_sorted.iloc[0]
        else:
            result[f'{int(p*100)}%'] = data_sorted.iloc[idx-1]
    
    return result

def gerar_tabela_estatisticas_ponderadas(df):
    """Gera tabelas de estatísticas descritivas ponderadas pelo peso populacional"""
    print("\n--- Gerando estatísticas descritivas ponderadas ---")
    
    # Filtrar para profissionais ocupados
    df_ocupados = df[(df['ocupado'] == 1) | (df['rendimento_trabalho_principal'] > 0)].copy()
    
    # Separar TI e não-TI
    df_ti = df_ocupados[df_ocupados['eh_ti'] == 1]
    df_nao_ti = df_ocupados[df_ocupados['eh_ti'] == 0]
    
    # População representada (soma dos pesos)
    pop_ti = df_ti['peso_populacional'].sum()
    pop_nao_ti = df_nao_ti['peso_populacional'].sum()
    
    print(f"\nPopulação total representada (ponderada): {pop_ti + pop_nao_ti:,.0f}")
    print(f"População TI representada: {pop_ti:,.0f} ({pop_ti/(pop_ti + pop_nao_ti)*100:.2f}%)")
    print(f"População não-TI representada: {pop_nao_ti:,.0f} ({pop_nao_ti/(pop_ti + pop_nao_ti)*100:.2f}%)")
    
    # Colunas para análise
    cols = ['idade', 'anos_estudo', 'rendimento_trabalho_principal']
    
    # Gerar estatísticas para cada coluna
    resultados = {}
    
    for col in cols:
        # TI
        stats_ti = estatisticas_ponderadas(df_ti[col], df_ti['peso_populacional'])
        
        # Não-TI
        stats_nao_ti = estatisticas_ponderadas(df_nao_ti[col], df_nao_ti['peso_populacional'])
        
        resultados[col] = {
            'TI': stats_ti,
            'Não-TI': stats_nao_ti
        }
    
    # Salvar resultados em CSV formatado
    for col, stats in resultados.items():
        # Criar DataFrame para salvar
        df_stats = pd.DataFrame({
            'Estatística': list(stats['TI'].keys()),
            'TI': list(stats['TI'].values()),
            'Não-TI': list(stats['Não-TI'].values()),
            'Diferença (TI - Não-TI)': [
                stats['TI'][k] - stats['Não-TI'][k] if k in stats['Não-TI'] and k in stats['TI'] else np.nan
                for k in stats['TI'].keys()
            ]
        })
        
        # Formatar o CSV para evitar notação científica
        pd.options.display.float_format = '{:.2f}'.format
        df_stats.to_csv(f'estatisticas/estatisticas_{col}_ponderadas.csv', index=False)
        
        # Mostrar no console de forma formatada - AQUI ESTÁ A CORREÇÃO:
        print(f"\n\nEstatísticas para {col} (ponderadas por peso_populacional):")
        print(tabulate(df_stats.values.tolist(), headers=list(df_stats.columns), tablefmt='psql', showindex=False, floatfmt=".2f"))
   
    # Gerar estatísticas por ano
    gerar_estatisticas_por_ano(df_ocupados)
    
    return resultados

def gerar_estatisticas_por_ano(df_ocupados):
    """Gera estatísticas anuais ponderadas"""
    print("\n\n--- Estatísticas por ano ---")
    
    # Calcular estatísticas por ano
    anos = sorted(df_ocupados['ano'].unique())
    resultados = []
    
    for ano in anos:
        df_ano = df_ocupados[df_ocupados['ano'] == ano]
        
        # TI e não-TI
        df_ti_ano = df_ano[df_ano['eh_ti'] == 1]
        df_nao_ti_ano = df_ano[df_ano['eh_ti'] == 0]
        
        # População representada (soma dos pesos)
        pop_ti = df_ti_ano['peso_populacional'].sum()
        pop_nao_ti = df_nao_ti_ano['peso_populacional'].sum()
        
        # Média de rendimento ponderada
        if len(df_ti_ano) > 0:
            renda_ti = np.average(
                df_ti_ano['rendimento_trabalho_principal'].dropna(),
                weights=df_ti_ano.loc[df_ti_ano['rendimento_trabalho_principal'].notna(), 'peso_populacional']
            )
        else:
            renda_ti = np.nan
            
        if len(df_nao_ti_ano) > 0:
            renda_nao_ti = np.average(
                df_nao_ti_ano['rendimento_trabalho_principal'].dropna(),
                weights=df_nao_ti_ano.loc[df_nao_ti_ano['rendimento_trabalho_principal'].notna(), 'peso_populacional']
            )
        else:
            renda_nao_ti = np.nan
        
        # Razão de rendimento TI/Não-TI
        razao_renda = renda_ti / renda_nao_ti if not np.isnan(renda_ti) and not np.isnan(renda_nao_ti) else np.nan
        
        resultados.append({
            'Ano': ano,
            'População TI': pop_ti,
            'População Não-TI': pop_nao_ti,
            'Percentual TI (%)': (pop_ti / (pop_ti + pop_nao_ti) * 100) if (pop_ti + pop_nao_ti) > 0 else 0,
            'Renda Média TI': renda_ti,
            'Renda Média Não-TI': renda_nao_ti,
            'Razão Renda (TI/Não-TI)': razao_renda
        })
    
    # Criar DataFrame e salvar
    df_anual = pd.DataFrame(resultados)
    
    # Salvar como CSV formatado
    df_anual.to_csv('estatisticas/estatisticas_anuais.csv', index=False)
    
    # Mostrar no console - AQUI ESTÁ OUTRA CORREÇÃO:
    print("\nResumo anual:")
    print(tabulate(df_anual.values.tolist(), headers=list(df_anual.columns), tablefmt='psql', showindex=False, floatfmt=".2f"))

def main():
    """Função principal para coordenar a geração de estatísticas descritivas."""
    configurar_ambiente()
    df = carregar_dados()
    
            
    if df is not None:
        print(f"\n\nEstatísticas para Consolidados PNAD - Sem peso populacional aplicado:")
        describe_df = df.describe().T
        print(tabulate(describe_df.reset_index().values.tolist(), headers=[""] + list(describe_df.columns), tablefmt='psql', showindex=False, floatfmt=".2f"))

        # Gerar estatísticas descritivas ponderadas
        resultados = gerar_tabela_estatisticas_ponderadas(df)
        
        print("\nProcessamento concluído! Todos os resultados foram salvos na pasta 'estatisticas'.")
    else:
        print("Erro: Nenhum dado carregado para mostrar estatísticas.")

if __name__ == "__main__":
    try:
        from tabulate import tabulate
    except ImportError:
        print("Instalando pacote 'tabulate' para formatação de tabelas...")
        import pip
        pip.main(['install', 'tabulate'])
        from tabulate import tabulate
    
    main()