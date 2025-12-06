import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, MaxNLocator
import warnings
import matplotlib as mpl

warnings.filterwarnings("ignore", message="Passing `palette` without assigning `hue` is deprecated")

def configurar_ambiente():
    """Cria o diretório para salvar os gráficos, se não existir."""
    if not os.path.exists('graficos'):
        os.makedirs('graficos')
    print("Diretório 'graficos' pronto.")

def obter_mapeamento_uf():
    """Retorna um dicionário para mapear o código da UF para a sigla do estado."""
    return {
        11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO',
        21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL',
        28: 'SE', 29: 'BA', 31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP', 41: 'PR',
        42: 'SC', 43: 'RS', 50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'
    }

def carregar_dados():
    """Carrega os dados da PNAD de um arquivo CSV."""
    try:
        # Tenta carregar do caminho especificado (ajuste se necessário)
        df = pd.read_csv(r'C:\TCC\dados\pnad\dados_pnad_consolidados.csv', sep=';')
        print("Dados completos carregados com sucesso.")
    except FileNotFoundError:
        print("Arquivo principal não encontrado. Carregando 'amostra_pnad.csv'.")
        # Se não encontrar, usa o arquivo de amostra
        df = pd.read_csv('amostra_pnad.csv', sep=';')

    # Garante que a coluna 'eh_ti' seja booleana
    if df['eh_ti'].dtype == 'object':
        df['eh_ti'] = df['eh_ti'].str.upper().map({'TRUE': True, 'FALSE': False})

    return df

def processar_dados_por_ano(dataframe, ano):
    """
    Processa os dados para um ano específico, calculando totais absolutos e percentuais
    de profissionais de TI por estado.
    """
    # garantir dtypes numéricos para pesos e flags antes de agregar
    df_all = dataframe.copy()
    df_all['peso_populacional'] = pd.to_numeric(df_all['peso_populacional'], errors='coerce').fillna(0)
    df_all['eh_ti'] = pd.to_numeric(df_all['eh_ti'].astype(object), errors='coerce').fillna(0).astype(int)
    df_all['rendimento_trabalho_principal'] = pd.to_numeric(df_all.get('rendimento_trabalho_principal', 0), errors='coerce').fillna(0)
    df_all['ocupado'] = pd.to_numeric(df_all.get('ocupado', 0), errors='coerce').fillna(0).astype(int)
    df_ano = df_all[df_all['ano'] == ano].copy()
    trimestres = df_ano['trimestre'].unique()

    if len(trimestres) == 0:
        return pd.DataFrame()

    # Agregamos por trimestre, depois somamos os trimestres (ou seja, acumula pesos)
    resultados_trimestrais = []
    for trimestre in trimestres:
        df_trim = df_ano[df_ano['trimestre'] == trimestre].copy()
        # vetorizado: calcula coluna auxiliar e agrega por uf sem groupby.apply
        df_trim['ti_peso'] = df_trim['eh_ti'] * df_trim['peso_populacional']
        total_ti = df_trim.groupby('uf')['ti_peso'].sum().rename('total_ti')
        total_prof = (df_trim.loc[(df_trim['ocupado'] == 1) | (df_trim['rendimento_trabalho_principal'] > 0)]
                              .groupby('uf')['peso_populacional'].sum()
                              .rename('total_profissionais'))
        por_uf = pd.concat([total_ti, total_prof], axis=1)
        resultados_trimestrais.append(por_uf)

    if resultados_trimestrais:
        df_resultados = pd.concat(resultados_trimestrais)
        # Somar os valores ao longo dos trimestres para obter o total anual ponderado
        resultado_anual = df_resultados.groupby('uf').sum().reset_index()
        # evitar divisão por zero
        resultado_anual['percentual_ti'] = (resultado_anual['total_ti'] / resultado_anual['total_profissionais'].replace(0, np.nan)) * 100

        # Mapeia para sigla do estado
        mapeamento_uf = obter_mapeamento_uf()
        resultado_anual['sigla_estado'] = resultado_anual['uf'].map(mapeamento_uf)

        return resultado_anual
    else:
        return pd.DataFrame()

def carregar_mapa_brasil():
    """Tenta carregar o GeoDataFrame do Brasil de várias fontes online."""
    map_sources = [
        'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson',
        'https://raw.githubusercontent.com/luizbd/mapas-geojson-brasil/main/estados.json',
        'https://raw.githubusercontent.com/georgeglaskin/georgeglaskin-capital-quiz/master/data/brazil-states.geojson',
        'https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-100-mun.json'
    
    ]
    mapa_brasil = None
    for src in map_sources:
        try:
            print(f"Tentando carregar mapa de: {src}")
            mapa_brasil = gpd.read_file(src)
            # Normalizar a coluna de sigla para 'sigla'
            if 'sigla' in mapa_brasil.columns:
                pass
            elif 'SIGLA' in mapa_brasil.columns:
                mapa_brasil.rename(columns={'SIGLA': 'sigla'}, inplace=True)
            elif 'id' in mapa_brasil.columns:
                 mapa_brasil.rename(columns={'id': 'sigla'}, inplace=True)
            else:
                 raise ValueError("Coluna de sigla não encontrada no GeoJSON.")
            print(f"Mapa carregado com sucesso de {src}")
            return mapa_brasil
        except Exception as e:
            print(f"Falha ao carregar de {src}: {e}")
    
    print("\nNão foi possível carregar o mapa do Brasil. Serão gerados apenas os gráficos de barras.")
    return None

def criar_visualizacao(dados_ano, ano, mapa_brasil):
    """Cria e salva as visualizações. Gera gráficos de barras sempre, e mapas se disponíveis."""
    if dados_ano.empty:
        print(f"Não há dados suficientes para gerar gráficos para o ano {ano}.")
        return

    # --- Etapa 1: Sempre criar os gráficos de barras ---
    print(f"Gerando gráficos de barras para o ano {ano}...")
    _criar_grafico_barras(dados_ano, ano, 'total_ti', 'Total de Profissionais de TI (ponderado)', 'barra_absoluto', 'Distribuição de Profissionais de TI por Estado')
    _criar_grafico_barras(dados_ano, ano, 'percentual_ti', 'Percentual de Profissionais de TI (%)', 'barra_percentual', 'Representatividade de Profissionais de TI por Estado')
    # --- Etapa 2: Criar mapas de calor apenas se o mapa foi carregado ---
    if mapa_brasil is not None:
        print(f"Gerando mapas de calor para o ano {ano}...")
        mapa_com_dados = mapa_brasil.merge(dados_ano, left_on='sigla', right_on='sigla_estado', how='left')
        
        # ALTERAÇÃO 2: Passa um título descritivo para a legenda do mapa.
        _criar_mapa_calor(mapa_com_dados, ano, 'total_ti', 'Distribuição de Profissionais de TI (Nº Absoluto)', 'mapa_absoluto', 'Nº de Profissionais de TI (estimado)')
        _criar_mapa_calor(mapa_com_dados, ano, 'percentual_ti', 'Concentração de Profissionais de TI (% entre Ocupados)', 'mapa_percentual', '% de Profissionais de TI entre Ocupados')

def _format_large(x, pos=None):
    try:
        x = float(x)
    except Exception:
        return str(x)
    neg = x < 0
    x_abs = abs(x)
    if x_abs >= 1_000_000:
        s = f'{x_abs/1_000_000:.1f}M'
    elif x_abs >= 1_000:
        s = f'{x_abs/1_000:.0f}K'
    else:
        s = f'{int(x_abs):,}'.replace(',', '.')
    return ('-' if neg else '') + s

def _format_percent(x, pos=None):
    try:
        return f'{x:.1f}%'
    except Exception:
        return str(x)

def _criar_mapa_calor(mapa_com_dados, ano, coluna_dados, titulo, nome_arquivo, legenda_label):
    """Função auxiliar para plotar um mapa de calor."""
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    cmap = LinearSegmentedColormap.from_list('GreenScale', [(0.9, 1, 0.9), (0, 0.5, 0)])

    # preparar dados para escala
    values = mapa_com_dados[coluna_dados].dropna()
    if values.empty:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = float(values.min()), float(values.max())

    # desenha sem legend automático; adicionamos colorbar manualmente com formatação
    mapa_com_dados.plot(
        column=coluna_dados, cmap=cmap, linewidth=0.8, ax=ax,
        edgecolor='0.8', legend=False, missing_kwds={'color': 'lightgray'},
    )

    # criar ScalarMappable para colorbar
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(mapa_com_dados[coluna_dados].fillna(0).values)

    # escolhe formatador dependendo se é percentual
    if coluna_dados == 'percentual_ti' or '%' in legenda_label:
        fmt = mpl.ticker.FuncFormatter(lambda val, pos: _format_percent(val))
    else:
        fmt = mpl.ticker.FuncFormatter(lambda val, pos: _format_large(val))

    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04, format=fmt)
    cbar.set_label(legenda_label)

    for idx, row in mapa_com_dados.iterrows():
        if pd.notna(row.get('geometry')):
            try:
                x = row['geometry'].centroid.x
                y = row['geometry'].centroid.y
                plt.annotate(text=row.get('sigla', ''), xy=(x, y),
                             horizontalalignment='center', fontsize=8, color='black')
            except Exception:
                continue

    ax.set_title(f'{titulo} - {ano}', fontsize=16)
    ax.axis('off')

    caminho_salvar = f'graficos/{nome_arquivo}_ti_{ano}.png'
    plt.savefig(caminho_salvar, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Mapa salvo em: {caminho_salvar}")

def _criar_grafico_barras(dados_ano, ano, coluna_y, ylabel, nome_arquivo, titulo):
    """Função auxiliar para plotar um gráfico de barras."""
    plt.figure(figsize=(16, 9))
    dados_ordenados = dados_ano.sort_values(coluna_y, ascending=False)

    ax = sns.barplot(x='sigla_estado', y=coluna_y, data=dados_ordenados, palette='Greens_r')

    # Formatação do eixo Y e rótulos dependendo se é percentual
    if coluna_y == 'percentual_ti' or '%' in ylabel:
        try:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: _format_percent(y)))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        except Exception:
            pass
    else:
        try:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: _format_large(y)))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
        except Exception:
            pass

    # Adicionar rótulos nas barras (formatados)
    for p in ax.patches:
        h = p.get_height()
        if coluna_y == 'percentual_ti' or '%' in ylabel:
            valor = f'{h:.1f}%'
        else:
            valor = _format_large(h)
        # posiciona acima da barra com deslocamento pequeno
        ax.annotate(valor, (p.get_x() + p.get_width() / 2., h),
                    ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 4),
                    textcoords='offset points', clip_on=False)

    plt.title(f'{titulo} - {ano}', fontsize=16)
    plt.xlabel('Estado', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    caminho_salvar = f'graficos/{nome_arquivo}_ti_{ano}.png'
    plt.savefig(caminho_salvar, dpi=300)
    plt.close()
    print(f"Gráfico de barras salvo em: {caminho_salvar}")

def main():
    """Função principal para orquestrar a análise e visualização."""
    configurar_ambiente()
    df = carregar_dados()
    mapa_brasil = carregar_mapa_brasil()

    anos_analise = [2012, 2018, 2024]
    anos_disponiveis = sorted(df['ano'].unique())
    print(f"Anos disponíveis no dataset: {anos_disponiveis}")

    for ano in anos_analise:
        if ano in anos_disponiveis:
            print(f"\n--- Processando dados para o ano {ano} ---")
            dados_ano = processar_dados_por_ano(df, ano)
            criar_visualizacao(dados_ano, ano, mapa_brasil)
        else:
            print(f"\nO ano {ano} não está disponível no dataset.")

    print("\nProcessamento concluído. Verifique os arquivos na pasta 'graficos'.")

if __name__ == "__main__":
    main()