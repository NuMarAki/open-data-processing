import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from matplotlib.ticker import FuncFormatter
from scripts.utils import configurar_ambiente

# Formatter global K/M para evitar notação exponencial
def _fmt_km(x, pos):
    try:
        if np.isnan(x):
            return ''
    except:
        pass
    absx = abs(x)
    if absx >= 1_000_000:
        v = x / 1_000_000.0
        return f'{v:.1f}M' if abs(v) < 10 else f'{v:.0f}M'
    if absx >= 1_000:
        v = x / 1_000.0
        return f'{v:.1f}K' if abs(v) < 10 else f'{v:.0f}K'
    return f'{int(x)}'

formatter_km = FuncFormatter(lambda x, pos: _fmt_km(x, pos))
formatter_reais = FuncFormatter(lambda x, pos: f'R$ {_fmt_km(x, pos)}')

def carregar_dados():
    """
    Carrega e consolida arquivos preprocessados da PNAD, filtrando anos 2012 e 2024 para otimizar.
    """
    import glob
    
    # Tentar múltiplos caminhos
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
        print("Erro: Pasta 'preprocessados' não encontrada.")
        return None
    
    print(f"Carregando arquivos preprocessados de: {preprocessados_dir}")
    
    # Encontrar todos os arquivos .csv preprocessados
    csv_files = glob.glob(os.path.join(preprocessados_dir, '*preprocessado.csv'))
    
    if not csv_files:
        print(f"Erro: Nenhum arquivo preprocessado encontrado em {preprocessados_dir}")
        return None
    
    # Filtrar apenas 2012 e 2024
    anos_desejados = [2012, 2024]
    dfs = []
    for csv_file in sorted(csv_files):
        basename = os.path.basename(csv_file)
        try:
            partes = basename.split('_')
            if len(partes) >= 2:
                ano_str = partes[1][2:6]
                ano = int(ano_str)
                if ano in anos_desejados:
                    print(f"  Carregando: {basename}...")
                    df_temp = pd.read_csv(csv_file, sep=';')
                    dfs.append(df_temp)
        except Exception as e:
            print(f"  Aviso ao carregar {basename}: {e}")
    
    if not dfs:
        print("Erro: Nenhum arquivo foi carregado com sucesso.")
        return None
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"[OK] Dados consolidados: {len(df)} registros totais")

    # Garante que a coluna 'eh_ti' seja booleana / numérica
    if 'eh_ti' in df.columns and df['eh_ti'].dtype == 'object':
        df['eh_ti'] = df['eh_ti'].str.upper().map({'TRUE': True, 'FALSE': False, 'SIM': True, 'NAO': False})
    df['eh_ti'] = pd.to_numeric(df.get('eh_ti', 0).astype(object), errors='coerce').fillna(0).astype(int)

    # Converte colunas de renda e peso para numérico, tratando erros.
    df['rendimento_bruto_mensal'] = pd.to_numeric(df.get('rendimento_bruto_mensal', 0), errors='coerce')
    df['rendimento_trabalho_principal'] = pd.to_numeric(df.get('rendimento_trabalho_principal', 0), errors='coerce')
    df['peso_populacional'] = pd.to_numeric(df.get('peso_populacional', 0), errors='coerce').fillna(0)
    # coluna ocupado pode não existir em todos os datasets; cria com 0/1 quando ausente
    if 'ocupado' not in df.columns:
        df['ocupado'] = 0
    df['ocupado'] = pd.to_numeric(df['ocupado'], errors='coerce').fillna(0).astype(int)

    return df

def gerar_grafico_renda_vs_estudo(df, ano, segmento):
    """
    Filtra dados para um ano e segmento (TI ou Não-TI), processa as médias ponderadas e a população
    adotando a mesma regra de ponderação (peso_populacional e população ocupada) usada nos outros scripts.
    Também salva um CSV com a tabela usada no gráfico para validação.
    """
    if segmento == 'TI':
        df_segmento = df[df['eh_ti'] == 1].copy()
        titulo_segmento = "Profissionais de TI"
        nome_arquivo_segmento = "TI"
    else:
        df_segmento = df[df['eh_ti'] == 0].copy()
        titulo_segmento = "Demais Profissionais (Não-TI)"
        nome_arquivo_segmento = "Nao_TI"

    print(f"\n--- Processando: Ano {ano} | Segmento: {titulo_segmento} ---")

    # 1. Filtrar dados para o ano
    df_ano = df_segmento[df_segmento['ano'] == ano].copy()

    # garantir que colunas existam e sejam numéricas antes de criar flags
    df_ano['ocupado'] = pd.to_numeric(df_ano.get('ocupado', 0), errors='coerce').fillna(0).astype(int)
    df_ano['rendimento_trabalho_principal'] = pd.to_numeric(
        df_ano.get('rendimento_trabalho_principal', 0), errors='coerce'
    ).fillna(0.0)

    # criar a flag eh_ocupado antes de qualquer uso (ocupado==1 ou rendimento_principal>0)
    df_ano['eh_ocupado'] = ((df_ano['ocupado'] == 1) | (df_ano['rendimento_trabalho_principal'] > 0)).astype(int)

    # garantir anos_estudo numérico e remover registros com 0 anos antes da agregação
    anos_numeric = pd.to_numeric(df_ano.get('anos_estudo'), errors='coerce')
    zeros = int((anos_numeric == 0).sum())
    if zeros > 0:
        print(f"Removendo {zeros} registros com 0 anos de estudo para o segmento '{titulo_segmento}' ({ano}).")
    # manter apenas registros com >= 1 anos de estudo
    df_ano = df_ano[anos_numeric >= 1].copy()
    if df_ano.empty:
        print(f"Após remover 0 anos, não há dados para {titulo_segmento} em {ano}.")
        return

    # criar coluna numérica consistente para agrupar
    df_ano['_anos_estudo_num'] = pd.to_numeric(df_ano['anos_estudo'], errors='coerce').fillna(-1).astype(int)
    df_ano = df_ano[df_ano['_anos_estudo_num'] >= 0].copy()

    # --- DIAGNÓSTICO RÁPIDO (executar antes das remoções/agregações) ---
    anos_numeric = pd.to_numeric(df_ano.get('anos_estudo'), errors='coerce')
    print("anos_estudo min/max:", anos_numeric.min(), anos_numeric.max())
    print("anos_estudo valores únicos (amostra):", sorted(pd.Series(anos_numeric.dropna().unique()).astype(int).tolist()))
    n_ge17 = int((anos_numeric >= 17).sum())
    print(f"Registros com anos_estudo >= 17: {n_ge17}")

    # checar anos == 1 sem rendimento_trabalho_principal (NaN)
    n_ano1_sem_renda = int(df_ano[anos_numeric == 1]['rendimento_trabalho_principal'].isna().sum())
    print(f"Registros com anos_estudo == 1 sem rendimento_trabalho_principal: {n_ano1_sem_renda}")

    # validar ponderação usada: total de pesos e média ponderada (exemplo TI / segmento atual)
    sample_seg = df_ano.copy()
    total_peso_all = sample_seg['peso_populacional'].sum()
    print(f"Peso populacional total (antes filtros): {total_peso_all}")

    # média ponderada usando somente registros 'ocupados' e com rendimento não-nulo (mesma regra do gráfico)
    filtro_ocup = ((sample_seg.get('ocupado', 0) == 1) | (sample_seg['rendimento_trabalho_principal'] > 0))
    rp = sample_seg.loc[filtro_ocup & sample_seg['peso_populacional'].gt(0)].dropna(subset=['rendimento_trabalho_principal'])
    peso_usado = rp['peso_populacional'].sum()
    if peso_usado > 0:
        media_ponderada = np.average(rp['rendimento_trabalho_principal'].astype(float), weights=rp['peso_populacional'])
    else:
        media_ponderada = np.nan
    print(f"Peso usado (ocupados c/ renda): {peso_usado}, média ponderada (rendimento_trabalho_principal): {media_ponderada}")

    # mostrar amostra de registros com anos >= 17 e exemplo de anos==1 sem renda (até 20 linhas)
    if n_ge17 > 0:
        print("Amostra de registros com anos_estudo >= 17:")
        print(sample_seg[anos_numeric >= 17].head(20).to_string(index=False))
    if n_ano1_sem_renda > 0:
        print("Amostra de registros com anos_estudo == 1 e sem rendimento_trabalho_principal:")
        print(sample_seg[anos_numeric == 1].head(20).to_string(index=False))

    # 2. Agrupar por 'anos_estudo' e calcular as médias ponderadas e a população (aplicando filtro de ocupação)
    rows = []
    for nivel, grp in df_ano.groupby('_anos_estudo_num'):
        # populaçao estimada: soma do peso_populacional apenas dos ocupados
        pop_est = grp.loc[grp['eh_ocupado'] == 1, 'peso_populacional'].sum()

        # Usar somente rendimento_trabalho_principal.
        # Considerar apenas registros ocupados na média ponderada (mesma regra usada para população)
        rp = grp.loc[grp['eh_ocupado'] == 1].dropna(subset=['rendimento_trabalho_principal'])
        # evitar pesos negativos/zeros
        rp = rp[rp['peso_populacional'] > 0]
        if not rp.empty and rp['peso_populacional'].sum() > 0:
            # já garantimos que rendimento_trabalho_principal foi convertido para numérico em carregar_dados()
            renda_principal_media = np.average(rp['rendimento_trabalho_principal'].astype(float), weights=rp['peso_populacional'])
        else:
            renda_principal_media = np.nan

        rows.append({
            'anos_estudo': int(nivel),
            'populacao_estimada': pop_est,
            'renda_principal_media': renda_principal_media
        })

    dados_agrupados = pd.DataFrame(rows).sort_values('anos_estudo').reset_index(drop=True)

    # Salvar CSV com a tabela usada para validação (especialmente útil para checar CBOS/outliers)
    # CSV desativado: geração de arquivos CSV removida conforme solicitação.
    """
    try:
        out_csv = f'graficos/renda_vs_estudo_{nome_arquivo_segmento}_{ano}.csv'
        dados_agrupados.to_csv(out_csv, index=False, sep=';', encoding='utf-8')
        print(f"Tabela de dados (por anos_estudo) salva em: {out_csv}")
    except Exception as e:
        print("Aviso: não foi possível salvar CSV de validação:", e)
    """

    if dados_agrupados.empty:
        print("Tabela agregada vazia — pulando gráfico.")
        return

    # 3. Gerar o Gráfico
    fig, ax1 = plt.subplots(figsize=(18, 10))

    # Uso da paleta verde consistente (apenas a renda principal agora)
    cor_principal = '#005A32'      # verde escuro

    # Para garantir alinhamento usamos posições X numéricas (mesmas para barras e linha)
    anos = dados_agrupados['anos_estudo'].astype(int).tolist()
    xpos = np.arange(len(anos))
    renda_vals = dados_agrupados['renda_principal_media'].astype(float).tolist()
    pop_vals = dados_agrupados['populacao_estimada'].astype(float).tolist()

    # Eixo primário (ax1) para a linha de Renda Principal (mesmas posições xpos)
    ax1.plot(xpos, renda_vals, color=cor_principal, marker='o', label='Renda Principal Mensal Média', linewidth=1.8)
    ax1.set_xlabel('Anos de Estudo Concluídos', fontsize=14)
    ax1.set_ylabel('Renda Média Mensal', fontsize=14)
    ax1.yaxis.set_major_formatter(formatter_reais)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.tick_params(axis='x', labelsize=12, rotation=0)

    # Eixo secundário (ax2) para as barras de População
    ax2 = ax1.twinx()
    ax2.bar(xpos, pop_vals, color='gray', alpha=0.3, width=0.6)
    ax2.set_ylabel('População Estimada', fontsize=14, color='gray')
    ax2.yaxis.set_major_formatter(formatter_km)
    ax2.tick_params(axis='y', colors='gray', labelsize=12)
    ax2.grid(False)

    # Ajustes finais do Gráfico
    plt.title(f'Renda Média Mensal vs. Anos de Estudo - {titulo_segmento} ({ano})', fontsize=18, pad=20)
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5)

    # Garante que o eixo X tenha todos os valores inteiros no intervalo
    # definir ticks nas posições usadas e rótulos com os anos reais
    ax1.set_xticks(xpos)
    ax1.set_xticklabels(anos, rotation=0)

    # Legenda: colocar abaixo do gráfico (fora da área dos eixos)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    fig.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.5, -0.12),
               ncol=3, fontsize=12)

    # Remove legenda interna do ax1 (para não duplicar) — somente se existir
    leg = ax1.get_legend()
    if leg is not None:
        leg.remove()

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])

    # Salvar o gráfico com nome diferenciado
    caminho_salvar = f'graficos/renda_vs_estudo_{nome_arquivo_segmento}_{ano}.png'
    plt.savefig(caminho_salvar, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico salvo em: {caminho_salvar}")

def gerar_grafico_renda_por_idade_comparativo(df, ano):
    """
    Gera dois subplots lado-a-lado (TI à esquerda, Não-TI à direita) com:
    - linha da renda principal média (ponderada por peso_populacional, apenas ocupados)
    - barras da população estimada por faixa
    Salva ambos em um único arquivo PNG.
    """
    df_year = df[df['ano'] == ano].copy()
    if df_year.empty:
        print(f"Sem dados para {ano}")
        return

    # garantir tipos
    df_year['peso_populacional'] = pd.to_numeric(df_year.get('peso_populacional', 0), errors='coerce').fillna(0)
    df_year['rendimento_trabalho_principal'] = pd.to_numeric(df_year.get('rendimento_trabalho_principal', 0), errors='coerce')
    df_year['ocupado'] = pd.to_numeric(df_year.get('ocupado', 0), errors='coerce').fillna(0).astype(int)
    df_year['eh_ti'] = pd.to_numeric(df_year.get('eh_ti', 0), errors='coerce').fillna(0).astype(int)
    df_year['idade'] = pd.to_numeric(df_year.get('idade'), errors='coerce')

    # flag ocupação
    df_year['eh_ocupado'] = ((df_year['ocupado'] == 1) | (df_year['rendimento_trabalho_principal'] > 0)).astype(int)

    # definir faixas de idade
    bins = [0, 24, 34, 44, 54, 64, 120]
    labels = ['<=24', '25-34', '35-44', '45-54', '55-64', '65+']
    df_year['faixa_idade'] = pd.cut(df_year['idade'], bins=bins, labels=labels, right=True)

    def agg_por_faixa(df_sub):
        rows = []
        for faixa in labels:
            grp = df_sub[df_sub['faixa_idade'] == faixa]
            pop_est = grp.loc[grp['eh_ocupado'] == 1, 'peso_populacional'].sum()
            rp = grp.loc[grp['eh_ocupado'] == 1].dropna(subset=['rendimento_trabalho_principal'])
            rp = rp[rp['peso_populacional'] > 0]
            if not rp.empty and rp['peso_populacional'].sum() > 0:
                renda_media = np.average(rp['rendimento_trabalho_principal'].astype(float), weights=rp['peso_populacional'])
            else:
                renda_media = np.nan
            rows.append({'faixa': faixa, 'pop_est': pop_est, 'renda_media': renda_media})
        return pd.DataFrame(rows).set_index('faixa')

    df_ti = df_year[df_year['eh_ti'] == 1]
    df_nti = df_year[df_year['eh_ti'] == 0]

    agg_ti = agg_por_faixa(df_ti)
    agg_nti = agg_por_faixa(df_nti)

    # plot side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    xpos = np.arange(len(labels))

    cor_ti = '#005A32'
    cor_nti = '#A4D3B3'

    # TI (left)
    ax = axs[0]
    ax2 = ax.twinx()
    renda_ti = agg_ti['renda_media'].replace(0, np.nan).astype(float).values
    pop_ti = agg_ti['pop_est'].astype(float).values

    ax.plot(xpos, renda_ti, color=cor_ti, marker='o', label='Renda Média TI', linewidth=2)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_title(f'Profissionais de TI ({ano})')
    ax.set_xlabel('Faixa de Idade')
    ax.set_ylabel('R$ Renda Média Mensal')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'R$ {int(x):,}'.replace(',', '.')))

    ax2.bar(xpos, pop_ti, color='lightgray', alpha=0.5, width=0.6, label='População Estimada (TI)')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x):,}'.replace(',', '.')))
    ax2.set_ylabel('População Estimada', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')

    # Não-TI (right)
    ax = axs[1]
    ax2 = ax.twinx()
    renda_nti = agg_nti['renda_media'].replace(0, np.nan).astype(float).values
    pop_nti = agg_nti['pop_est'].astype(float).values

    ax.plot(xpos, renda_nti, color=cor_nti, marker='o', label='Renda Média Não-TI', linewidth=2)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_title(f'Demais Profissionais (Não-TI) ({ano})')
    ax.set_xlabel('Faixa de Idade')
    # eixo y da renda já compartilhado com o 1º subplot

    ax2.bar(xpos, pop_nti, color='lightgray', alpha=0.5, width=0.6, label='População Estimada (Não-TI)')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x):,}'.replace(',', '.')))
    ax2.tick_params(axis='y', labelcolor='gray')

    # montar legenda única abaixo dos plots
    handles = []
    labels_leg = []
    # coletar legendas manualmente (se existem)
    for a in axs:
        h, l = a.get_legend_handles_labels()
        handles += h
        labels_leg += l
        h2, l2 = a.twinx().get_legend_handles_labels()
        handles += h2
        labels_leg += l2
    # remover duplicatas mantendo ordem
    seen = set()
    uniq = []
    uniq_labels = []
    for hh, ll in zip(handles, labels_leg):
        if ll not in seen:
            uniq.append(hh); uniq_labels.append(ll); seen.add(ll)

    fig.legend(uniq, uniq_labels, loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=12)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    caminho = f'graficos/renda_por_idade_TI_vs_NTI_{ano}.png'
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráficos (TI e Não-TI) salvos em: {caminho}")

def _aggregate_por_estudo(df_segmento, ano):
    """Retorna DataFrame com anos_estudo, populacao_estimada e renda_principal_media (ponderada)."""
    df_ano = df_segmento[df_segmento['ano'] == ano].copy()
    if df_ano.empty:
        return pd.DataFrame(columns=['anos_estudo','populacao_estimada','renda_principal_media'])
    df_ano['ocupado'] = pd.to_numeric(df_ano.get('ocupado', 0), errors='coerce').fillna(0).astype(int)
    df_ano['rendimento_trabalho_principal'] = pd.to_numeric(df_ano.get('rendimento_trabalho_principal', 0), errors='coerce')
    df_ano['eh_ocupado'] = ((df_ano['ocupado'] == 1) | (df_ano['rendimento_trabalho_principal'] > 0)).astype(int)
    df_ano['_anos_estudo_num'] = pd.to_numeric(df_ano.get('anos_estudo'), errors='coerce').fillna(-1).astype(int)
    df_ano = df_ano[df_ano['_anos_estudo_num'] >= 1].copy()  # removemos 0 anos conforme solicitado

    rows = []
    for nivel, grp in df_ano.groupby('_anos_estudo_num'):
        pop_est = grp.loc[grp['eh_ocupado'] == 1, 'peso_populacional'].sum()
        rp = grp.loc[grp['eh_ocupado'] == 1].dropna(subset=['rendimento_trabalho_principal'])
        rp = rp[rp['peso_populacional'] > 0]
        if not rp.empty and rp['peso_populacional'].sum() > 0:
            renda_principal_media = np.average(rp['rendimento_trabalho_principal'].astype(float), weights=rp['peso_populacional'])
        else:
            renda_principal_media = np.nan
        rows.append({'anos_estudo': int(nivel), 'populacao_estimada': pop_est, 'renda_principal_media': renda_principal_media})
    return pd.DataFrame(rows).sort_values('anos_estudo').reset_index(drop=True)

def _aggregate_por_idade(df_segmento, ano):
    """Retorna DataFrame com faixas de idade, populacao_estimada e renda_principal_media (ponderada)."""
    df_ano = df_segmento[df_segmento['ano'] == ano].copy()
    if df_ano.empty:
        return pd.DataFrame(columns=['faixa','pop_est','renda_media'])
    df_ano['peso_populacional'] = pd.to_numeric(df_ano.get('peso_populacional', 0), errors='coerce').fillna(0)
    df_ano['rendimento_trabalho_principal'] = pd.to_numeric(df_ano.get('rendimento_trabalho_principal', 0), errors='coerce')
    df_ano['ocupado'] = pd.to_numeric(df_ano.get('ocupado', 0), errors='coerce').fillna(0).astype(int)
    df_ano['eh_ocupado'] = ((df_ano['ocupado'] == 1) | (df_ano['rendimento_trabalho_principal'] > 0)).astype(int)
    df_ano['idade'] = pd.to_numeric(df_ano.get('idade'), errors='coerce')

    bins = [0, 24, 34, 44, 54, 64, 120]
    labels = ['<=24', '25-34', '35-44', '45-54', '55-64', '65+']
    df_ano['faixa_idade'] = pd.cut(df_ano['idade'], bins=bins, labels=labels, right=True)

    rows = []
    for faixa in labels:
        grp = df_ano[df_ano['faixa_idade'] == faixa]
        pop_est = grp.loc[grp['eh_ocupado'] == 1, 'peso_populacional'].sum()
        rp = grp.loc[grp['eh_ocupado'] == 1].dropna(subset=['rendimento_trabalho_principal'])
        rp = rp[rp['peso_populacional'] > 0]
        if not rp.empty and rp['peso_populacional'].sum() > 0:
            renda_media = np.average(rp['rendimento_trabalho_principal'].astype(float), weights=rp['peso_populacional'])
        else:
            renda_media = np.nan
        rows.append({'faixa': faixa, 'pop_est': pop_est, 'renda_media': renda_media})
    return pd.DataFrame(rows).set_index('faixa')

def gerar_2x2_por_ano(df, ano):
    """
    Cria um único arquivo PNG com 2x2 subplots:
      linha 1 = TI (esq: renda vs estudo, dir: renda vs idade)
      linha 2 = Não-TI (esq: renda vs estudo, dir: renda vs idade)
    """
    segs = [('TI', 1), ('Nao_TI', 0)]
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), sharey='row')
    for row_idx, (nome, flag) in enumerate(segs):
        df_seg = df[df['eh_ti'] == flag].copy()
        # esquerda: estudo
        dados_estudo = _aggregate_por_estudo(df_seg, ano)
        ax = axes[row_idx, 0]
        if not dados_estudo.empty:
            xpos = np.arange(len(dados_estudo))
            ax.plot(xpos, dados_estudo['renda_principal_media'].astype(float).values, color='#005A32', marker='o', label='Renda Média')
            ax.bar(xpos, dados_estudo['populacao_estimada'].astype(float).values, alpha=0.25, color='gray', width=0.6)
            ax.set_xticks(xpos); ax.set_xticklabels(dados_estudo['anos_estudo'].astype(int).tolist())
        ax.set_title(f"{'Profissionais de TI' if flag==1 else 'Demais Profissionais (Não-TI)'} - Renda vs Anos de Estudo ({ano})")
        ax.set_xlabel('Anos de Estudo')
        ax.set_ylabel('R$ Renda Média Mensal')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'R$ {int(x):,}'.replace(',', '.')))

        # direita: idade
        dados_idade = _aggregate_por_idade(df_seg, ano)
        ax2 = axes[row_idx, 1]
        labels = dados_idade.index.tolist()
        xpos2 = np.arange(len(labels))
        if not dados_idade.empty:
            ax2.plot(xpos2, dados_idade['renda_media'].astype(float).values, color='#005A32' if flag==1 else '#A4D3B3', marker='o', label='Renda Média')
            ax2.bar(xpos2, dados_idade['pop_est'].astype(float).values, alpha=0.25, color='gray', width=0.6)
            ax2.set_xticks(xpos2); ax2.set_xticklabels(labels)
        ax2.set_title(f"{'Profissionais de TI' if flag==1 else 'Demais Profissionais (Não-TI)'} - Renda vs Idade ({ano})")
        ax2.set_xlabel('Faixa de Idade')
        # eixo y da renda já compartilhado com o 1º subplot

    # legenda única e salvar
    fig.legend(['Renda Média','População Estimada'], loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=2)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    caminho = f'graficos/renda_2x2_TI_NTI_{ano}.png'
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Arquivo 2x2 salvo em: {caminho}")

def gerar_comparativo_unico_por_ano(df, ano):
    """
    Gera um único PNG com dois subplots lado-a-lado:
      - esquerda: Renda (linhas) e População (barras) por Anos de Estudo (TI vs Não-TI)
      - direita: Renda (linhas) e População (barras) por Faixa de Idade (TI vs Não-TI)
    Barra: duas barras por categoria (TI / Não-TI). Linhas: renda TI / renda Não-TI.
    """
    # preparações e tipos
    df_year = df[df['ano'] == ano].copy()
    if df_year.empty:
        print(f"Sem dados para {ano}")
        return

    df_year['peso_populacional'] = pd.to_numeric(df_year.get('peso_populacional', 0), errors='coerce').fillna(0)
    df_year['rendimento_trabalho_principal'] = pd.to_numeric(df_year.get('rendimento_trabalho_principal', 0), errors='coerce')
    df_year['ocupado'] = pd.to_numeric(df_year.get('ocupado', 0), errors='coerce').fillna(0).astype(int)
    df_year['eh_ti'] = pd.to_numeric(df_year.get('eh_ti', 0), errors='coerce').fillna(0).astype(int)
    df_year['anos_estudo'] = pd.to_numeric(df_year.get('anos_estudo'), errors='coerce')
    df_year['idade'] = pd.to_numeric(df_year.get('idade'), errors='coerce')
    df_year['eh_ocupado'] = ((df_year['ocupado'] == 1) | (df_year['rendimento_trabalho_principal'] > 0)).astype(int)

    # faixas de idade
    bins = [0, 24, 34, 44, 54, 64, 120]
    labels = ['<=24', '25-34', '35-44', '45-54', '55-64', '65+']
    df_year['faixa_idade'] = pd.cut(df_year['idade'], bins=bins, labels=labels, right=True)

    # helper: agrega por anos_estudo (removendo 0 anos)
    def agreg_estudo(df_sub):
        sub = df_sub[df_sub['anos_estudo'].ge(1)].copy()
        grupos = sorted(sub['anos_estudo'].dropna().unique().astype(int).tolist())
        rows = []
        for ano_est in grupos:
            g = sub[sub['anos_estudo'].astype(int) == int(ano_est)]
            pop = g.loc[g['eh_ocupado'] == 1, 'peso_populacional'].sum()
            rp = g.loc[g['eh_ocupado'] == 1].dropna(subset=['rendimento_trabalho_principal'])
            rp = rp[rp['peso_populacional'] > 0]
            renda = np.nan
            if not rp.empty and rp['peso_populacional'].sum() > 0:
                renda = np.average(rp['rendimento_trabalho_principal'].astype(float), weights=rp['peso_populacional'])
            rows.append({'anos_estudo': int(ano_est), 'pop_est': pop, 'renda_media': renda})
        return pd.DataFrame(rows).sort_values('anos_estudo').reset_index(drop=True)

    # helper: agrega por faixa de idade
    def agreg_idade(df_sub):
        rows = []
        for faixa in labels:
            g = df_sub[df_sub['faixa_idade'] == faixa]
            pop = g.loc[g['eh_ocupado'] == 1, 'peso_populacional'].sum()
            rp = g.loc[g['eh_ocupado'] == 1].dropna(subset=['rendimento_trabalho_principal'])
            rp = rp[rp['peso_populacional'] > 0]
            renda = np.nan
            if not rp.empty and rp['peso_populacional'].sum() > 0:
                renda = np.average(rp['rendimento_trabalho_principal'].astype(float), weights=rp['peso_populacional'])
            rows.append({'faixa': faixa, 'pop_est': pop, 'renda_media': renda})
        return pd.DataFrame(rows)

    # agrega para TI e Não-TI
    df_ti = df_year[df_year['eh_ti'] == 1]
    df_nti = df_year[df_year['eh_ti'] == 0]

    est_ti = agreg_estudo(df_ti)
    est_nti = agreg_estudo(df_nti)

    id_ti = agreg_idade(df_ti)
    id_nti = agreg_idade(df_nti)

    # Normalizar nomes de colunas das agregações (compatibilizar versões diferentes)
    def _normalize_est(df_est):
        if df_est is None or df_est.empty:
            return df_est
        df_est = df_est.copy()
        # populaçao
        if 'pop_est' not in df_est.columns:
            if 'populacao_estimada' in df_est.columns:
                df_est = df_est.rename(columns={'populacao_estimada': 'pop_est'})
            elif 'pop' in df_est.columns:
                df_est = df_est.rename(columns={'pop': 'pop_est'})
            else:
                df_est['pop_est'] = 0
        # renda
        if 'renda_principal_media' not in df_est.columns:
            if 'renda_media' in df_est.columns:
                df_est = df_est.rename(columns={'renda_media': 'renda_principal_media'})
            elif 'renda' in df_est.columns:
                df_est = df_est.rename(columns={'renda': 'renda_principal_media'})
            else:
                df_est['renda_principal_media'] = np.nan
        # garantir colunas esperadas
        if 'anos_estudo' not in df_est.columns and df_est.index.name == 'anos_estudo':
            df_est = df_est.reset_index()
        return df_est

    est_ti = _normalize_est(est_ti)
    est_nti = _normalize_est(est_nti)
    # id_* normalmente já tem 'pop_est' e 'renda_media' — garantir tipos numéricos
    if id_ti is not None and not id_ti.empty:
        id_ti['pop_est'] = pd.to_numeric(id_ti.get('pop_est', 0), errors='coerce').fillna(0)
        if 'renda_media' not in id_ti.columns and 'renda_principal_media' in id_ti.columns:
            id_ti = id_ti.rename(columns={'renda_principal_media': 'renda_media'})
    if id_nti is not None and not id_nti.empty:
        id_nti['pop_est'] = pd.to_numeric(id_nti.get('pop_est', 0), errors='coerce').fillna(0)
        if 'renda_media' not in id_nti.columns and 'renda_principal_media' in id_nti.columns:
            id_nti = id_nti.rename(columns={'renda_principal_media': 'renda_media'})

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=False)

    # cores (barras visíveis em verde, linhas em padrão)
    bar_ti_color = '#37a546'   # verde vivo para TI (barras)
    bar_nti_color = '#9bd6b6'  # verde claro para Não-TI (barras)
    line_ti_color = '#005A32'  # linha TI (escuro)
    line_nti_color = '#2E7D52'  # cor Não-TI um pouco mais escura para melhor contraste

    # --- subplot esquerdo: anos de estudo ---
    ax = axes[0]
    if not est_ti.empty or not est_nti.empty:
        # unify index (anos presentes em qualquer segmento)
        anos_all = sorted(set(est_ti['anos_estudo'].tolist()) | set(est_nti['anos_estudo'].tolist()))
        xpos = np.arange(len(anos_all))
        width = 0.35
        # map values
        map_est_ti = est_ti.set_index('anos_estudo')
        map_est_nti = est_nti.set_index('anos_estudo')
        pop_ti = [map_est_ti.loc[a,'pop_est'] if a in map_est_ti.index else 0 for a in anos_all]
        pop_nti = [map_est_nti.loc[a,'pop_est'] if a in map_est_nti.index else 0 for a in anos_all]
        renda_ti = [map_est_ti.loc[a,'renda_media'] if a in map_est_ti.index else np.nan for a in anos_all]
        renda_nti = [map_est_nti.loc[a,'renda_media'] if a in map_est_nti.index else np.nan for a in anos_all]

        # --- preencher/interpolar renda ausente e suavizar curvas ---
        renda_ti_s = pd.Series(renda_ti, index=anos_all).astype(float)
        renda_nti_s = pd.Series(renda_nti, index=anos_all).astype(float)
        # interpola linear onde faltam pontos (limit_direction both para bordas)
        renda_ti_f = renda_ti_s.interpolate(method='linear', limit_direction='both')
        renda_nti_f = renda_nti_s.interpolate(method='linear', limit_direction='both')
        # suaviza com média móvel centrada (window=3) mantendo bordas
        renda_ti_smooth = renda_ti_f.rolling(window=3, center=True, min_periods=1).mean()
        renda_nti_smooth = renda_nti_f.rolling(window=3, center=True, min_periods=1).mean()

        # --- normalizar populações para visualização (0..1) e anotar máximo absoluto ---
        max_pop = max(max(pop_ti) if pop_ti else 0, max(pop_nti) if pop_nti else 0)
        if max_pop <= 0:
            max_pop = 1.0
        pop_ti_norm = [p / max_pop for p in pop_ti]
        pop_nti_norm = [p / max_pop for p in pop_nti]

        # barras agrupadas (normalizadas)
        ax.bar(xpos - width/2, pop_ti_norm, width=width, color=bar_ti_color, alpha=0.9, label='Pop. TI (norm.)')
        ax.bar(xpos + width/2, pop_nti_norm, width=width, color=bar_nti_color, alpha=0.9, label='Pop. Não-TI (norm.)')

        # linhas de renda (eixo y secundário, reais) usando séries suavizadas/interpoladas
        ax2 = ax.twinx()
        ax2.plot(xpos, renda_ti_smooth.values, color=line_ti_color, marker='o', label='Renda TI', linewidth=2)
        ax2.plot(xpos, renda_nti_smooth.values, color=line_nti_color, marker='o', label='Renda Não-TI', linewidth=2)

        ax.set_xticks(xpos)
        ax.set_xticklabels(anos_all)
        ax.set_xlabel('Anos de Estudo')
        ax.set_title(f'População (norm.) e Renda por Anos de Estudo - {ano}')
        ax.set_ylabel('População Estimada (normalizada)')
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x*100)}%'))
        # mostrar o máximo absoluto como referência
        ax.text(0.99, 0.02, f'Max pop: {int(max_pop):,}', transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color='gray')

        ax2.set_ylabel('R$ Renda Média Mensal')
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'R$ {int(x):,}'.replace(',', '.')))

        # legenda combinada
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        handles = h1 + h2
        labels = l1 + l2
        ax2.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.01, 0.95))

    # --- subplot direito: idade ---
    ax = axes[1]
    if not id_ti.empty or not id_nti.empty:
        labels_id = id_ti['faixa'].tolist()  # same order used in agreg_idade
        xpos2 = np.arange(len(labels_id))
        pop_ti_id = id_ti['pop_est'].tolist()
        pop_nti_id = id_nti['pop_est'].tolist()
        renda_ti_id = id_ti['renda_media'].tolist()
        renda_nti_id = id_nti['renda_media'].tolist()

        # --- interpolar/suavizar renda por idade (caso faltem valores) ---
        renda_ti_id_s = pd.Series(renda_ti_id, index=labels_id).astype(float)
        renda_nti_id_s = pd.Series(renda_nti_id, index=labels_id).astype(float)
        renda_ti_id_f = renda_ti_id_s.interpolate(method='linear', limit_direction='both')
        renda_nti_id_f = renda_nti_id_s.interpolate(method='linear', limit_direction='both')
        renda_ti_id_smooth = renda_ti_id_f.rolling(window=3, center=True, min_periods=1).mean()
        renda_nti_id_smooth = renda_nti_id_f.rolling(window=3, center=True, min_periods=1).mean()

        # normalizar populações por faixa (para melhorar visual)
        max_pop_id = max(max(pop_ti_id) if pop_ti_id else 0, max(pop_nti_id) if pop_nti_id else 0)
        if max_pop_id <= 0:
            max_pop_id = 1.0
        pop_ti_id_norm = [p / max_pop_id for p in pop_ti_id]
        pop_nti_id_norm = [p / max_pop_id for p in pop_nti_id]

        ax.bar(xpos2 - width/2, pop_ti_id_norm, width=width, color=bar_ti_color, alpha=0.9, label='Pop. TI (norm.)')
        ax.bar(xpos2 + width/2, pop_nti_id_norm, width=width, color=bar_nti_color, alpha=0.9, label='Pop. Não-TI (norm.)')

        ax2 = ax.twinx()
        ax2.plot(xpos2, renda_ti_id_smooth.values, color=line_ti_color, marker='o', label='Renda TI', linewidth=2)
        ax2.plot(xpos2, renda_nti_id_smooth.values, color=line_nti_color, marker='o', label='Renda Não-TI', linewidth=2)

        ax.set_xticks(xpos2)
        ax.set_xticklabels(labels_id)
        ax.set_xlabel('Faixa de Idade')
        ax.set_title(f'População (norm.) e Renda por Faixa de Idade - {ano}')
        ax.set_ylabel('População Estimada (normalizada)')
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x*100)}%'))
        ax.text(0.99, 0.02, f'Max pop: {int(max_pop_id):,}', transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color='gray')

        ax2.set_ylabel('R$ Renda Média Mensal')
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'R$ {int(x):,}'.replace(',', '.')))

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        handles = h1 + h2
        labels = l1 + l2
        ax2.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.01, 0.95))

    plt.tight_layout()
    caminho = f'graficos/comparativo_unico_TI_NTI_{ano}.png'
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparativo único (estudo + idade) salvo em: {caminho}")

def gerar_4plots_por_ano(df, ano):
    """
    Para o ano fornecido gera 1 arquivo PNG com 4 subplots (2x2):
      [0,0] Renda vs Anos de Estudo - TI
      [0,1] Renda vs Anos de Estudo - Não-TI
      [1,0] Renda vs Faixa de Idade - TI
      [1,1] Renda vs Faixa de Idade - Não-TI

    - barras mostram a população estimada (valor absoluto, sem normalização)
    - linhas mostram renda principal média (ponderada por peso_populacional)
    - linhas são interpoladas e suavizadas para garantir pontos conectados
    """
    df_year = df[df['ano'] == ano].copy()
    if df_year.empty:
        print(f"Sem dados para {ano}")
        return

    # helpers existentes: _aggregate_por_estudo e _aggregate_por_idade
    df_ti = df_year[df_year['eh_ti'] == 1]
    df_nti = df_year[df_year['eh_ti'] == 0]

    est_ti = _aggregate_por_estudo(df_ti, ano)
    est_nti = _aggregate_por_estudo(df_nti, ano)
    id_ti = _aggregate_por_idade(df_ti, ano)
    id_nti = _aggregate_por_idade(df_nti, ano)

    # Normalizar nomes de colunas das agregações (compatibilizar versões diferentes)
    def _normalize_est(df_est):
        if df_est is None or df_est.empty:
            return df_est
        df_est = df_est.copy()
        # populaçao
        if 'pop_est' not in df_est.columns:
            if 'populacao_estimada' in df_est.columns:
                df_est = df_est.rename(columns={'populacao_estimada': 'pop_est'})
            elif 'pop' in df_est.columns:
                df_est = df_est.rename(columns={'pop': 'pop_est'})
            else:
                df_est['pop_est'] = 0
        # renda
        if 'renda_principal_media' not in df_est.columns:
            if 'renda_media' in df_est.columns:
                df_est = df_est.rename(columns={'renda_media': 'renda_principal_media'})
            elif 'renda' in df_est.columns:
                df_est = df_est.rename(columns={'renda': 'renda_principal_media'})
            else:
                df_est['renda_principal_media'] = np.nan
        # garantir colunas esperadas
        if 'anos_estudo' not in df_est.columns and df_est.index.name == 'anos_estudo':
            df_est = df_est.reset_index()
        return df_est

    est_ti = _normalize_est(est_ti)
    est_nti = _normalize_est(est_nti)
    # id_* normalmente já tem 'pop_est' e 'renda_media' — garantir tipos numéricos
    if id_ti is not None and not id_ti.empty:
        id_ti['pop_est'] = pd.to_numeric(id_ti.get('pop_est', 0), errors='coerce').fillna(0)
        if 'renda_media' not in id_ti.columns and 'renda_principal_media' in id_ti.columns:
            id_ti = id_ti.rename(columns={'renda_principal_media': 'renda_media'})
    if id_nti is not None and not id_nti.empty:
        id_nti['pop_est'] = pd.to_numeric(id_nti.get('pop_est', 0), errors='coerce').fillna(0)
        if 'renda_media' not in id_nti.columns and 'renda_principal_media' in id_nti.columns:
            id_nti = id_nti.rename(columns={'renda_principal_media': 'renda_media'})

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    line_ti_color = '#005A32'
    line_nti_color = '#2E7D52'  # cor Não-TI um pouco mais escura para melhor contraste
    bar_ti_color = '#37a546'
    bar_nti_color = '#9bd6b6'

    # util: interp + smooth (series index numeric)
    def interp_smooth(x_index, y_vals):
        s = pd.Series(y_vals, index=x_index).astype(float)
        s = s.interpolate(method='linear', limit_direction='both')
        s = s.rolling(window=3, center=True, min_periods=1).mean()
        return s

    # Plot 0,0: Renda vs Anos de Estudo - TI
    ax = axes[0,0]
    if not est_ti.empty:
        anos_present = est_ti['anos_estudo'].astype(int).tolist()
        anos_all = list(range(min(anos_present), max(anos_present)+1))
        pop_map = est_ti.set_index('anos_estudo')['pop_est'].to_dict()
        renda_map = est_ti.set_index('anos_estudo')['renda_principal_media'].to_dict()
        pop_vals = [pop_map.get(a, 0) for a in anos_all]
        renda_vals = [renda_map.get(a, np.nan) for a in anos_all]
        # barra (população absoluta)
        ax.bar(anos_all, pop_vals, color=bar_ti_color, alpha=0.8, width=0.7, label='Pop. TI')
        ax.set_xlabel('Anos de Estudo')
        ax.set_ylabel('População Estimada', color='black')
        ax.tick_params(axis='y', labelcolor='black')
        ax.yaxis.set_major_formatter(formatter_km)

        # renda eixo secundário
        ax2 = ax.twinx()
        renda_s = interp_smooth(anos_all, renda_vals)
        ax2.plot(anos_all, renda_s.values, color=line_ti_color, marker='o', label='Renda TI', linewidth=2)
        ax2.set_ylabel('R$ Renda Média Mensal')
        ax2.yaxis.set_major_formatter(formatter_reais)

        ax.set_title(f'Renda vs Anos de Estudo - TI ({ano})')
        # legendas
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, loc='upper left')
    else:
        ax.text(0.5,0.5,"Sem dados TI", ha='center', va='center')
        ax.set_axis_off()

    # Plot 0,1: Renda vs Anos de Estudo - Não-TI
    ax = axes[0,1]
    if not est_nti.empty:
        anos_present = est_nti['anos_estudo'].astype(int).tolist()
        anos_all = list(range(min(anos_present), max(anos_present)+1))
        pop_map = est_nti.set_index('anos_estudo')['pop_est'].to_dict()
        renda_map = est_nti.set_index('anos_estudo')['renda_principal_media'].to_dict()
        pop_vals = [pop_map.get(a, 0) for a in anos_all]
        renda_vals = [renda_map.get(a, np.nan) for a in anos_all]
        ax.bar(anos_all, pop_vals, color=bar_nti_color, alpha=0.8, width=0.7, label='Pop. Não-TI')
        ax.set_xlabel('Anos de Estudo')
        ax.set_ylabel('População Estimada', color='black')
        ax.tick_params(axis='y', labelcolor='black')
        ax.yaxis.set_major_formatter(formatter_km)
        ax2 = ax.twinx()
        renda_s = interp_smooth(anos_all, renda_vals)
        ax2.plot(anos_all, renda_s.values, color=line_nti_color, marker='o', label='Renda Não-TI', linewidth=2)
        ax2.set_ylabel('R$ Renda Média Mensal')
        ax2.yaxis.set_major_formatter(formatter_reais)

        ax.set_title(f'Renda vs Anos de Estudo - Não-TI ({ano})')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, loc='upper left')
    else:
        ax.text(0.5,0.5,"Sem dados Não-TI", ha='center', va='center')
        ax.set_axis_off()

    # Plot 1,0: Renda vs Faixa de Idade - TI
    ax = axes[1,0]
    if not id_ti.empty:
        labels_id = id_ti.index.tolist()
        xpos = np.arange(len(labels_id))
        pop_vals = id_ti['pop_est'].astype(float).tolist()
        renda_vals = id_ti['renda_media'].astype(float).tolist()
        ax.bar(xpos, pop_vals, color=bar_ti_color, alpha=0.8, width=0.6, label='Pop. TI')
        ax.set_xticks(xpos); ax.set_xticklabels(labels_id)
        ax.set_xlabel('Faixa de Idade')
        ax.set_ylabel('População Estimada', color='black')
        ax.tick_params(axis='y', labelcolor='black')
        ax.yaxis.set_major_formatter(formatter_km)
        ax2 = ax.twinx()
        renda_s = interp_smooth(xpos, renda_vals)
        ax2.plot(xpos, renda_s.values, color=line_ti_color, marker='o', label='Renda TI', linewidth=2)
        ax2.set_ylabel('R$ Renda Média Mensal')
        ax2.yaxis.set_major_formatter(formatter_reais)

        ax.set_title(f'Renda vs Faixa de Idade - TI ({ano})')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, loc='upper left')
    else:
        ax.text(0.5,0.5,"Sem dados TI", ha='center', va='center')
        ax.set_axis_off()

    # Plot 1,1: Renda vs Faixa de Idade - Não-TI
    ax = axes[1,1]
    if not id_nti.empty:
        labels_id = id_nti.index.tolist()
        xpos = np.arange(len(labels_id))
        pop_vals = id_nti['pop_est'].astype(float).tolist()
        renda_vals = id_nti['renda_media'].astype(float).tolist()
        ax.bar(xpos, pop_vals, color=bar_nti_color, alpha=0.8, width=0.6, label='Pop. Não-TI')
        ax.set_xticks(xpos); ax.set_xticklabels(labels_id)
        ax.set_xlabel('Faixa de Idade')
        ax.set_ylabel('População Estimada', color='black')
        ax.tick_params(axis='y', labelcolor='black')
        ax.yaxis.set_major_formatter(formatter_km)
        ax2 = ax.twinx()
        renda_s = interp_smooth(xpos, renda_vals)
        ax2.plot(xpos, renda_s.values, color=line_nti_color, marker='o', label='Renda Não-TI', linewidth=2)
        ax2.set_ylabel('R$ Renda Média Mensal')
        ax2.yaxis.set_major_formatter(formatter_reais)

        ax.set_title(f'Renda vs Faixa de Idade - Não-TI ({ano})')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax2.legend(h1 + h2, l1 + l2, loc='upper left')
    else:
        ax.text(0.5,0.5,"Sem dados Não-TI", ha='center', va='center')
        ax.set_axis_off()

    plt.tight_layout()
    caminho = f'graficos/comparativo_4plots_{ano}.png'
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Comparativo 4-plots salvo em: {caminho}")

def main():
    """Função principal para orquestrar a análise e visualização."""
    configurar_ambiente()
    df = carregar_dados()
    if df is None:
        return
    anos_analise = [2012, 2024]
    anos_disponiveis = sorted(df['ano'].unique())
    print(f"\nAnos disponíveis no dataset: {anos_disponiveis}")
    for ano in anos_analise:
        if ano in anos_disponiveis:
            gerar_4plots_por_ano(df, ano)
        else:
            print(f"Ano {ano} não disponível — pulando.")

    print("\nProcessamento concluído.")

if __name__ == "__main__":
    main()