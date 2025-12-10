import pandas as pd
from pathlib import Path

# --- DEFINIÇÕES DO PROJETO ---
CBO_FAMILIA_TI = [
    '1330', '2434', '2511', '2512', '2513', '2514', '2519', '2521', 
    '2522', '2523', '2529', '3511', '3512', '3513', '3514', '3522'
]
MAPEAMENTO_SEXO = {1: 'Homens', 2: 'Mulheres'}

def gerar_sumario_populacional(df_ti: pd.DataFrame, ano: int):
    """
    Calcula e exibe a população estimada de profissionais de TI
    por código de estudo e sexo para um ano específico, usando pesos populacionais.
    """
    print(f"\n--- Análise de População Estimada para o Ano: {ano} ---")
    
    df_ano = df_ti[df_ti['ano'] == ano].copy()

    if df_ano.empty:
        print(f"Nenhum dado de TI encontrado para o ano {ano}.")
        return

    coluna_estudo = 'curso_mais_elevado_antes_2015' if ano < 2015 else 'curso_mais_elevado'
    
    df_ano['sexo_desc'] = df_ano['sexo'].map(MAPEAMENTO_SEXO)

    df_analise = df_ano.dropna(subset=[coluna_estudo, 'sexo_desc', 'peso_populacional']).copy()
    
    # --- AGREGAÇÃO COM PESO POPULACIONAL ---
    # A coluna 'peso_populacional' já foi tratada (convertida para inteiro) na função main.
    estimativa = df_analise.groupby([coluna_estudo, 'sexo_desc'])['peso_populacional'].sum()
    
    sumario_pivot = estimativa.unstack().fillna(0).astype(int)

    if sumario_pivot.empty:
        print("Não foram encontrados registros de TI com dados de escolaridade para este ano.")
    else:
        print(f"População estimada por código de estudo (coluna '{coluna_estudo}') e sexo:")
        print(sumario_pivot.to_string(float_format='{:,.0f}'.format))


def main():
    """
    Função principal para carregar os dados e gerar a análise populacional.
    """
    caminho_do_arquivo = Path(r"C:\TCC\dados\pnad\dados_pnad_consolidados.csv")
    print("--- INICIANDO SCRIPT DE ANÁLISE POPULACIONAL (COM PESOS) ---")

    if not caminho_do_arquivo.exists():
        print(f"❌ ERRO: Arquivo de dados não encontrado em: {caminho_do_arquivo}")
        return

    try:
        print(f"📄 Carregando dados de: {caminho_do_arquivo}")
        df = pd.read_csv(caminho_do_arquivo, sep=';', low_memory=False)
        print(f"✅ Dados carregados. Total de {len(df)} registros.")
    except Exception as e:
        print(f"❌ ERRO ao ler o arquivo CSV: {e}")
        return

    # --- TRATAMENTO DO PESO POPULACIONAL (LÓGICA CORRIGIDA) ---
    print("Limpando e convertendo a coluna 'peso_populacional'...")
    # 1. Converte a coluna para um tipo numérico, tratando erros.
    #    Isso lida tanto com números inteiros quanto com decimais (ex: 150.00155468)
    df['peso_populacional'] = pd.to_numeric(df['peso_populacional'], errors='coerce')
    
    # 2. Remove linhas onde a conversão falhou (resultou em Nulo/NaN)
    df.dropna(subset=['peso_populacional'], inplace=True)
    
    # 3. Converte para inteiro. Esta ação TRUNCA (remove) a parte decimal.
    #    Ex: 150.00155468 se torna 150.
    df['peso_populacional'] = df['peso_populacional'].astype(int)
    print("✅ 'peso_populacional' convertido para números inteiros.")

    # 1. CLASSIFICAÇÃO TI
    df['cbo_ocupacao'] = df['cbo_ocupacao'].astype(str).fillna('0000')
    df['cbo_familia'] = df['cbo_ocupacao'].str[:4]
    df['eh_ti'] = df['cbo_familia'].isin(CBO_FAMILIA_TI)

    # 2. FILTRAR APENAS PROFISSIONAIS DE TI
    df_ti = df[df['eh_ti']].copy()

    # 3. GERAR SUMÁRIOS PARA OS ANOS DE INTERESSE
    gerar_sumario_populacional(df_ti, 2012)
    gerar_sumario_populacional(df_ti, 2024)
    
    print("\n--- Processo concluído ---")

# --- EXECUÇÃO DO SCRIPT ---
if __name__ == '__main__':
    main()