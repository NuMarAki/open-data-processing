import os
import glob
import pandas as pd

BASES = {
    'pnad': 'C:/TCC/dados/pnad/preprocessados',
    'rais': 'C:/TCC/dados/rais/preprocessados',
    'caged': 'C:/TCC/dados/caged/preprocessados'
}

COLUNAS_ESSENCIAIS = ['idade', 'cbo_ocupacao']

def diagnostico_base(nome, pasta):
    arquivos = glob.glob(os.path.join(pasta, '*.csv')) + glob.glob(os.path.join(pasta, '*.parquet'))
    if not arquivos:
        return f"{nome.upper()}: Nenhum arquivo preprocessado encontrado.\n"
    resumo = [f"{nome.upper()}:"]
    total_registros = 0
    anos = set()
    colunas = set()
    ti_count = 0
    outros_count = 0
    for arq in arquivos:
        try:
            if arq.endswith('.parquet'):
                df = pd.read_parquet(arq, engine='pyarrow')
            else:
                df = pd.read_csv(arq, sep=';', nrows=100_000)  # Amostra para não travar
            total_registros += len(df)
            colunas.update(df.columns)
            if 'ano' in df.columns:
                anos.update(df['ano'].dropna().unique())
            if 'eh_ti' in df.columns:
                ti_count += (df['eh_ti'] == True).sum()
                outros_count += (df['eh_ti'] == False).sum()
        except Exception as e:
            resumo.append(f"  Erro ao ler {arq}: {e}")
    resumo.append(f"  Total de registros (amostrados): {total_registros:,}")
    if anos:
        resumo.append(f"  Anos cobertos: {sorted([int(a) for a in anos if str(a).isdigit()])}")
    if colunas:
        resumo.append(f"  Colunas: {sorted(list(colunas))}")
    if ti_count + outros_count > 0:
        resumo.append(f"  Profissionais TI: {ti_count:,} | Outros: {outros_count:,}")

    try:
        if all(col in df.columns for col in ['peso_populacional', 'eh_ti', 'idade']):
            from stats_pnad import gerar_estatisticas_ponderadas
            est = gerar_estatisticas_ponderadas(
                df,
                peso_col='peso_populacional',
                col_eh_ti='eh_ti',
                col_idade='idade',
                salvar=False
            )
            resumo.append(f"  Estatísticas ponderadas: {est}")
    except Exception as e:
        resumo.append(f"  Erro estatísticas ponderadas: {e}")

    return '\n'.join(resumo) + '\n'

def validar_colunas_essenciais():
    print("\nValidação das colunas essenciais nos arquivos preprocessados:")
    for nome, pasta in BASES.items():
        arquivos = glob.glob(os.path.join(pasta, '*.csv')) + glob.glob(os.path.join(pasta, '*.parquet'))
        if not arquivos:
            print(f"{nome.upper()}: Nenhum arquivo preprocessado encontrado.")
            continue
        for arq in arquivos:
            try:
                if arq.endswith('.parquet'):
                    df = pd.read_parquet(arq, engine='pyarrow')
                else:
                    df = pd.read_csv(arq, sep=';', nrows=100)
                faltando = [col for col in COLUNAS_ESSENCIAIS if col not in df.columns]
                if faltando:
                    print(f"  {arq}: FALTANDO colunas {faltando}")
                else:
                    print(f"  {arq}: OK")
            except Exception as e:
                print(f"  {arq}: Erro ao ler ({e})")

def main():
    with open('diagnostico_dados.txt', 'w', encoding='utf-8') as f:
        for nome, pasta in BASES.items():
            f.write(diagnostico_base(nome, pasta))
            f.write('\n')
    print("Arquivo 'diagnostico_dados.txt' gerado com sucesso.")
    validar_colunas_essenciais()

if __name__ == '__main__':
    main()