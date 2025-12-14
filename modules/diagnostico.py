"""Diagnóstico rápido das bases preprocessadas"""
import os
import glob
from datetime import datetime
import pandas as pd


def _diagnosticar_base(base: str, resumo: list):
    """Coleta estatísticas rápidas de uma base preprocessada"""
    base_dir = os.path.join('dados', base, 'preprocessados')
    linha = {
        'base': base,
        'status': 'ok',
        'arquivos': 0,
        'amostras_lidas': 0,
        'colunas': '',
        'anos': '',
        'tem_eh_ti': 'nao',
        'erro': ''
    }
    if not os.path.exists(base_dir):
        linha['status'] = 'faltante'
        linha['erro'] = 'diretorio nao encontrado'
        resumo.append(linha)
        print(f"    [!] {base.upper()}: diretório {base_dir} não encontrado")
        return
    arquivos = glob.glob(os.path.join(base_dir, '*.csv'))
    linha['arquivos'] = len(arquivos)
    if not arquivos:
        linha['status'] = 'faltante'
        linha['erro'] = 'nenhum csv preprocessado'
        resumo.append(linha)
        print(f"    [!] {base.upper()}: nenhum csv preprocessado encontrado")
        return
    anos = set()
    colunas_total = set()
    tem_eh_ti = False
    amostras_lidas = 0
    for csv_file in arquivos[:10]:  # limita para manter rápido
        try:
            df = pd.read_csv(csv_file, sep=';', nrows=20000, low_memory=False)
            amostras_lidas += len(df)
            colunas_total.update(df.columns)
            if 'ano' in df.columns:
                anos.update([a for a in df['ano'].dropna().unique() if pd.notna(a)])
            if 'eh_ti' in df.columns:
                tem_eh_ti = True
        except Exception as e:
            linha['status'] = 'parcial'
            linha['erro'] = f"erro lendo {os.path.basename(csv_file)}: {e}"
            print(f"    [!] {base.upper()}: erro lendo {os.path.basename(csv_file)}: {e}")
            break
    linha['amostras_lidas'] = amostras_lidas
    if colunas_total:
        linha['colunas'] = ','.join(sorted(colunas_total)[:20])
    if anos:
        try:
            anos_int = sorted({int(float(a)) for a in anos})
            linha['anos'] = ','.join(map(str, anos_int))
        except Exception:
            pass
    linha['tem_eh_ti'] = 'sim' if tem_eh_ti else 'nao'
    resumo.append(linha)
    print(f"    [OK] {base.upper()}: {linha['arquivos']} arquivos, {amostras_lidas} linhas amostradas")


def executar_diagnostico():
    """Executa diagnóstico leve das bases preprocessadas"""
    print("\n[*] Executando diagnóstico dos dados...")
    resumo = []
    for base in ['pnad', 'rais', 'caged']:
        _diagnosticar_base(base, resumo)
    # Salvar relatório
    os.makedirs('relatorios', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    saida = os.path.join('relatorios', f'diagnostico_bases_{ts}.csv')
    pd.DataFrame(resumo).to_csv(saida, sep=';', index=False, encoding='utf-8-sig')
    print(f"\n[✓] Diagnóstico concluído! Relatório salvo em: {saida}")
    return saida
