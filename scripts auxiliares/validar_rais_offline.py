#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Executa validação externa com RAIS a partir de um CSV de predições salvo,
sem necessidade de retreinar o modelo.
"""
import argparse
from pathlib import Path
import sys
import pandas as pd

# Inserir a raiz do repositório no sys.path para habilitar imports da raiz
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main():
    ap = argparse.ArgumentParser(description="Validação externa com RAIS a partir de predições salvas.")
    ap.add_argument("--pred", required=True, help=r"Caminho para o CSV de predições (ex.: Z:\...\rf_ocupado_predicoes_teste.csv)")
    ap.add_argument("--rais-dir", required=True, help=r"Diretório dos Parquets RAIS preprocessados (ex.: C:\TCC\dados\rais\preprocessados)")
    ap.add_argument("--out", required=True, help=r"Diretório de saída (ex.: Z:\...\resultados\ocupado)")
    args = ap.parse_args()

    pred_path = Path(args.pred); rais_dir = Path(args.rais_dir); out_dir = Path(args.out)
    if not pred_path.exists():
        print(f"[erro] Arquivo de predições não encontrado: {pred_path}"); sys.exit(1)
    if not rais_dir.exists():
        print(f"[erro] Diretório RAIS não encontrado: {rais_dir}"); sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from metrics_validacao_rais import validar_com_rais
    except ImportError as e:
        print(f"[erro] Não foi possível importar metrics_validacao_rais.validar_com_rais: {e}")
        print(f"       Repo root in sys.path? {REPO_ROOT}")
        sys.exit(1)

    print(f"[info] Lendo predições: {pred_path}")
    df_pred = pd.read_csv(pred_path, sep=';', dtype=str, low_memory=False, encoding='utf-8')

    print(f"[info] Rodando validação RAIS (diretório: {rais_dir})...")
    met = validar_com_rais(df_predicoes=df_pred, rais_dir=str(rais_dir), resultados_dir=out_dir)

    print("[ok] Validação concluída.")
    print("Resumo:")
    for k, v in met.items():
        print(f"  - {k}: {v}")

    print(f"Arquivos gerados em: {out_dir.resolve()}")
    print(f"  - validacao_rais_por_grupo.csv (se houve interseção)")
    print(f"  - validacao_rais_resumo.txt")

if __name__ == "__main__":
    main()