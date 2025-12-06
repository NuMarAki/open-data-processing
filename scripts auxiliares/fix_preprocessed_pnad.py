#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fix_preprocessed_pnad.py

Correções mínimas e seguras no CSV preprocessado da PNAD para validar CBOs:
 - Preserva/normaliza cbo_raw sem coercionar para float (evita '3221.0' -> NaN)
 - Cria coluna cbo_raw_str (texto) e cbo_norm (4 dígitos ou None)
 - Faz merge V3009/V3009A em coluna v3009_merged (prefere V3009A se preenchido)
 - Normaliza peso_populacional com heurística BR/EN
 - Preenche carteira_assinada vazia/0 -> 2 (não)
 - Gera coluna employed a partir de forca_trabalho / ocupado
 - Processa em chunks para arquivos grandes

Uso:
 python scripts/auxiliares/fix_preprocessed_pnad.py --csv C:/TCC/dados/pnad/preprocessados/PNADC_032019_preprocessado.csv --out C:/TCC/dados/pnad/preprocessados/PNADC_032019_preprocessado_fixed.csv --chunksize 200000

Saída:
 - CSV corrigido salvo em --out (mantém colunas originais e adiciona colunas novas)
 - Resumo impresso ao final (contagens de cbo válidos, etc.)

Observação:
 - Este script NÃO altera os arquivos TXT originais. É uma correção conservadora do CSV
   preprocessado (onde vimos perda/coerção de CBOs). Rode primeiro num subset/
   em modo --chunksize pequeno para validação.
"""
from __future__ import annotations
import argparse
import os
import re
import math
import csv
from typing import Optional

import pandas as pd

def try_parse_weight(s: Optional[str]) -> float:
    if s is None:
        return float("nan")
    s = str(s).strip()
    if s == "" or s.lower() in ("nan","none"):
        return float("nan")
    # direct
    try:
        return float(s)
    except Exception:
        pass
    # Brazilian style: 1.234.567,89 -> 1234567.89
    try:
        t = s.replace(".", "").replace(",", ".")
        return float(t)
    except Exception:
        pass
    # fallback removing commas
    try:
        t = s.replace(",", "")
        return float(t)
    except Exception:
        pass
    return float("nan")

def norm_cbo_str(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    st = str(s).strip()
    if st == "" or st.lower() in ("nan","none"):
        return None
    # if looks like numeric with .0 (e.g., '3221.0'), remove decimal part
    if re.fullmatch(r'\d+\.\d+', st):
        # take integer part
        try:
            st = str(int(float(st)))
        except Exception:
            st = re.sub(r'\.0+$','', st)
    # extract digits
    digits = re.sub(r'[^0-9]', '', st)
    if digits == "" or digits == "0000":
        return None
    digits = digits.zfill(4)[-4:]
    if re.fullmatch(r'\d{4}', digits):
        return digits
    return None

def choose_v3009(a, b):
    # prefer V3009A (b) if present and non-empty; otherwise V3009 (a)
    for v in (b, a):
        if v is None:
            continue
        s = str(v).strip()
        if s == "" or s.lower() in ("nan","none"):
            continue
        return s
    return None

def normalize_carteira(val):
    if val is None:
        return 2
    s = str(val).strip()
    if s == "" or s in ("0","0.0","nan","None"):
        return 2
    try:
        iv = int(float(s))
        if iv == 0:
            return 2
        return iv
    except Exception:
        return s

def derive_employed(ft, oc):
    # treat '1' as yes; otherwise False
    try:
        if ft is not None and str(ft).strip() != "" and int(float(str(ft))) == 1:
            return True
    except Exception:
        pass
    try:
        if oc is not None and str(oc).strip() != "" and int(float(str(oc))) == 1:
            return True
    except Exception:
        pass
    return False

def process_chunk(df: pd.DataFrame):
    # work on string-preserving dataframe (we read with dtype=str)
    # create cbo_raw_str preferring existing cbo_raw, then cbo_ocupacao, then cbo_familia
    prefer_cols = ['cbo_raw','cbo_ocupacao','cbo_familia']
    def pick_raw(row):
        for c in prefer_cols:
            if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
                return row[c]
        return None

    df['cbo_raw_str'] = df.apply(pick_raw, axis=1)
    df['cbo_norm'] = df['cbo_raw_str'].apply(norm_cbo_str)

    # merge V3009/V3009A if columns present
    v3009_cols = ('V3009','V3009A')
    if any(c in df.columns for c in v3009_cols):
        a = df['V3009'] if 'V3009' in df.columns else (df['v3009'] if 'v3009' in df.columns else None)
        b = df['V3009A'] if 'V3009A' in df.columns else (df['v3009a'] if 'v3009a' in df.columns else None)
        df['v3009_merged'] = [choose_v3009(a_i, b_i) for a_i, b_i in zip(a if a is not None else [None]*len(df), b if b is not None else [None]*len(df))]
    else:
        df['v3009_merged'] = None

    # parse weight
    if 'peso_populacional' in df.columns:
        df['peso_parsed'] = df['peso_populacional'].apply(try_parse_weight)
    else:
        df['peso_parsed'] = float('nan')

    # carteira_assinada
    if 'carteira_assinada' in df.columns:
        df['carteira_assinada_norm'] = df['carteira_assinada'].apply(normalize_carteira)
    else:
        # try common alternative names
        alt = None
        for altname in ('V4029','v4029','carteira'):
            if altname in df.columns:
                alt = altname; break
        if alt:
            df['carteira_assinada_norm'] = df[alt].apply(normalize_carteira)
        else:
            df['carteira_assinada_norm'] = 2

    # derive employed
    ft_col = None
    oc_col = None
    for name in ('forca_trabalho','VD4001','vd4001'):
        if name in df.columns:
            ft_col = name; break
    for name in ('ocupado','VD4002','vd4002'):
        if name in df.columns:
            oc_col = name; break
    df['employed_derived'] = df.apply(lambda r: derive_employed(r.get(ft_col) if ft_col else None, r.get(oc_col) if oc_col else None), axis=1)

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="CSV preprocessado de entrada")
    parser.add_argument("--out", required=True, help="CSV corrigido de saída")
    parser.add_argument("--chunksize", type=int, default=200000, help="Número de linhas por chunk (memória)")
    args = parser.parse_args()

    csv_in = args.csv
    csv_out = args.out
    chunksize = args.chunksize

    if not os.path.exists(csv_in):
        raise SystemExit(f"Arquivo não encontrado: {csv_in}")

    # We'll read as strings to avoid coercion
    reader = pd.read_csv(csv_in, sep=';', encoding='utf-8-sig', dtype=str, low_memory=True, chunksize=chunksize)

    header_written = False
    total = 0
    stats = {
        "rows": 0,
        "cbo_norm_nonnull": 0,
        "cbo_raw_nonnull": 0,
        "v3009_merged_nonnull": 0,
        "employed_true": 0
    }

    for chunk in reader:
        processed = process_chunk(chunk)
        # update stats
        stats['rows'] += len(processed)
        stats['cbo_norm_nonnull'] += processed['cbo_norm'].notna().sum()
        stats['cbo_raw_nonnull'] += processed['cbo_raw_str'].notna().sum()
        stats['v3009_merged_nonnull'] += processed['v3009_merged'].notna().sum()
        stats['employed_true'] += processed['employed_derived'].sum()

        # write with ';' and utf-8-sig
        if not header_written:
            processed.to_csv(csv_out, sep=';', index=False, encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL)
            header_written = True
        else:
            processed.to_csv(csv_out, sep=';', index=False, encoding='utf-8-sig', header=False, mode='a', quoting=csv.QUOTE_MINIMAL)

    # print summary
    print("Processamento concluído.")
    print(f"Arquivo de saída: {csv_out}")
    print("Resumo:")
    print(f"  Linhas processadas: {stats['rows']:,}")
    print(f"  cbo_raw presentes (texto): {stats['cbo_raw_nonnull']:,}")
    print(f"  cbo_norm (4 dígitos) válidos: {stats['cbo_norm_nonnull']:,}")
    print(f"  v3009 merged não-nulos: {stats['v3009_merged_nonnull']:,}")
    print(f"  employed_derived True: {stats['employed_true']:,}")
    print("\nPróximo passo sugerido (apenas se stats mostra perda de cbo):")
    print(" - Se houver muitos registros com txt_cbo válido mas cbo_norm null no CSV, aplicar preservação")
    print("   similar diretamente no pipeline de criação do CSV (garantir leitura como str e não converter para float).")

if __name__ == "__main__":
    main()