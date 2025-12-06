#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_txt_csv_cbo_by_keys.py

Compara valores de CBO entre o TXT posicional e o CSV preprocessado, alinhando por chaves
(ano,trimestre,uf,idade,grupo_amostra,tipo_area) em vez de por índice de linha.

Objetivo: verificar se os '0000' no CSV já existem no TXT (problema de fonte)
ou se foram introduzidos pelo processamento (problema de leitura/renomeação/conversão).

Uso:
 python scripts\\auxiliares\\compare_txt_csv_cbo_by_keys.py --txt "Z:/TCC/IBGE/PNADC_032019.txt" --csv "C:/TCC/dados/pnad/preprocessados/PNADC_032019_preprocessado.csv" --positions 152,155 --n 200 --out compare_output.txt

Parâmetros principais:
 --txt       : arquivo TXT posicional original (ou só um deles para testar)
 --csv       : CSV preprocessado correspondente (pode ser consolidação parcial)
 --positions : posição do CBO no TXT (START,END) 1-based inclusive (default 152,155)
 --n         : número de primeiras linhas do TXT a verificar (default 200)
 --out       : arquivo de saída (texto) com o resumo (default compare_output.txt)

Observações:
 - Key match usa ano,trimestre,uf,idade,grupo_amostra,tipo_area (ajuste no código se quiser incluir mais chaves).
 - O script tenta ler CSV com vários encodings.
 - Peso populacional é apenas normalizado para inspeção (vários attempts de parse).
"""
from __future__ import annotations
import argparse
import os
import re
import sys
import json
from collections import defaultdict
from typing import Tuple, List, Dict, Any

import pandas as pd
import numpy as np

# Layout positions (1-based inclusive) — use as no seu cfg (ajuste se necessário)
LAYOUT_POS = {
    "ano": (1, 4),
    "trimestre": (5, 5),
    "uf": (6, 7),
    "grupo_amostra": (30, 31),
    "tipo_area": (34, 34),
    "peso_populacional": (50, 64),
    "V3009A": (125, 126),
    "V3009": (125, 126),   # fallback / mesma posição histórica
    "cbo": (152, 155),
    "carteira_assinada": (195, 195),
    "idade": (104, 106),
    "forca_trabalho": (409, 409),
    "ocupado": (410, 410),
}

def extract(line: str, start:int, end:int) -> str:
    return line[start-1:end]

def try_parse_weight(s: str) -> float:
    """Tenta parsear peso populacional com heurísticas BR/EN."""
    if s is None:
        return float("nan")
    s = str(s).strip()
    if s == "":
        return float("nan")
    # remove thousand separators heuristically
    # Try direct parse
    try:
        return float(s)
    except Exception:
        pass
    # Try Brazilian style: '.' thousand, ',' decimal -> remove dots, replace comma
    try:
        t = s.replace(".", "").replace(",", ".")
        return float(t)
    except Exception:
        pass
    # Try swap (some file may use comma thousand)
    try:
        t = s.replace(",", "")
        return float(t)
    except Exception:
        pass
    return float("nan")

def norm_cbo(s: Any) -> str | None:
    if pd.isna(s):
        return None
    st = str(s).strip()
    st = re.sub(r'[^0-9]', '', st)
    if st == "" or st == "0000":
        return None
    st = st.zfill(4)[:4]
    if re.fullmatch(r'\d{4}', st):
        return st
    return None

def read_csv_try(path: str) -> pd.DataFrame:
    encs = ["utf-8-sig","utf-8","latin1","iso-8859-1"]
    last = None
    for e in encs:
        try:
            df = pd.read_csv(path, sep=';', encoding=e, low_memory=False)
            print(f"Lido CSV com encoding {e}")
            return df
        except Exception as ex:
            last = ex
    raise RuntimeError(f"Não foi possível ler CSV: {last}")

def build_csv_index(df: pd.DataFrame) -> Dict[Tuple, List[int]]:
    """
    Cria índice por chaves (ano,trimestre,uf,idade,grupo_amostra,tipo_area) apontando para linhas do CSV.
    As chaves são strings sem zeros à esquerda para robustez.
    """
    idx = defaultdict(list)
    for i, row in df.iterrows():
        key = (
            to_key_str(row.get('ano')),
            to_key_str(row.get('trimestre')),
            to_key_str(row.get('uf')),
            to_key_str(row.get('idade')),
            to_key_str(row.get('grupo_amostra')),
            to_key_str(row.get('tipo_area')),
        )
        idx[key].append(i)
    return idx

def to_key_str(v) -> str:
    if pd.isna(v):
        return ""
    try:
        # if numeric like 2012.0 -> integer
        if float(v).is_integer():
            return str(int(float(v)))
    except Exception:
        pass
    return str(v).strip()

def csv_row_summary(row: pd.Series) -> Dict[str,Any]:
    # prepare merged V3009 / V3009A logic
    v3009a = row.get('V3009A') if 'V3009A' in row.index else row.get('v3009a')
    v3009 = row.get('V3009') if 'V3009' in row.index else row.get('v3009')
    # prefer V3009A if present and non-null
    v3009_val = v3009a if (not pd.isna(v3009a) and str(v3009a).strip()!='') else v3009
    # carteira_assinada mapping: '' or 0 => 2 (não)
    carteira = row.get('carteira_assinada') if 'carteira_assinada' in row.index else row.get('V4029') if 'V4029' in row.index else row.get('v4029')
    if pd.isna(carteira) or str(carteira).strip() in ["", "0", "0.0"]:
        carteira = 2
    # employed flag from forca_trabalho / ocupado
    ft = row.get('forca_trabalho') if 'forca_trabalho' in row.index else row.get('VD4001') if 'VD4001' in row.index else row.get('vd4001')
    oc = row.get('ocupado') if 'ocupado' in row.index else row.get('VD4002') if 'VD4002' in row.index else row.get('vd4002')
    try:
        employed = (int(ft)==1) or (int(oc)==1)
    except Exception:
        # fallback: if both equal and equal to 1
        try:
            employed = (str(ft).strip() == str(oc).strip() == '1')
        except Exception:
            employed = False
    return {
        "cbo_raw": row.get('cbo_raw') if 'cbo_raw' in row.index else row.get('cbo_ocupacao') if 'cbo_ocupacao' in row.index else None,
        "cbo_norm": norm_cbo(row.get('cbo_raw') if 'cbo_raw' in row.index else row.get('cbo_ocupacao') if 'cbo_ocupacao' in row.index else None),
        "v3009": v3009_val,
        "carteira_assinada": carteira,
        "forca_trabalho": ft,
        "ocupado": oc,
        "employed": employed,
        "peso_parsed": try_parse_weight(row.get('peso_populacional')) if 'peso_populacional' in row.index else None
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--txt", required=True, help="TXT posicional (original)")
    p.add_argument("--csv", required=True, help="CSV preprocessado correspondente")
    p.add_argument("--positions", default="152,155", help="posições START,END do CBO no TXT (1-based inclusive)")
    p.add_argument("--n", type=int, default=200, help="n primeiras linhas do TXT a verificar")
    p.add_argument("--out", default="compare_output.txt", help="arquivo de saída resumo")
    args = p.parse_args()

    # sanity
    if not os.path.exists(args.txt):
        print("TXT não encontrado:", args.txt); sys.exit(1)
    if not os.path.exists(args.csv):
        print("CSV não encontrado:", args.csv); sys.exit(1)

    start,end = map(int, args.positions.split(","))

    # read CSV and build index
    df = read_csv_try(args.csv)
    csv_index = build_csv_index(df)

    # read first N lines of TXT and process
    results = []
    summary = {"lines_checked": 0, "matches_by_key_found": 0, "txt_has_value_csv_empty":0, "csv_has_value_txt_empty":0, "exact_match_num":0}
    with open(args.txt, 'r', encoding='latin1', errors='ignore') as f:
        for i, line in enumerate(f, start=1):
            if i > args.n:
                break
            summary["lines_checked"] += 1
            # extract key values from TXT using LAYOUT_POS
            txt_keys = {}
            for k in ("ano","trimestre","uf","idade","grupo_amostra","tipo_area"):
                s,e = LAYOUT_POS[k]
                txt_keys[k] = extract(line, s, e).strip()
            cbo_txt_raw = extract(line, start, end).strip()
            cbo_txt_norm = norm_cbo(cbo_txt_raw)
            # build key tuple compatible with csv_index
            key = (txt_keys['ano'].lstrip("0"), txt_keys['trimestre'].lstrip("0"), txt_keys['uf'].lstrip("0"),
                   txt_keys['idade'].lstrip("0"), txt_keys['grupo_amostra'].lstrip("0"), txt_keys['tipo_area'].lstrip("0"))
            # lookup csv rows
            matched_rows = csv_index.get(key, [])
            entry = {
                "line_no_txt": i,
                "txt_cbo_raw": cbo_txt_raw,
                "txt_cbo_norm": cbo_txt_norm,
                "key": txt_keys,
                "csv_matches": []
            }
            if matched_rows:
                summary["matches_by_key_found"] += 1
                for ridx in matched_rows:
                    crow = df.loc[ridx]
                    csum = csv_row_summary(crow)
                    entry["csv_matches"].append({
                        "csv_index": int(ridx),
                        "csv_cbo_raw": csum["cbo_raw"],
                        "csv_cbo_norm": csum["cbo_norm"],
                        "v3009": csum["v3009"],
                        "carteira_assinada": csum["carteira_assinada"],
                        "employed": csum["employed"],
                        "peso_parsed": csum["peso_parsed"]
                    })
                    # compare
                    if cbo_txt_norm and csum["cbo_norm"]:
                        if cbo_txt_norm == csum["cbo_norm"]:
                            summary["exact_match_num"] += 1
                    if cbo_txt_norm and not csum["cbo_norm"]:
                        summary["txt_has_value_csv_empty"] += 1
                    if (not cbo_txt_norm) and csum["cbo_norm"]:
                        summary["csv_has_value_txt_empty"] += 1
            else:
                # no CSV match for this key
                entry["csv_matches"] = []
            results.append(entry)

    # write summary
    with open(args.out, "w", encoding="utf-8") as fo:
        fo.write("Comparison TXT vs CSV by keys\n")
        fo.write(f"TXT file: {args.txt}\nCSV file: {args.csv}\npositions: {start}-{end}\n")
        fo.write(f"Lines checked: {summary['lines_checked']}\n")
        fo.write(f"Lines with CSV matches by key: {summary['matches_by_key_found']}\n")
        fo.write(f"TXT has CBO but CSV empty (count): {summary['txt_has_value_csv_empty']}\n")
        fo.write(f"CSV has CBO but TXT empty (count): {summary['csv_has_value_txt_empty']}\n")
        fo.write(f"Exact numeric matches (normed 4-digit): {summary['exact_match_num']}\n\n")
        fo.write("Detailed entries (first 200):\n\n")
        for r in results:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Resumo salvo em", args.out)
    print("Linhas verificadas:", summary['lines_checked'])
    print("Matches found by key:", summary['matches_by_key_found'])
    print("TXT->CSV: txt value & csv empty:", summary['txt_has_value_csv_empty'])
    print("CSV->TXT: csv value & txt empty:", summary['csv_has_value_txt_empty'])
    print("Exact matches (4-digit normalized):", summary['exact_match_num'])

if __name__ == "__main__":
    main()