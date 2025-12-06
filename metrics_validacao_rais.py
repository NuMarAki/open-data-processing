# -*- coding: utf-8 -*-
"""
Validação externa com RAIS:
Compara a probabilidade prevista de 'ocupado' (PNAD) agregada por grupos
com a taxa de vínculo ativo 31/12 (RAIS), reutilizando o mesmo de-para e normalizações.

Saídas:
- validacao_rais_por_grupo.csv
- validacao_rais_resumo.txt
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from integracao_rais_pnad_features import (
    carregar_rais_agregado,
    _clean_uf_as_code,
    _derive_uf_from_municipio,
)

POSSIVEIS_CHAVES = ['ano','uf','sexo','nivel_instrucao','eh_ti']

def _clean_uf_as_sigla(s: pd.Series) -> pd.Series:
    """Converte série de UF para siglas padronizadas (SP)."""
    from integracao_rais_pnad_features import _SIGLAS_VALIDAS, _UF_IBGE_TO_SIGLA
    
    if s is None or len(s) == 0:
        return pd.Series([], dtype='object')
    
    raw = s.astype(str).str.strip().str.upper()
    
    # Já são siglas
    mask_sigla = raw.str.match(r'^[A-Z]{2}$', na=False)
    siglas = raw.where(~mask_sigla, raw.str.upper())
    
    # Códigos numéricos para siglas
    mask_num = ~mask_sigla & raw.str.match(r'^\d{1,2}(?:\.\d+)?$', na=False)
    if mask_num.any():
        nums = pd.to_numeric(raw[mask_num].str.split('.').str[0], errors='coerce')
        siglas.loc[mask_num] = nums.map(_UF_IBGE_TO_SIGLA)
    
    # Validar siglas
    siglas = siglas.where(siglas.isin(_SIGLAS_VALIDAS), pd.NA)
    return siglas

def _agrupar_predicoes(pred: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    g = pred.groupby(keys, dropna=False)
    out = g.agg(
        n_obs=('y_proba','size'),
        proba_pnad=('y_proba','mean')
    ).reset_index()
    return out

def _harmonizar_tipos(pred: pd.DataFrame, ref: pd.DataFrame, keys: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    p = pred.copy(); r = ref.copy()
    if 'ano' in keys:
        p['ano'] = _coerce_num(p.get('ano', np.nan)).astype('Int64')
        r['ano'] = _coerce_num(r.get('ano', np.nan)).astype('Int64')
    if 'uf' in keys:
        p['uf'] = _clean_uf_as_code(p.get('uf', np.nan))
        r['uf'] = _clean_uf_as_code(r.get('uf', np.nan))
    if 'sexo' in keys:
        p['sexo'] = _clean_sexo(p.get('sexo', np.nan))
        r['sexo'] = _clean_sexo(r.get('sexo', np.nan))
    if 'nivel_instrucao' in keys:
        p['nivel_instrucao'] = map_escolaridade_rais_to_pnad(p.get('nivel_instrucao', pd.Series([np.nan]*len(p)))).astype('object')
        r['nivel_instrucao'] = map_escolaridade_rais_to_pnad(r.get('nivel_instrucao', pd.Series([np.nan]*len(r)))).astype('object')
    if 'eh_ti' in keys:
        p['eh_ti'] = _coerce_num(p.get('eh_ti', np.nan)).astype('Int64')
        r['eh_ti'] = _coerce_num(r.get('eh_ti', np.nan)).astype('Int64')
    return p, r

def _merge_melhor_chave(pred: pd.DataFrame, aggs_rais: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    # Tenta chaves mais granulares primeiro
    for tabela in ('full','uf','ano'):
        ref = aggs_rais.get(tabela)
        if ref is None or ref.empty:
            continue
        keys = [k for k in POSSIVEIS_CHAVES if k in pred.columns and k in ref.columns]
        if not keys:
            continue
        pred_g = _agrupar_predicoes(pred, keys)
        pred_g2, ref2 = _harmonizar_tipos(pred_g, ref, keys)
        merged = pd.merge(pred_g2, ref2, on=keys, how='inner')
        if not merged.empty:
            return merged, keys
    # Fallback mínimo por ano
    if 'ano' in pred.columns and 'ano' in aggs_rais.get('ano', pd.DataFrame()).columns:
        pred_g = _agrupar_predicoes(pred, ['ano'])
        p2, r2 = _harmonizar_tipos(pred_g, aggs_rais['ano'], ['ano'])
        merged = pd.merge(p2, r2, on=['ano'], how='inner')
        if not merged.empty:
            return merged, ['ano']
    return pd.DataFrame(), []

def _normalize_uf_code(df: pd.DataFrame, col='uf') -> pd.DataFrame:
    out = df.copy()
    if col in out.columns:
        out[col] = _clean_uf_as_code(out[col])  # força código IBGE (Int64)
    return out

def validar_com_rais(df_predicoes: pd.DataFrame, rais_dir: str, resultados_dir: Path) -> Dict[str, float]:
    dfp = df_predicoes.copy()
    # Normalização mínima
    if 'ano' in dfp.columns: dfp['ano'] = _coerce_num(dfp['ano']).astype('Int64')
    if 'uf' in dfp.columns: dfp['uf'] = _clean_uf_as_sigla(dfp['uf'])
    if 'sexo' in dfp.columns: dfp['sexo'] = _clean_sexo(dfp['sexo'])
    if 'nivel_instrucao' in dfp.columns: dfp['nivel_instrucao'] = map_escolaridade_rais_to_pnad(dfp['nivel_instrucao']).astype('object')
    if 'eh_ti' in dfp.columns: dfp['eh_ti'] = _coerce_num(dfp['eh_ti']).astype('Int64')
    if 'y_proba' in dfp.columns: dfp['y_proba'] = _coerce_num(dfp['y_proba'])

    aggs = carregar_rais_agregado(rais_dir)
    merged, keys = _merge_melhor_chave(dfp, aggs)

    resultados_dir.mkdir(parents=True, exist_ok=True)

    if merged.empty:
        with open(resultados_dir / 'validacao_rais_resumo.txt', 'w', encoding='utf-8') as f:
            f.write("Sem interseção PNAD×RAIS após harmonização de chaves.\n")
        return {
            'grupos_comparados': 0,
            'corr_pearson': np.nan, 'corr_spearman': np.nan,
            'mae': np.nan, 'rmse': np.nan,
            'chaves_usadas': ''
        }

    x = _coerce_num(merged['proba_pnad'])
    y = _coerce_num(merged['taxa_ativo_3112_rais'])
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]

    if len(x) == 0:
        with open(resultados_dir / 'validacao_rais_resumo.txt', 'w', encoding='utf-8') as f:
            f.write("Sem valores válidos para comparar proba_pnad e taxa_ativo_3112_rais.\n")
        return {
            'grupos_comparados': 0,
            'corr_pearson': np.nan, 'corr_spearman': np.nan,
            'mae': np.nan, 'rmse': np.nan,
            'chaves_usadas': ','.join(keys)
        }

    corr_pearson = float(pd.Series(x).corr(pd.Series(y), method='pearson'))
    corr_spearman = float(pd.Series(x).corr(pd.Series(y), method='spearman'))
    diff = x - y
    mae = float(np.nanmean(np.abs(diff)))
    rmse = float(np.sqrt(np.nanmean(diff**2)))

    merged.loc[mask, 'diff'] = diff
    merged.loc[mask, 'abs_diff'] = np.abs(diff)
    merged.loc[mask, 'sq_diff'] = diff**2
    merged.to_csv(resultados_dir / 'validacao_rais_por_grupo.csv', index=False, sep=';')

    with open(resultados_dir / 'validacao_rais_resumo.txt', 'w', encoding='utf-8') as f:
        f.write(f"Chaves usadas: {', '.join(keys)}\n")
        f.write(f"Grupos comparados: {len(x)}\n")
        f.write(f"Correlação (Pearson): {corr_pearson:.4f}\n")
        f.write(f"Correlação (Spearman): {corr_spearman:.4f}\n")
        f.write(f"MAE: {mae:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")

    return {
        'grupos_comparados': len(x),
        'corr_pearson': corr_pearson, 'corr_spearman': corr_spearman,
        'mae': mae, 'rmse': rmse,
        'chaves_usadas': ','.join(keys)
    }