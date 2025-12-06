# -*- coding: utf-8 -*-
r"""
Integração leve de features agregadas da RAIS para enriquecer o modelo da PNAD (alvo=ocupado).

- Lê Parquets de C:\TCC\dados\rais\preprocessados
- Deriva UF a partir do município (mun_trab/municipio) quando UF explícita não existir.
- Harmoniza escolaridade da RAIS para categorias PNAD/IBGE via de-para (1..11 ou rótulos).
- Agrega por chaves ['ano','uf','sexo','nivel_instrucao','eh_ti'] quando existirem e não forem totalmente NaN (com fallbacks).
- Faz merge no df da PNAD com fallbacks (ignorando colunas vazias) para máxima cobertura.
"""

from __future__ import annotations
import os
import re
import glob
from typing import List, Optional, Tuple, Dict, Set
import pandas as pd
import numpy as np
from typing import Optional

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None

_AGGS_CACHE: Dict[str, Dict[str, pd.DataFrame]] = {}

_UF_IBGE_TO_SIGLA = {
    '11':'RO','12':'AC','13':'AM','14':'RR','15':'PA','16':'AP','17':'TO',
    '21':'MA','22':'PI','23':'CE','24':'RN','25':'PB','26':'PE','27':'AL','28':'SE','29':'BA',
    '31':'MG','32':'ES','33':'RJ','35':'SP',
    '41':'PR','42':'SC','43':'RS',
    '50':'MS','51':'MT','52':'GO','53':'DF'
}
_SIGLAS_VALIDAS = set(_UF_IBGE_TO_SIGLA.values())
_SIGLA_TO_UF_CODE = {v: int(k) for k, v in _UF_IBGE_TO_SIGLA.items()}

def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = (
        pd.Series(df.columns)
        .astype(str).str.strip().str.lower()
        .str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        .str.replace(r'\s+', '_', regex=True).str.replace(r'[^a-z0-9_]', '', regex=True)
    )
    df.columns = cols
    return df

def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def _clean_uf_as_code(s: pd.Series) -> pd.Series:
    """Converte série com siglas (SP) ou códigos ('35'/'35.0'/35) para código UF IBGE (Int64)."""
    raw = s.astype(str).str.strip().replace({'nan': None})
    is_sigla = raw.str.match(r'^[A-Za-z]{2}$', na=False)
    out = pd.Series(pd.NA, index=raw.index, dtype='Int64')
    if is_sigla.any():
        out.loc[is_sigla] = pd.to_numeric(raw[is_sigla].str.upper().map(_SIGLA_TO_UF_CODE), errors='coerce').astype('Int64')
    rest = ~is_sigla
    if rest.any():
        out.loc[rest] = pd.to_numeric(raw[rest].str.replace(r'\.0$', '', regex=True), errors='coerce').astype('Int64')
    return out

def _derive_uf_from_municipio(s: pd.Series) -> pd.Series:
    """Extrai os 2 primeiros dígitos do código IBGE do município (mun_trab) como UF (Int64)."""
    raw = s.astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.replace(r'\D+', '', regex=True).str.zfill(6)
    return pd.to_numeric(raw.str[:2], errors='coerce').astype('Int64')

def _clean_sexo(s: pd.Series) -> pd.Series:
    t = s.astype(str).str.lower().str.strip()
    return t.replace({'1':'M','2':'F','masculino':'M','feminino':'F','m':'M','f':'F'})

def _coerce_num(s: pd.Series, to_int: bool = False) -> pd.Series:
    out = pd.to_numeric(s, errors='coerce')
    if to_int:
        try:
            return out.astype('Int64')
        except Exception:
            return out
    return out

def _safe_nanmean(s: pd.Series) -> float:
    arr = pd.to_numeric(s, errors='coerce')
    mask = np.isfinite(arr)
    return float(np.nanmean(arr[mask])) if mask.any() else float('nan')

def _safe_nanmedian(s: pd.Series) -> float:
    arr = pd.to_numeric(s, errors='coerce')
    mask = np.isfinite(arr)
    return float(np.nanmedian(arr[mask])) if mask.any() else float('nan')

def _normalize_str(x: str) -> str:
    import unicodedata, re as _re
    if x is None:
        return ""
    s = str(x).strip().upper()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _re.sub(r"[^A-Z0-9\s\.]", " ", s)
    s = _re.sub(r"\s+", " ", s).strip()
    return s

def map_escolaridade_rais_to_pnad(serie: pd.Series, granular: bool = False) -> pd.Series:
    s_raw = serie.copy()
    s_num = pd.to_numeric(s_raw, errors="coerce")
    if granular:
        num_map = {1:"sem_instrucao", 2:"fundamental_incompleto",3:"fundamental_incompleto",4:"fundamental_incompleto",
                   5:"fundamental_completo", 6:"medio_incompleto", 7:"medio_completo",
                   8:"superior_incompleto", 9:"superior_completo", 10:"pos_graduacao", 11:"pos_graduacao"}
    else:
        num_map = {1:"fundamental_incompleto", 2:"fundamental_incompleto",3:"fundamental_incompleto",4:"fundamental_incompleto",
                   5:"fundamental_completo", 6:"medio_incompleto", 7:"medio_completo",
                   8:"superior_incompleto", 9:"superior_completo", 10:"pos_graduacao", 11:"pos_graduacao"}
    out = pd.Series([None]*len(s_raw), dtype="object")
    mask_num = s_num.notna()
    out.loc[mask_num] = s_num[mask_num].map(num_map).astype("object")
    mask_str = ~mask_num
    if mask_str.any():
        s_txt = s_raw[mask_str].astype(str).map(_normalize_str)
        def map_text(v: str):
            if not v: return None
            if "ANALFABETO" in v: return "sem_instrucao" if granular else "fundamental_incompleto"
            if "FUND" in v: return "fundamental_completo" if "COMPL" in v else "fundamental_incompleto"
            if "ATE 5" in v or "5 A" in v or "6 A 9" in v: return "fundamental_incompleto"
            if "MEDIO" in v:
                if "INCOMP" in v: return "medio_incompleto"
                if "COMPL" in v: return "medio_completo"
            if "SUP" in v or "SUPERIOR" in v:
                if "INCOMP" in v: return "superior_incompleto"
                if "COMP" in v: return "superior_completo"
            if "MESTRADO" in v or "DOUTORADO" in v or "POS" in v: return "pos_graduacao"
            return None
        mapped = s_txt.map(map_text).astype("object")
        out.loc[mask_str] = out.loc[mask_str].where(out.loc[mask_str].notna(), mapped)
    return out.astype("object")

def _is_ti_from_path(path: str) -> Optional[int]:
    name = os.path.basename(path).lower()
    if 'processado_ti' in name or re.search(r'(^|[_\-])ti([_\-\.]|$)', name): return 1
    if 'processado_nao_ti' in name or re.search(r'nao[_\-]?ti', name): return 0
    return None

def _non_empty_keys(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    # Usa apenas colunas presentes e que não sejam totalmente NaN
    keys = []
    for k in candidates:
        if k in df.columns:
            col = df[k]
            try:
                all_nan = col.isna().all()
            except Exception:
                all_nan = False
            if not all_nan:
                keys.append(k)
    return keys

def _aggregate_one(df: pd.DataFrame, assumed_eh_ti: Optional[int] = None) -> pd.DataFrame:
    df = _norm_cols(df)
    col_ano = _pick(df, ['ano'])
    df['ano'] = _coerce_num(df[col_ano], to_int=True) if col_ano else np.nan

    # produzir 'uf' como CÓDIGO numérico (compatível com PNAD)
    col_uf = _pick(df, ['uf','sigla_uf','uf_trabalhador','uf_estabelecimento'])
    if col_uf:
        df['uf'] = _clean_uf_as_code(df[col_uf])
    else:
        col_mun = _pick(df, ['mun_trab','municipio','cod_municipio','municipio_ibge','cod_municipio_ibge',
                             'municipio_trabalhador','municipio_estabelecimento','munic_estab','codemun','cod_mun'])
        df['uf'] = _derive_uf_from_municipio(df[col_mun]) if col_mun else pd.Series([pd.NA]*len(df), dtype='Int64')

    col_sexo = _pick(df, ['sexo','sexo_trabalhador'])
    df['sexo'] = _clean_sexo(df[col_sexo]) if col_sexo else np.nan

    col_instr = _pick(df, ['escolaridade_apos_2005','nivel_instrucao','grau_instrucao','grau_de_instrucao','escolaridade'])
    df['nivel_instrucao'] = map_escolaridade_rais_to_pnad(df[col_instr], granular=False) if col_instr else np.nan

    col_eh_ti = _pick(df, ['eh_ti'])
    df['eh_ti'] = (_coerce_num(df[col_eh_ti], to_int=True).fillna(assumed_eh_ti) if col_eh_ti else assumed_eh_ti)

    col_ativo = _pick(df, [
        'vinculo_ativo_31_12','vinculo_ativo_3112','ind_vinculo_ativo_3112',
        'ativo_3112','ind_vinculo_ativo','indicador_vinculo_ativo_31_12','ind_vinculo_ativo_31_12'
    ])
    if col_ativo:
        base = df[col_ativo].astype(str).str.strip().str.lower()
        base = base.replace({'sim':'1','nao':'0','s':'1','n':'0','true':'1','false':'0'})
        df['ativo_3112'] = pd.to_numeric(base, errors='coerce')
    else:
        col_sit = _pick(df, ['situacao_vinculo','situacao','classe_desligamento'])
        if col_sit:
            s = df[col_sit].astype(str).str.lower().str.strip()
            df['ativo_3112'] = np.where(s.str.contains('ativo'), 1,
                                 np.where(s.str.contains('deslig'), 0, np.nan))
        else:
            df['ativo_3112'] = np.nan

    col_sal = _pick(df, ['salario','salario_mensal','remuneracao_media','rem_dezembro_corrigida','vl_remun_media_nom'])
    df['salario_mensal'] = _coerce_num(df[col_sal]) if col_sal else np.nan

    # Chaves dinâmicas (ignora colunas totalmente NaN)
    candidate_keys = ['ano','uf','sexo','nivel_instrucao','eh_ti']
    keys = _non_empty_keys(df, candidate_keys)

    grp = df.groupby(keys, dropna=False) if keys else None
    if grp is None:
        # Sem chaves válidas, retorna agregados globais
        out = pd.DataFrame({
            'n_vinculos_rais': [int(len(df))],
            'taxa_ativo_3112_rais': [_safe_nanmean(df['ativo_3112'])],
            'salario_median_rais': [_safe_nanmedian(df['salario_mensal'])]
        })
    else:
        out = grp.agg(
            n_vinculos_rais=('ativo_3112', 'size'),
            taxa_ativo_3112_rais=('ativo_3112', _safe_nanmean),
            salario_median_rais=('salario_mensal', _safe_nanmedian)
        ).reset_index()

    # Tipos finais
    for k in ['ano','eh_ti']:
        if k in out.columns: out[k] = _coerce_num(out[k], to_int=True)
    if 'uf' in out.columns:
        out['uf'] = _clean_uf_as_code(out['uf'])  # garante Int64
    if 'sexo' in out.columns: out['sexo'] = _clean_sexo(out['sexo'])
    if 'nivel_instrucao' in out.columns: out['nivel_instrucao'] = out['nivel_instrucao'].astype(str).str.lower().str.strip()
    return out

def _desired_cols() -> Set[str]:
    return {
        'ano',
        'uf','sigla_uf','uf_trabalhador','uf_estabelecimento',
        'mun_trab','municipio','cod_municipio','municipio_ibge','cod_municipio_ibge',
        'municipio_trabalhador','municipio_estabelecimento','munic_estab','codemun','cod_mun',
        'sexo','sexo_trabalhador',
        'nivel_instrucao','grau_instrucao','grau_de_instrucao','escolaridade','escolaridade_apos_2005',
        'eh_ti',
        'vinculo_ativo_31_12','vinculo_ativo_3112','ind_vinculo_ativo_3112','ativo_3112','ind_vinculo_ativo',
        'indicador_vinculo_ativo_31_12','ind_vinculo_ativo_31_12',
        'situacao_vinculo','situacao','classe_desligamento',
        'salario','salario_mensal','remuneracao_media','rem_dezembro_corrigida','vl_remun_media_nom'
    }

def carregar_rais_agregado(rais_dir: str) -> Dict[str, pd.DataFrame]:
    if rais_dir in _AGGS_CACHE:
        return _AGGS_CACHE[rais_dir]

    paths = glob.glob(os.path.join(rais_dir, "*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Nenhum parquet encontrado em {rais_dir}")

    aggs = []
    want = _desired_cols()
    for p in paths:
        assumed = _is_ti_from_path(p)
        try:
            if pq is not None:
                available = set(pq.ParquetFile(p).schema.names)
                cols = list(available & want) if (available & want) else None
            else:
                cols = None
            df = pd.read_parquet(p, columns=cols)
        except Exception:
            df = pd.read_parquet(p)
        try:
            aggs.append(_aggregate_one(df, assumed_eh_ti=assumed))
        except Exception:
            continue

    if not aggs:
        raise RuntimeError("Falha ao agregar RAIS: nenhuma agregação gerada")

    full = pd.concat(aggs, ignore_index=True)

    # Remover colunas inteiramente NaN para evitar merges impossíveis
    for c in ['ano','uf','sexo','nivel_instrucao','eh_ti']:
        if c in full.columns and full[c].isna().all():
            full = full.drop(columns=[c])

    # Consolidação por chaves disponíveis
    key_full = [c for c in ['ano','uf','sexo','nivel_instrucao','eh_ti'] if c in full.columns]
    if key_full:
        full = full.groupby(key_full, dropna=False).agg(
            n_vinculos_rais=('n_vinculos_rais','sum'),
            taxa_ativo_3112_rais=('taxa_ativo_3112_rais','mean'),
            salario_median_rais=('salario_median_rais','median')
        ).reset_index()

    # Fallback por UF+eh_ti, se possível
    key_uf = [k for k in ['uf','eh_ti'] if k in full.columns]
    if key_uf:
        agg_uf = full.groupby(key_uf, dropna=False).agg(
            n_vinculos_rais=('n_vinculos_rais','sum'),
            taxa_ativo_3112_rais=('taxa_ativo_3112_rais','mean'),
            salario_median_rais=('salario_median_rais','median')
        ).reset_index()
    else:
        agg_uf = pd.DataFrame()

    # Fallback por eh_ti apenas (sem ano/uf), se necessário
    key_ano = [k for k in ['ano','eh_ti'] if k in full.columns]
    if key_ano:
        agg_ano = full.groupby(key_ano, dropna=False).agg(
            n_vinculos_rais=('n_vinculos_rais','sum'),
            taxa_ativo_3112_rais=('taxa_ativo_3112_rais','mean'),
            salario_median_rais=('salario_median_rais','median')
        ).reset_index()
    else:
        # se não houver ano, disponibiliza uma versão por 'eh_ti' apenas, se existir
        if 'eh_ti' in full.columns:
            agg_ano = full.groupby(['eh_ti'], dropna=False).agg(
                n_vinculos_rais=('n_vinculos_rais','sum'),
                taxa_ativo_3112_rais=('taxa_ativo_3112_rais','mean'),
                salario_median_rais=('salario_median_rais','median')
            ).reset_index()
        else:
            agg_ano = pd.DataFrame()

    _AGGS_CACHE[rais_dir] = {'full': full, 'uf': agg_uf, 'ano': agg_ano}
    return _AGGS_CACHE[rais_dir]

def augment_pnad_with_rais(df_pnad: pd.DataFrame, rais_dir: str) -> pd.DataFrame:
    try:
        aggs = carregar_rais_agregado(rais_dir)
    except Exception:
        return df_pnad.copy()

    df = df_pnad.copy()
    if 'ano' in df.columns:   df['ano']   = _coerce_num(df['ano'], to_int=True)
    if 'eh_ti' in df.columns: df['eh_ti'] = _coerce_num(df['eh_ti'], to_int=True)
    if 'uf' in df.columns:    df['uf']    = _clean_uf_as_code(df['uf'])

    for k in ['ano','eh_ti']:
        if k in aggs.columns: aggs[k] = _coerce_num(aggs[k], to_int=True)
    if 'uf' in aggs.columns:
        aggs['uf'] = _clean_uf_as_code(aggs['uf'])
    elif 'mun_trab' in aggs.columns:
        aggs['uf'] = _derive_uf_from_municipio(aggs['mun_trab'])

    keys = [k for k in ['ano','uf','eh_ti'] if k in df.columns and k in aggs.columns]
    if not keys:
        return df

    merged = df.merge(aggs, on=keys, how='left', suffixes=('', '_raisagg'))

    rais_feats = [c for c in merged.columns if c.endswith('_rais') or c.endswith('_raisagg')]
    if rais_feats:
        cov = {c: float(merged[c].notna().mean()) for c in rais_feats}
        anos_rais = sorted(aggs['ano'].dropna().unique().tolist()) if 'ano' in aggs.columns else []
        anos_pnad = sorted(df['ano'].dropna().unique().tolist()) if 'ano' in df.columns else []
        inter = sorted(set(anos_rais).intersection(anos_pnad))
        print(f"[RAIS] Merge por {keys} | Cobertura: " + ", ".join([f"{k}={cov[k]:.1%}" for k in cov]))
        print(f"[RAIS] Anos RAIS: {anos_rais[:10]} | Anos PNAD: {anos_pnad[:10]} | Interseção: {inter[:10]}")
    return merged