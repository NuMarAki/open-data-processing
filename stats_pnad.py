#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funções auxiliares para estatísticas ponderadas da PNADc.
Calcula médias e proporções ponderadas consistentes.

Uso rápido:
from stats_pnad import gerar_estatisticas_ponderadas
res = gerar_estatisticas_ponderadas(df, peso_col='peso_populacional')

"""
from __future__ import annotations
import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors='coerce')
    weights = pd.to_numeric(weights, errors='coerce')
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float('nan')
    return float(np.average(values[mask], weights=weights[mask]))


def weighted_proportion(flag: pd.Series, weights: pd.Series) -> float:
    f = flag.astype(int)
    weights = pd.to_numeric(weights, errors='coerce')
    mask = weights.notna() & (weights > 0)
    if not mask.any():
        return float('nan')
    return float(np.sum(f[mask] * weights[mask]) / np.sum(weights[mask]))


def gerar_estatisticas_ponderadas(
    df: pd.DataFrame,
    peso_col: str = "peso_populacional",
    col_eh_ti: str = "eh_ti",
    col_idade: str = "idade",
    salvar: bool = True,
    caminho_saida: str = "reports/data_quality",
    nome_prefixo: str = "estatisticas_pnad",
) -> Dict[str, float]:
    """
    Retorna dict: proporção TI ponderada, idade média ponderada TI e não TI.
    Salva CSV opcionalmente.
    """
    if peso_col not in df.columns:
        return {"erro": "coluna_peso_ausente"}
    out: Dict[str, float] = {}
    if col_eh_ti in df.columns:
        out["proporcao_ti_ponderada"] = weighted_proportion(df[col_eh_ti], df[peso_col])
    if col_idade in df.columns and col_eh_ti in df.columns:
        out["idade_media_ti_ponderada"] = weighted_mean(
            df.loc[df[col_eh_ti], col_idade], df.loc[df[col_eh_ti], peso_col]
        )
        out["idade_media_outros_ponderada"] = weighted_mean(
            df.loc[~df[col_eh_ti], col_idade], df.loc[~df[col_eh_ti], peso_col]
        )
    if salvar:
        os.makedirs(caminho_saida, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        arq = os.path.join(caminho_saida, f"{nome_prefixo}_{ts}.csv")
        pd.DataFrame([out]).to_csv(arq, index=False, sep=';', encoding='utf-8-sig')
    return out


def resumo_rapido_ocupados(df: pd.DataFrame, peso_col='peso_populacional') -> Dict[str, float]:
    """
    Retorna um pequeno resumo de proporções entre ocupados.
    """
    r = {}
    if 'ocupado' not in df.columns or 'eh_ti' not in df.columns:
        return r
    df_occ = df[df['ocupado'] == 1]
    r['ocupados_total'] = len(df_occ)
    r['ocupados_ti'] = int(df_occ['eh_ti'].sum())
    r['prop_ti_ocup_simples'] = float(df_occ['eh_ti'].mean()) if len(df_occ) else float('nan')
    if peso_col in df_occ.columns:
        try:
            r['prop_ti_ocup_ponderada'] = weighted_proportion(df_occ['eh_ti'], df_occ[peso_col])
        except Exception:
            pass
    return r