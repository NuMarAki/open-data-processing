#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_all_pnad.py

Script único para rodar todas as checagens/diagnósticos auxiliares da
pipeline PNAD que sugeri. Reúne as seguintes funções:

- verify-layout: extrai substrings posicionales de um arquivo .txt (FWF)
  para verificar se a posição do CBO (V4010) está correta.
- diagnostics: faz diagnóstico detalhado de um CSV preprocessado (counts,
  proporções simples/ponderadas, top CBOs, idade, cobertura com cfg).
- aggregate-series: varre todos os CSVs preprocessados e gera série
  temporal (ano,trimestre) de participação TI entre ocupados (ponderada).
- prepare-ml: concatena CSVs preprocessados e prepara dataset cross-sectional
  (coorte 35-55 por default) pronto para modelagem.

Uso:
 python validate_all_pnad.py <comando> [opções]

Exemplos:
 python validate_all_pnad.py verify-layout --txt "Z:/TCC/IBGE/PNADC_032019.txt" --positions 152,155 --n 500
 python validate_all_pnad.py diagnostics --csv "C:/TCC/dados/pnad/preprocessados/PNADC_022017_preprocessado.csv" --cfg colunas_pnad.cfg
 python validate_all_pnad.py aggregate-series --dir "C:/TCC/dados/pnad/preprocessados" --out ti_series.csv
 python validate_all_pnad.py prepare-ml --dir "C:/TCC/dados/pnad/preprocessados" --out pnad_ml_35_55.csv

Saídas:
 - reports/validate_pnad/ : relatórios JSON/CSV, amostras e tops CBOs
 - aggregate CSV e ML CSV conforme opties

Observações:
 - Tenta carregar cfg via config_manager se disponível; se não, aceita --cfg (arquivo .cfg)
 - Usa utf-8-sig como padrão de escrita para compatibilidade com Excel
 - Não altera seus arquivos originais; apenas gera relatórios e arquivos derivados
"""
from __future__ import annotations
import os
import sys
import argparse
import logging
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
from collections import Counter

import pandas as pd
import numpy as np
import configparser

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("validate_all_pnad")

# ----------------------
# Utilitários comuns
# ----------------------
def try_read_csv(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    encs = ["utf-8-sig", "utf-8", "latin1", "iso-8859-1"]
    last_exc = None
    for enc in encs:
        try:
            df = pd.read_csv(path, sep=";", encoding=enc, low_memory=False, nrows=nrows)
            logger.info(f"Lido CSV com encoding {enc}: {path}")
            return df
        except Exception as e:
            last_exc = e
            logger.debug(f"Falha leitura CSV com {enc}: {e}")
    logger.error(f"Não foi possível ler CSV: {path}. Erro final: {last_exc}")
    raise last_exc

def safe_get_config_cbos(cfg_path: Optional[str]) -> Optional[List[str]]:
    # 1) try import config_manager
    try:
        import config_manager  # type: ignore
        cm = config_manager.ConfigManager()
        # tentar carregar config padrão (arquivo já depende do seu ambiente)
        try:
            cfg = cm.carregar_configuracao('pnad', 'colunas_pnad.cfg')
        except Exception:
            # tentar obter o primeiro config carregado
            cfg = cm.get_config('pnad')
        if cfg:
            cbos = getattr(cfg, 'cbo_ti', None) or getattr(cfg, 'cbos_ti', None)
            if cbos:
                logger.info(f"CBOs obtidos via config_manager: {cbos}")
                return list(map(str, cbos))
    except Exception:
        logger.debug("config_manager não disponível via import; tentaremos ler arquivo cfg diretamente.")

    # 2) ler arquivo cfg passado
    if not cfg_path:
        logger.warning("Nenhum config_manager disponível e nenhum --cfg informado; cbos_ti não carregados.")
        return None

    if not os.path.exists(cfg_path):
        logger.warning(f"Arquivo cfg informado não encontrado: {cfg_path}")
        return None

    try:
        parser = configparser.ConfigParser()
        parser.optionxform = str
        parser.read(cfg_path, encoding="utf-8")
        if parser.has_section('cbo_ti'):
            raw = parser.get('cbo_ti', 'codigos', fallback='')
            cbos = [c.strip() for c in raw.split(',') if c.strip()]
            cbos_fam = []
            for c in cbos:
                digits = ''.join(ch for ch in c if ch.isdigit())
                if len(digits) >= 4:
                    cbos_fam.append(digits[:4])
            cbos_fam = sorted(set(cbos_fam))
            logger.info(f"CBOs carregados do cfg ({cfg_path}): {cbos_fam}")
            return cbos_fam
        else:
            logger.warning(f"Seção [cbo_ti] não encontrada em {cfg_path}")
            return None
    except Exception as e:
        logger.error(f"Erro ao ler cfg {cfg_path}: {e}")
        return None

def normalizar_cbo_familia_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).fillna('').str.strip()
    s = s.str.replace(r'[^0-9]', '', regex=True)
    s = s.where(s.str.len() > 0, other=None)
    s = s.map(lambda x: None if x is None else x.zfill(4)[:4])
    s = s.where(lambda x: x != '0000', other=None)
    s = s.where(s.str.match(r'^[0-9]{4}$'), other=None)
    return s

def weighted_proportion(flag: pd.Series, weights: pd.Series) -> float:
    f = flag.astype(int)
    w = pd.to_numeric(weights, errors="coerce")
    mask = w.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float((f[mask] * w[mask]).sum() / w[mask].sum())

def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = vals.notna() & w.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(vals[mask], weights=w[mask]))

# ----------------------
# Subcomando: verify-layout
# ----------------------
def cmd_verify_layout(args):
    txt = args.txt
    if not os.path.exists(txt):
        logger.error("Arquivo TXT não encontrado: %s", txt)
        return
    start, end = map(int, args.positions.split(","))
    n = args.n
    counter = Counter()
    samples = []
    with open(txt, 'r', encoding=args.encoding or 'latin1', errors='ignore') as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            sub = line[start-1:end]
            counter[sub] += 1
            context = line[max(0, start-11):min(len(line), end+11)]
            samples.append((i+1, sub, context.replace("\n","\\n")))
    out = {
        "file": txt,
        "positions": f"{start}-{end}",
        "sample_lines": n,
        "top_values": counter.most_common(200),
        "samples": samples[:min(50,len(samples))]
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or "reports/validate_pnad"
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"layout_verify_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info("Layout verification saved: %s", path)
    # print summary
    print("Top 40 substrings:")
    for v, c in counter.most_common(40):
        print(f"{v!r}: {c}")

# ----------------------
# Subcomando: diagnostics
# ----------------------
def cmd_diagnostics(args):
    csv_path = args.csv
    if not csv_path:
        # find first csv in default dir
        default_dir = os.path.join("C:/TCC/dados/pnad", "preprocessados")
        if not os.path.exists(default_dir):
            default_dir = os.path.join("dados", "pnad", "preprocessados")
        if os.path.exists(default_dir):
            files = [os.path.join(default_dir, f) for f in os.listdir(default_dir) if f.lower().endswith(".csv")]
            files = sorted(files)
            if files:
                csv_path = files[0]
                logger.info(f"Nenhum --csv informado; usando primeiro encontrado: {csv_path}")
            else:
                logger.error(f"Nenhum CSV encontrado em {default_dir}. Forneça --csv explicitamente.")
                return
        else:
            logger.error(f"Pasta padrão {default_dir} não existe. Forneça --csv explicitamente.")
            return
    if not os.path.exists(csv_path):
        logger.error("CSV não encontrado: %s", csv_path)
        return

    df = try_read_csv(csv_path, nrows=args.nrows)
    # try load cbos
    cbos_cfg = safe_get_config_cbos(args.cfg)
    # normalize cbo_familia if needed
    if 'cbo_familia' not in df.columns and 'cbo_ocupacao' in df.columns:
        try:
            df['cbo_familia'] = normalizar_cbo_familia_series(df['cbo_ocupacao'])
            logger.info("cbo_familia gerada a partir de cbo_ocupacao (normalização aplicada).")
        except Exception as e:
            logger.warning("Falha ao normalizar cbo_familia: %s", e)

    # ensure eh_ti boolean
    if 'eh_ti' in df.columns:
        df['eh_ti'] = df['eh_ti'].astype(str).str.lower().map({'true': True, 'false': False, '1': True, '0': False}).fillna(df['eh_ti'])
        if df['eh_ti'].dtype != bool:
            df['eh_ti'] = df['eh_ti'].astype(bool)
    else:
        logger.warning("Coluna 'eh_ti' não encontrada no CSV; criando como False.")
        df['eh_ti'] = False

    # create outdir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.outdir or "reports/validate_pnad"
    os.makedirs(outdir, exist_ok=True)

    report = {}
    report['source_csv'] = csv_path
    report['timestamp'] = ts
    report['total_records'] = int(len(df))
    report['ti_count'] = int(df['eh_ti'].sum())
    report['ti_pct_simple'] = float(report['ti_count'] / report['total_records'] * 100) if report['total_records'] else float('nan')
    if 'peso_populacional' in df.columns:
        try:
            report['ti_pct_weighted_total'] = weighted_proportion(df['eh_ti'], df['peso_populacional']) * 100
        except Exception:
            report['ti_pct_weighted_total'] = float('nan')
    else:
        report['ti_pct_weighted_total'] = float('nan')

    if 'ocupado' in df.columns:
        df_occ = df[df['ocupado'] == 1]
        report['ocupados_total'] = int(len(df_occ))
        report['ti_ocupados_count'] = int(df_occ['eh_ti'].sum())
        report['ti_ocupados_pct_simple'] = float(report['ti_ocupados_count'] / report['ocupados_total'] * 100) if report['ocupados_total'] else float('nan')
        if 'peso_populacional' in df_occ.columns:
            try:
                report['ti_ocupados_pct_weighted'] = weighted_proportion(df_occ['eh_ti'], df_occ['peso_populacional']) * 100
            except Exception:
                report['ti_ocupados_pct_weighted'] = float('nan')
    else:
        report['ocupados_total'] = None
        report['ti_ocupados_count'] = None
        report['ti_ocupados_pct_simple'] = None
        report['ti_ocupados_pct_weighted'] = None

    report['idade'] = {}
    if 'idade' in df.columns:
        idn = pd.to_numeric(df['idade'], errors='coerce')
        report['idade']['min'] = float(idn.min(skipna=True)) if len(idn.dropna()) else None
        report['idade']['max'] = float(idn.max(skipna=True)) if len(idn.dropna()) else None
        report['idade']['mean'] = float(idn.mean(skipna=True)) if len(idn.dropna()) else None
        report['idade']['count_gt_100'] = int((idn > 100).sum())

    # top cbos
    top_cbos = {}
    for col in ['cbo_familia','cbo_ocupacao','cbo_raw']:
        if col in df.columns:
            s = df[col].astype(str).str.replace(r'[^0-9]', '', regex=True).str.zfill(4).str[:4]
            vc = s.value_counts().head(50).to_dict()
            top_cbos[col] = vc
    report['top_cbos'] = top_cbos

    report['cbo_coverage'] = {}
    if cbos_cfg:
        # pick cbo_familia if exists
        if 'cbo_familia' in df.columns:
            fam = df['cbo_familia'].astype(str).replace({'nan':None})
            fam = fam.where(fam != 'None', None)
            total = len(fam)
            present_mask = fam.isin(cbos_cfg)
            found = int(present_mask.sum())
            report['cbo_coverage']['present'] = True
            report['cbo_coverage']['found'] = found
            report['cbo_coverage']['pct'] = float(found / total * 100) if total else float('nan')
            report['cbo_coverage']['by_code'] = {c: int((fam==c).sum()) for c in cbos_cfg}
    else:
        report['cbo_coverage']['present'] = False

    # columns check
    required = ['idade', 'cbo_ocupacao', 'cbo_familia', 'eh_ti', 'peso_populacional', 'ocupado']
    cols_check = {}
    for col in required:
        cols_check[col] = {
            "present": col in df.columns,
            "missing_count": int(df[col].isna().sum()) if col in df.columns else None,
            "unique_values": int(df[col].nunique()) if col in df.columns else None
        }
    report['columns_check'] = cols_check

    # save json + csv summary
    json_path = os.path.join(outdir, f"diagnostic_report_{ts}.json")
    csv_path = os.path.join(outdir, f"diagnostic_summary_{ts}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    flat = {
        "source_csv": report['source_csv'],
        "timestamp": report['timestamp'],
        "total_records": report['total_records'],
        "ti_count": report['ti_count'],
        "ti_pct_simple": report['ti_pct_simple'],
        "ti_pct_weighted_total": report['ti_pct_weighted_total'],
        "ocupados_total": report['ocupados_total'],
        "ti_ocupados_count": report['ti_ocupados_count'],
        "ti_ocupados_pct_simple": report['ti_ocupados_pct_simple'],
        "ti_ocupados_pct_weighted": report['ti_ocupados_pct_weighted'],
        "idade_min": report['idade'].get('min'),
        "idade_max": report['idade'].get('max'),
        "idade_count_gt_100": report['idade'].get('count_gt_100'),
        "cbo_coverage_present": report['cbo_coverage'].get('present'),
        "cbo_coverage_pct": report['cbo_coverage'].get('pct')
    }
    pd.DataFrame([flat]).to_csv(csv_path, index=False, sep=';', encoding='utf-8-sig')
    logger.info("Relatório JSON salvo: %s", json_path)
    logger.info("Resumo CSV salvo: %s", csv_path)

    # save sample TI if exists
    if 'eh_ti' in df.columns and df['eh_ti'].sum() > 0:
        sample = df[df['eh_ti']].head(args.save_sample)
        sample_path = os.path.join(outdir, f"sample_ti_{ts}.csv")
        sample.to_csv(sample_path, index=False, sep=';', encoding='utf-8-sig')
        logger.info("Amostra TI salva em: %s", sample_path)
    else:
        logger.info("Nenhum registro TI encontrado para salvar amostra.")

    # save top cbo csvs
    for col, vc in top_cbos.items():
        tdf = pd.DataFrame.from_dict(vc, orient='index', columns=['count'])
        tdf.index.name = col
        tpath = os.path.join(outdir, f"top_{col}_{ts}.csv")
        tdf.to_csv(tpath, sep=';', encoding='utf-8-sig')
        logger.info("Top CBO salvo: %s", tpath)

    # print quick summary
    print("\n=== RELATÓRIO RÁPIDO ===")
    print(f"Arquivo analisado: {csv_path}")
    print(f"Total registros: {report['total_records']:,}")
    print(f"Profissionais TI identificados: {report['ti_count']:,} ({report['ti_pct_simple']:.6f}%)")
    if not np.isnan(report['ti_pct_weighted_total']):
        print(f"Proporção TI ponderada (toda população): {report['ti_pct_weighted_total']:.6f}%")
    if report['ocupados_total'] is not None:
        print(f"Ocupados: {report['ocupados_total']:,} | TI entre ocupados: {report['ti_ocupados_count']:,} ({report['ti_ocupados_pct_simple']:.6f}%)")
        if report['ti_ocupados_pct_weighted'] is not None:
            print(f"Proporção TI ocupados (ponderada): {report['ti_ocupados_pct_weighted']:.6f}%")
    print(f"Idade (min,max): {report['idade'].get('min')},{report['idade'].get('max')}")
    print(f"Registros com idade > 100: {report['idade'].get('count_gt_100')}")
    print("Relatórios e amostras salvos em:", outdir)

# ----------------------
# Subcomando: aggregate-series
# ----------------------
def cmd_aggregate_series(args):
    dir_pre = args.dir
    if not os.path.exists(dir_pre):
        logger.error("Diretório não encontrado: %s", dir_pre)
        return
    files = sorted([os.path.join(dir_pre, f) for f in os.listdir(dir_pre) if f.lower().endswith("_preprocessado.csv")])
    rows = []
    for f in files:
        logger.info("Processando %s", f)
        try:
            df = try_read_csv(f)
        except Exception:
            logger.warning("Falha ao ler %s, pulando", f)
            continue
        # basic conversions
        df['ano'] = pd.to_numeric(df.get('ano'), errors='coerce')
        df['trimestre'] = pd.to_numeric(df.get('trimestre'), errors='coerce')
        for (ano, trimestre), g in df.groupby(['ano','trimestre']):
            if pd.isna(ano) or pd.isna(trimestre):
                continue
            if 'ocupado' in g.columns:
                g_occ = g[g['ocupado'] == 1]
            else:
                g_occ = g
            ti_count = int(g_occ['eh_ti'].sum()) if 'eh_ti' in g_occ.columns else 0
            prop_simple = float(g_occ['eh_ti'].mean()) if 'eh_ti' in g_occ.columns and len(g_occ)>0 else np.nan
            prop_w = weighted_proportion(g_occ['eh_ti'], g_occ['peso_populacional']) if 'peso_populacional' in g_occ.columns and 'eh_ti' in g_occ.columns else np.nan
            rows.append({
                "ano": int(ano),
                "trimestre": int(trimestre),
                "n_registros": len(g),
                "n_ocupados": len(g_occ),
                "ti_count_ocupados": ti_count,
                "ti_prop_ocupados_simple": prop_simple,
                "ti_prop_ocupados_weighted": prop_w
            })
    outdf = pd.DataFrame(rows)
    if outdf.empty:
        logger.warning("Nenhum dado agregado (outdf vazio).")
        return
    outdf = outdf.sort_values(['ano','trimestre'])
    out_path = args.out or "ti_timeseries_by_quarter.csv"
    outdf.to_csv(out_path, index=False, sep=';', encoding='utf-8-sig')
    logger.info("Séries gravadas em %s", out_path)
    print("Séries gravadas em", out_path)

# ----------------------
# Subcomando: prepare-ml
# ----------------------
def cmd_prepare_ml(args):
    dir_pre = args.dir
    if not os.path.exists(dir_pre):
        logger.error("Diretório não encontrado: %s", dir_pre)
        return
    files = sorted([os.path.join(dir_pre, f) for f in os.listdir(dir_pre) if f.lower().endswith("_preprocessado.csv")])
    li = []
    for f in files:
        try:
            df = try_read_csv(f)
        except Exception:
            logger.warning("Falha ao ler %s, pulando", f)
            continue
        li.append(df)
    if not li:
        logger.error("Nenhum CSV preprocessado encontrado.")
        return
    df = pd.concat(li, ignore_index=True)
    # filter age
    df['idade'] = pd.to_numeric(df.get('idade'), errors='coerce')
    amin, amax = args.age_min, args.age_max
    df = df[(df['idade'] >= amin) & (df['idade'] <= amax)]
    # features
    features = pd.DataFrame()
    features['ano'] = df.get('ano')
    features['trimestre'] = df.get('trimestre')
    features['uf'] = df.get('uf')
    features['sexo'] = df.get('sexo')
    features['idade'] = df.get('idade')
    features['nivel_instrucao'] = df.get('nivel_instrucao')
    features['anos_estudo'] = pd.to_numeric(df.get('anos_estudo'), errors='coerce')
    features['horas_trabalhadas_semana'] = pd.to_numeric(df.get('horas_trabalhadas_semana'), errors='coerce')
    features['rendimento_trabalho_principal'] = pd.to_numeric(df.get('rendimento_trabalho_principal'), errors='coerce')
    # dummies for uf/sexo/nivel_instrucao (simple)
    features['uf'] = features['uf'].astype(str)
    features = pd.get_dummies(features, columns=['uf','sexo','nivel_instrucao'], dummy_na=True)
    features['eh_ti'] = df.get('eh_ti', False).astype(int)
    out = args.out or "pnad_ml_dataset.csv"
    features.to_csv(out, index=False, sep=';', encoding='utf-8-sig')
    logger.info("Dataset ML salvo em %s", out)
    print("Dataset salvo em", out)

# ----------------------
# CLI
# ----------------------
def main():
    parser = argparse.ArgumentParser(prog="validate_all_pnad.py", description="Ferramenta unificada de validação PNAD")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_verify = sub.add_parser("verify-layout", help="Verifica substrings em posições fixas do TXT (ex.: CBO em 152-155)")
    p_verify.add_argument("--txt", required=True, help="Caminho para o arquivo .txt descompactado PNAD")
    p_verify.add_argument("--positions", required=True, help="Formato inicio,fim (1-based inclusive); ex: 152,155")
    p_verify.add_argument("--n", type=int, default=500, help="Número de linhas a inspecionar")
    p_verify.add_argument("--encoding", default="latin1", help="Encoding do TXT (padrão latin1)")
    p_verify.add_argument("--outdir", default="reports/validate_pnad", help="Diretório de saída")

    p_diag = sub.add_parser("diagnostics", help="Diagnóstico detalhado de um CSV preprocessado")
    p_diag.add_argument("--csv", required=False, help="Caminho para o CSV preprocessado")
    p_diag.add_argument("--cfg", required=False, help="Caminho para o arquivo .cfg (opcional)")
    p_diag.add_argument("--outdir", required=False, default="reports/validate_pnad", help="Diretório de saída")
    p_diag.add_argument("--save-sample", required=False, type=int, default=200, help="N amostra TI a salvar")
    p_diag.add_argument("--nrows", required=False, type=int, default=None, help="Ler apenas nrows (útil para testes)")

    p_agg = sub.add_parser("aggregate-series", help="Agrega série TI por (ano,trimestre)")
    p_agg.add_argument("--dir", required=True, help="Diretório com CSV preprocessados")
    p_agg.add_argument("--out", required=False, help="Arquivo de saída CSV")
    p_agg.add_argument("--min-year", type=int, default=2012, help="Ano mínimo a incluir")

    p_ml = sub.add_parser("prepare-ml", help="Prepara dataset PNAD para ML (cross-section)")
    p_ml.add_argument("--dir", required=True, help="Diretório com CSV preprocessados")
    p_ml.add_argument("--out", required=False, default="pnad_ml_dataset.csv", help="Arquivo de saída")
    p_ml.add_argument("--age-min", type=int, default=35)
    p_ml.add_argument("--age-max", type=int, default=55)

    args = parser.parse_args()
    if args.cmd == "verify-layout":
        cmd_verify_layout(args)
    elif args.cmd == "diagnostics":
        cmd_diagnostics(args)
    elif args.cmd == "aggregate-series":
        cmd_aggregate_series(args)
    elif args.cmd == "prepare-ml":
        cmd_prepare_ml(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()