# -*- coding: utf-8 -*-
"""
Script simples para predição RAIS (multi-arquivos), alinhado ao restante do TCC.
Mantém a mesma lógica original, só ajusta mensagens/organização para ficar
coeso com os demais scripts preditivos do projeto.
"""

import argparse
import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from tabulate import tabulate

import numpy as np
import pandas as pd
import joblib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score
)
from sklearn.inspection import permutation_importance
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import configurar_ambiente
PREFIX = "[preditivo_rais]"


# Reuso de utilitários do repo para consistência de normalização RAIS
from preditivos.integracao_rais_pnad_features import (
    _norm_cols, _pick, _clean_uf_as_code, _derive_uf_from_municipio,
    map_escolaridade_rais_to_pnad
)


def _make_one_hot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True, dtype=np.float32)


def _coerce_num(s, to_int=False):
    x = pd.to_numeric(s, errors="coerce")
    if to_int:
        try:
            return x.round().astype("Int64")
        except Exception:
            return x.astype("Int64")
    return x


def _safe_metric(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return float("nan")


def _write_metrics_table(out_dir: Path, prefix: str, metrics_rows):
    """Salva tabela de métricas em CSV + PNG (formato simples, tipo planilha)."""
    metrics_df = pd.DataFrame(metrics_rows, columns=["Métrica", "Valor"])
    metrics_df.to_csv(out_dir / f"{prefix}_metricas.csv", index=False, sep=";")

    fig, ax = plt.subplots(figsize=(6.0, 2.6), dpi=200)
    ax.axis("off")
    table = ax.table(
        cellText=[[m, ("nan" if (isinstance(v, float) and np.isnan(v)) else f"{v:.4f}")] for m, v in metrics_rows],
        colLabels=["Métrica", "Valor"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.2)
    fig.tight_layout()
    fig.savefig(str(out_dir / f"{prefix}_tabela_metricas.png"), bbox_inches="tight")
    plt.close(fig)


def gerar_metricas_de_predicoes_csv(predicoes_csv: Path, out_dir: Path, prefix: str) -> Dict[str, float]:
    """Gera métricas (e tabela) a partir de um CSV de predições já salvo.

    Espera colunas: y_true, y_proba, y_pred_0_5 (ou calcula y_pred a partir de y_proba).
    """
    dfp = pd.read_csv(predicoes_csv, sep=";", low_memory=True)
    if "y_true" not in dfp.columns:
        raise ValueError(f"CSV não possui coluna 'y_true': {predicoes_csv}")
    if "y_proba" not in dfp.columns and "y_pred_0_5" not in dfp.columns:
        raise ValueError("CSV precisa ter 'y_proba' e/ou 'y_pred_0_5'.")

    y_true_series = pd.Series(pd.to_numeric(dfp["y_true"], errors="coerce"))
    y_true_series = y_true_series.dropna().astype(int)
    y_true = y_true_series
    dfp = dfp.loc[y_true.index]

    y_proba = None
    if "y_proba" in dfp.columns:
        y_proba = pd.Series(pd.to_numeric(dfp["y_proba"], errors="coerce"))

    if "y_pred_0_5" in dfp.columns:
        y_pred = pd.Series(pd.to_numeric(dfp["y_pred_0_5"], errors="coerce"))
    elif y_proba is not None:
        y_pred = (y_proba >= 0.5).astype("int64")
    else:
        raise ValueError("Não foi possível obter y_pred.")

    y_pred = pd.Series(y_pred).fillna(0).astype(int)

    acc = _safe_metric(accuracy_score, y_true, y_pred)
    bal_acc = _safe_metric(balanced_accuracy_score, y_true, y_pred)
    prec = _safe_metric(precision_score, y_true, y_pred, zero_division=0)
    rec = _safe_metric(recall_score, y_true, y_pred, zero_division=0)
    f1 = _safe_metric(f1_score, y_true, y_pred, zero_division=0)

    roc = float("nan")
    ap = float("nan")
    if y_proba is not None:
        roc = _safe_metric(roc_auc_score, y_true, y_proba)
        ap = _safe_metric(average_precision_score, y_true, y_proba)

    metrics_rows = [
        ("Acurácia", acc),
        ("Acurácia balanceada", bal_acc),
        ("Precisão", prec),
        ("Recall", rec),
        ("F1-score", f1),
        ("ROC AUC", roc),
        ("PR AUC", ap),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metrics_table(out_dir, prefix, metrics_rows)

    with open(out_dir / "resumo_metricas_de_predicoes.txt", "w", encoding="utf-8") as f:
        f.write(f"Fonte predições: {predicoes_csv}\n")
        vals, cnts = np.unique(y_true, return_counts=True)
        f.write(f"Distribuição y_true: {dict(zip(map(int, vals), map(int, cnts)))}\n\n")
        for m, v in metrics_rows:
            f.write(f"{m}: {v:.6f}\n" if not (isinstance(v, float) and np.isnan(v)) else f"{m}: nan\n")

    return {"acc": acc, "bal_acc": bal_acc, "prec": prec, "rec": rec, "f1": f1, "roc": roc, "ap": ap}


ALVO_CANDIDATOS = [
    "vinculo_ativo_3112", "vinculo_ativo_31_12", "ind_vinculo_ativo_3112", "ativo_3112"
]

# Colunas potencialmente vazadoras do desfecho (pós-evento/indicadores diretos)
LEAK_COLS = [
    "mes_desligamento", "motivo_desligamento",
    "causa_afastamento_1", "causa_afastamento_2", "causa_afastamento_3",
    "qtd_dias_afastamento",
    "vl_remun_dezembro_nom", "vl_remun_dezembro_sm",
] + [f"vl_rem_{mes}_cc" for mes in [
    "janeiro","fevereiro","marco","abril","maio","junho","julho","agosto","setembro","outubro","novembro"
]]


def _normalize_rais_df(df: pd.DataFrame) -> pd.DataFrame:
    df = _norm_cols(df)

    # Derivar UF (código IBGE) a partir de mun_trab ou usar UF já existente (se houver)
    if 'uf' in df.columns:
        df['uf'] = _clean_uf_as_code(df['uf'])
    else:
        col_mun = _pick(df, ['mun_trab','municipio','cod_municipio','municipio_ibge','cod_municipio_ibge',
                             'municipio_trabalhador','municipio_estabelecimento','munic_estab','codemun','cod_mun'])
        df['uf'] = _derive_uf_from_municipio(df[col_mun]) if col_mun else pd.Series([pd.NA]*len(df), dtype='Int64')

    # Idade
    if 'idade' in df.columns:
        df['idade'] = _coerce_num(df['idade'])
    else:
        df['idade'] = pd.Series([np.nan]*len(df), dtype='float64')

    # Sexo
    if 'sexo_trabalhador' in df.columns:
        s = df['sexo_trabalhador'].astype(str).str.strip().str.lower()
        df['sexo_trabalhador'] = np.where(s.isin(['1','m','masc','masculino']), 'M',
                                   np.where(s.isin(['2','f','fem','feminino']), 'F', s))

    # Escolaridade -> harmonização para categorias PNAD (granular False)
    if 'escolaridade_apos_2005' in df.columns:
        df['escolaridade_pnad'] = map_escolaridade_rais_to_pnad(df['escolaridade_apos_2005'], granular=False)
    else:
        df['escolaridade_pnad'] = pd.Series([np.nan]*len(df), dtype='object')

    # CBO família (já fornecido), garantir string curta
    if 'cbo_familia' in df.columns:
        df['cbo_familia'] = df['cbo_familia'].astype(str).str.extract(r'(\d{4})', expand=False)

    # CNAE (classe/subclasse após normalização)
    if 'cnae_20_classe' in df.columns:
        df['cnae_20_classe'] = df['cnae_20_classe'].astype(str).str.strip()
    if 'cnae_20_subclasse' in df.columns:
        df['cnae_20_subclasse'] = df['cnae_20_subclasse'].astype(str).str.strip()

    # Tempo de emprego, horas contratadas, média salarial
    if 'tempo_emprego' in df.columns:
        df['tempo_emprego'] = _coerce_num(df['tempo_emprego'])
    if 'qtd_hora_contr' in df.columns:
        df['qtd_hora_contr'] = _coerce_num(df['qtd_hora_contr'])
    if 'vl_remun_media_nom' in df.columns:
        df['vl_remun_media_nom'] = _coerce_num(df['vl_remun_media_nom'])

    # Indicadores binários comuns (usar float64 para acomodar NaN)
    for b in ['ind_simples','ind_cei_vinculado','ind_portador_defic']:
        if b in df.columns:
            s = df[b].astype(str).str.strip().str.lower()
            vals = np.where(
                s.isin(['1','sim','s','true']), 1.0,
                np.where(s.isin(['0','nao','n','false']), 0.0, np.nan)
            )
            df[b] = pd.Series(vals, index=df.index, dtype='float64')

    return df


def preparar_y(df: pd.DataFrame) -> pd.Series:
    # Detecta o alvo (normalizado) entre variações possíveis
    candidatos = [c for c in ALVO_CANDIDATOS if c in df.columns]
    if not candidatos:
        raise ValueError("Coluna de alvo não encontrada. Esperado algo como 'vinculo_ativo_31/12' (normaliza para 'vinculo_ativo_3112').")
    col = candidatos[0]
    s = df[col].astype(str).str.strip().str.lower()
    s = s.replace({'sim':'1','nao':'0','s':'1','n':'0','true':'1','false':'0'})
    y = pd.to_numeric(s, errors='coerce')
    return y


def selecionar_features(df: pd.DataFrame, ex_ante: bool = False, usar_cnae_subclasse: bool = False) -> Tuple[List[str], List[str]]:
    """
    Retorna listas (features_num, features_cat).
    ex_ante=False por padrão (prioriza qualidade preditiva). Ative para evitar vazamentos.
    """
    # Numéricas ex-ante
    features_num = [c for c in [
        'idade','qtd_hora_contr','tempo_emprego','vl_remun_media_nom'
    ] if c in df.columns]

    # Categóricas ex-ante
    cat_base = [
        'uf',
        'sexo_trabalhador','raca_cor','nacionalidade',
        'escolaridade_pnad',
        'cbo_familia',
        'cnae_20_classe',
        'natureza_juridica',
        'tamanho_estabelecimento',
        'tipo_admissao','tipo_vinculo',
        'tipo_estab','tipo_estab1',  # 'tipo_estab.1' -> 'tipo_estab1' após normalização
        'ibge_subsetor',
        'ind_simples','ind_cei_vinculado','ind_portador_defic'
    ]
    if usar_cnae_subclasse and 'cnae_20_subclasse' in df.columns:
        cat_base.append('cnae_20_subclasse')

    features_cat = [c for c in cat_base if c in df.columns]

    # Excluir colunas vazadoras (se existirem) e quaisquer alias do alvo
    drop_cols = set(ALVO_CANDIDATOS)
    if ex_ante:
        drop_cols.update([c for c in LEAK_COLS if c in df.columns])

    features_num = [c for c in features_num if c not in drop_cols]
    features_cat = [c for c in features_cat if c not in drop_cols]

    return features_num, features_cat


def _aggregate_importances(encoder: Optional[OneHotEncoder], features_num: List[str], features_cat: List[str], importances: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Agrega importâncias por variável original: soma das importâncias de todas as dummies daquele campo.
    """
    if encoder is not None and len(features_cat) > 0:
        try:
            cat_out = encoder.get_feature_names_out(features_cat)
        except Exception:
            cat_out = encoder.get_feature_names(features_cat)
    else:
        cat_out = np.array([], dtype=str)

    transformed_names = list(features_num) + list(cat_out)
    imp_df = pd.DataFrame({"feature_transformed": transformed_names, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)

    # Agregar por original
    agg_rows: Dict[str, float] = {}
    for name, imp in zip(transformed_names, importances):
        base = name.split("_", 1)[0] if name in cat_out else name
        agg_rows[base] = agg_rows.get(base, 0.0) + float(imp)

    agg_df = pd.DataFrame({"feature_original": list(agg_rows.keys()), "importance_sum": list(agg_rows.values())})
    agg_df = agg_df.sort_values("importance_sum", ascending=False).reset_index(drop=True)
    return imp_df, agg_df


def _plot_top20_stability(summary_df: pd.DataFrame, out_png: Path, title: str = "Top-20 Permutation Importance (Bootstrap)"):
    # summary_df: columns = feature, mean, std, q025, q975
    top20 = summary_df.sort_values("mean", ascending=True).tail(20)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top20, x="mean", y="feature", orient="h", color="#2c7fb8")
    # barras de erro (IC95%)
    for i, (m, lo, hi) in enumerate(zip(top20["mean"], top20["q025"], top20["q975"])):
        plt.plot([lo, hi], [i, i], color="#08306b", linewidth=2)
    plt.title(title)
    plt.xlabel("Permutation importance (média bootstrap)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _bootstrap_perm_importance(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, n_boot: int = 50, n_repeats: int = 3, scoring: str = "f1", seed: int = 42, restrict_to: Optional[List[str]] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    features = list(X_test.columns)
    feat_idx_map = {f: i for i, f in enumerate(features)}

    # baseline para escolher top-k e reduzir custo, se restrito
    pi0 = permutation_importance(
        pipe, X_test, y_test, n_repeats=n_repeats, scoring=scoring, n_jobs=-1, random_state=seed
    )
    base_imp = pd.Series(pi0.importances_mean, index=features).sort_values(ascending=False)
    if restrict_to is None:
        restrict_to = list(base_imp.index[:20])  # top-20 por padrão

    # coletar importâncias ao longo do bootstrap
    records = {f: [] for f in restrict_to}
    for b in range(n_boot):
        idx = rng.integers(0, len(X_test), size=len(X_test))
        Xb = X_test.iloc[idx]
        yb = y_test.iloc[idx]
        pi = permutation_importance(
            pipe, Xb, yb, n_repeats=n_repeats, scoring=scoring, n_jobs=-1, random_state=seed + b + 1
        )
        imp_b = pi.importances_mean
        for f in restrict_to:
            j = feat_idx_map[f]
            records[f].append(float(imp_b[j]))

    # sumarizar
    rows = []
    for f, vals in records.items():
        arr = np.array(vals, dtype=float)
        rows.append({
            "feature": f,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "q025": float(np.quantile(arr, 0.025)),
            "q975": float(np.quantile(arr, 0.975)),
            "n_boot": int(n_boot),
            "n_repeats": int(n_repeats),
            "scoring": scoring
        })
    summary = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    return summary


def _read_one_file(path: str, rows_per_file: Optional[int]) -> pd.DataFrame:
    try:
        if path.lower().endswith(".parquet"):
            df = pd.read_parquet(path)
        else:
            # CSV RAIS geralmente usa ';' e latin1
            df = pd.read_csv(path, sep=';', dtype=str, low_memory=True, encoding='latin1')
    except Exception:
        # fallback leitura CSV padrão
        df = pd.read_csv(path, dtype=str, low_memory=True, encoding='latin1')
    if rows_per_file and len(df) > rows_per_file:
        df = df.sample(n=rows_per_file, random_state=42)
    return df


def _collect_from_dir(rais_dir: str, patterns: List[str], rows_per_file: Optional[int], max_files: Optional[int]) -> pd.DataFrame:
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(rais_dir, pat)))
    paths = sorted(set(paths))
    if max_files is not None:
        paths = paths[:max_files]

    if not paths:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {rais_dir} com padrões: {patterns}")

    print(f"{PREFIX} Arquivos encontrados: {len(paths)}")
    df_parts: List[pd.DataFrame] = []
    total_rows = 0
    for i, p in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] Lendo: {os.path.basename(p)}")
        raw = _read_one_file(p, rows_per_file)
        total_rows += len(raw)
        print(f"    linhas lidas: {len(raw):,} (acumulado bruto: {total_rows:,})")
        norm = _normalize_rais_df(raw)
        df_parts.append(norm)
        del raw  # Liberar memória
        import gc
        gc.collect()
    print(f"{PREFIX} Concatenando {len(df_parts)} DataFrames...")
    df_all = pd.concat(df_parts, ignore_index=True)
    print(f"{PREFIX} Total concatenado (normalizado): {len(df_all):,} linhas")
    del df_parts  # Liberar memória da lista de partes
    gc.collect()

    print(f"\n{PREFIX} Calculando estatísticas descritivas (pode demorar)...")
    try:
        describe_df = df_all.describe().T
        print(tabulate(describe_df.reset_index().values.tolist(), headers=[""] + list(describe_df.columns), tablefmt='psql', showindex=False, floatfmt=".2f"))
    except Exception as e:
        print(f"{PREFIX} [aviso] Erro ao calcular estatísticas: {e}")

    return df_all


def treinar_avaliar(df: pd.DataFrame, features_num: List[str], features_cat: List[str], out_dir: Path, prefix: str,
                    n_boot: int = 50, n_repeats_perm: int = 3, sample_frac: Optional[float] = None):
    # Montar X e y
    y = preparar_y(df)
    X = df[features_num + features_cat].copy()
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask].astype(int)

    # Amostragem após limpeza (igual código original)
    if sample_frac is not None:
        if not (0.0 < sample_frac <= 1.0):
            raise ValueError("sample_frac deve estar entre (0, 1]")
        original_len = len(X)
        idx_sample = X.sample(frac=sample_frac, random_state=42).index
        X = X.loc[idx_sample]
        y = y.loc[idx_sample]
        print(f"{PREFIX} Amostragem aplicada: {original_len:,} -> {len(X):,} linhas ({sample_frac*100:.1f}%)")
    else:
        print(f"{PREFIX} Sem amostragem. Usando todos os dados: {len(X):,} linhas")

    strat = y if y.nunique() > 1 else None

    # Modelo + pré-processador
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), features_num),
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", _make_one_hot())
            ]), features_cat)
        ],
        remainder="drop"
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", rf)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=strat
    )

    pipe.fit(X_train, y_train)

    # Probabilidades robustas a classe única
    clf = pipe.named_steps["clf"]
    classes_ = getattr(clf, "classes_", np.array([0,1]))
    if len(classes_) == 1:
        cls = int(classes_[0])
        y_proba = np.full(len(X_test), 1.0 if cls == 1 else 0.0, dtype=float)
    else:
        col1 = int(np.where(classes_ == 1)[0][0]) if 1 in classes_ else int(np.argmax(classes_))
        y_proba = pipe.predict_proba(X_test)[:, col1]

    y_pred = (y_proba >= 0.5).astype(int)

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc = float("nan")
    try:
        ap = average_precision_score(y_test, y_proba)
    except Exception:
        ap = float("nan")

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "modelo.joblib")

    # Tabela de métricas (CSV + PNG) para uso em relatório/TCC
    try:
        metrics_rows = [
            ("Acurácia", float(acc)),
            ("Acurácia balanceada", float(bal_acc)),
            ("Precisão", float(prec)),
            ("Recall", float(rec)),
            ("F1-score", float(f1)),
            ("ROC AUC", float(roc)),
            ("PR AUC", float(ap)),
        ]
        _write_metrics_table(out_dir, prefix, metrics_rows)
    except Exception as e:
        print(f"[aviso] Não foi possível gerar tabela de métricas (CSV/PNG): {e}")

    # Predições
    meta_cols = [c for c in ["uf","idade","sexo_trabalhador","escolaridade_pnad","cnae_20_classe","cbo_familia","vl_remun_media_nom","qtd_hora_contr","tempo_emprego"] if c in X_test.columns]
    pred_df = X_test[meta_cols].reset_index(drop=True).copy()
    pred_df["y_true"] = y_test.reset_index(drop=True)
    pred_df["y_proba"] = y_proba
    pred_df["y_pred_0_5"] = y_pred
    pred_df.to_csv(out_dir / f"{prefix}_predicoes_teste.csv", index=False, sep=";")

    # Importâncias (árvore) + agregadas
    try:
        encoder = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["oh"] if features_cat else None
        importances = pipe.named_steps["clf"].feature_importances_
        imp_df, agg_df = _aggregate_importances(encoder, features_num, features_cat, importances)
        imp_df.to_csv(out_dir / "importances.csv", index=False, sep=";")
        agg_df.to_csv(out_dir / "importances_aggregated.csv", index=False, sep=";")
    except Exception as e:
        print(f"[aviso] Não foi possível extrair importâncias: {e}")

    # Permutation importance (global, no teste)
    perm_df = None
    try:
        pi = permutation_importance(
            pipe, X_test, y_test,
            n_repeats=n_repeats_perm, scoring="f1",
            n_jobs=-1, random_state=42
        )
        perm_df = pd.DataFrame({
            "feature": list(X.columns),
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std
        }).sort_values("importance_mean", ascending=False)
        perm_df.to_csv(out_dir / "permutation_importance.csv", index=False, sep=";")
    except Exception as e:
        print(f"[aviso] Falha permutation_importance: {e}")

    # Estabilidade (bootstrap) dos top-20 via permutation importance
    top20_summary = None
    try:
        summary = _bootstrap_perm_importance(
            pipe, X_test, y_test,
            n_boot=n_boot, n_repeats=n_repeats_perm, scoring="f1", seed=123
        )
        summary.to_csv(out_dir / "perm_importance_bootstrap_summary.csv", index=False, sep=";")
        # gráfico dos top-20
        _plot_top20_stability(summary, out_dir / "perm_importance_bootstrap_top20.png")
        top20_summary = summary.sort_values("mean", ascending=False).head(20)
    except Exception as e:
        print(f"[aviso] Falha bootstrap de permutation importance: {e}")

    # Resumo
    with open(out_dir / "resumo.txt", "w", encoding="utf-8") as f:
        f.write("Alvo: vinculo_ativo_3112 (1=ativo ao final do ano)\n")
        f.write(f"Arquivos RAIS processados para treino: múltiplos (ver console)\n")
        f.write(f"Amostra total após concatenação: {len(X):,}\n")
        f.write(f"Treino/Teste: {len(X_train):,}/{len(X_test):,}\n")
        vals, cnts = np.unique(y_train, return_counts=True)
        f.write(f"Distribuição treino: {dict(zip(map(int, vals), map(int, cnts)))}\n")
        vals, cnts = np.unique(y_test, return_counts=True)
        f.write(f"Distribuição teste: {dict(zip(map(int, vals), map(int, cnts)))}\n\n")
        f.write("Métricas (não ponderadas):\n")
        f.write(f"  Acc: {acc:.4f}  BalAcc: {bal_acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}\n")
        if not np.isnan(roc): f.write(f"  ROC AUC: {roc:.4f}\n")
        if not np.isnan(ap):  f.write(f"  PR AUC: {ap:.4f}\n")
        # seção top-20 estabilidade
        if top20_summary is not None and not top20_summary.empty:
            f.write("\nTop-20 fatores (Permutation Importance, média bootstrap, IC95%):\n")
            for _, row in top20_summary.iterrows():
                f.write(f"  - {row['feature']}: mean={row['mean']:.5f} | 95% CI [{row['q025']:.5f}, {row['q975']:.5f}]\n")

    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "roc": roc, "ap": ap,
        "n_test": len(X_test)
    }


def main():
    # manter o padrão de criação de pastas usado nos demais scripts
    configurar_ambiente(os.path.join("resultados", "rais", "ativo_3112"))

    ap = argparse.ArgumentParser(description="Predição RAIS (multi-arquivos): probabilidade de estar ativo em 31/12")
    ap.add_argument("--somente-metricas", action="store_true", help="Não treina; apenas gera tabela de métricas a partir do CSV de predições já salvo")
    ap.add_argument("--predicoes-csv", default=None, help="Caminho do CSV de predições (se omitido, usa resultados/rais/ativo_3112/rais_ativo_3112_predicoes_teste.csv)")
    ap.add_argument("--rais-dir", default=os.path.join("dados", "rais", "preprocessados"), help="Diretório com arquivos RAIS")
    ap.add_argument("--rows-per-file", type=int, default=None, help="Amostrar no máx. N linhas por arquivo (None = todas). Use valores baixos (ex: 50000) se tiver pouca RAM")
    ap.add_argument("--max-files", type=int, default=None, help="Limitar a N arquivos (None = todos). Use para debug ou memória limitada")
    ap.add_argument("--patterns", default="*_processado_ti.parquet,*_processado_ti.csv,*_processado.csv",
                    help="Padrões de arquivo separados por vírgula")
    ap.add_argument("--ex-ante", action="store_true", help="Ativar modo ex-ante (excluir vazamentos). Padrão: ativado.", default=True)
    ap.add_argument("--usar-cnae-subclasse", action="store_true", help="Incluir cnae_20_subclasse (alta cardinalidade)", default=False)
    ap.add_argument("--sample-frac", type=float, default=None, help="Fração de amostragem APÓS carregar arquivos (ex: 0.5 = 50%%). Use None para todos os dados. Padrão: None (100%%)")
    ap.add_argument("--n-boot", type=int, default=50, help="N bootstraps p/ estabilidade")
    ap.add_argument("--n-repeats", type=int, default=3, help="n_repeats do permutation importance")
    args = ap.parse_args()

    out_dir = Path("resultados") / "rais" / "ativo_3112"
    prefix = "rais_ativo_3112"

    if args.somente_metricas:
        pred_csv = Path(args.predicoes_csv) if args.predicoes_csv else (out_dir / f"{prefix}_predicoes_teste.csv")
        print(f"{PREFIX} Gerando métricas a partir de: {pred_csv}")
        gerar_metricas_de_predicoes_csv(pred_csv, out_dir=out_dir, prefix=prefix)
        print(f"{PREFIX} OK. Arquivos gerados em: {out_dir}")
        return

    patterns = [p.strip() for p in args.patterns.split(",") if p.strip()]
    print(f"{PREFIX} Varredura em: {args.rais_dir}")
    print(f"{PREFIX} Padrões: {patterns}")
    if args.rows_per_file:
        print(f"{PREFIX} Amostragem por arquivo: {args.rows_per_file} linhas")
    else:
        print(f"{PREFIX} [!] AVISO: Carregando TODOS os arquivos COMPLETOS. Isso pode consumir muita RAM.")
    if args.max_files:
        print(f"{PREFIX} Limite de arquivos: {args.max_files}")

    if args.sample_frac and args.sample_frac < 1.0:
        print(f"{PREFIX} [!] Amostragem final: {args.sample_frac*100:.1f}% dos dados após carregar")
    else:
        print(f"{PREFIX} [!] Usando 100% dos dados (sem amostragem)")

    df = _collect_from_dir(args.rais_dir, patterns, args.rows_per_file, args.max_files)
    features_num, features_cat = selecionar_features(df, ex_ante=args.ex_ante, usar_cnae_subclasse=args.usar_cnae_subclasse)

    metrics = treinar_avaliar(df, features_num, features_cat, out_dir, prefix,
                              n_boot=args.n_boot, n_repeats_perm=args.n_repeats,
                              sample_frac=args.sample_frac)
    print(f"{PREFIX} Treino concluído. Resultados em: {out_dir.resolve()}")
    print(f"{PREFIX} Resumo -> Acc={metrics['acc']:.3f} Prec={metrics['prec']:.3f} Rec={metrics['rec']:.3f} F1={metrics['f1']:.3f} ROC={metrics['roc']:.3f} PR-AUC={metrics['ap']:.3f}")


if __name__ == "__main__":
    main()
