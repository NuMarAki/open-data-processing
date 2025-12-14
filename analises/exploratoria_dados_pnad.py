import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend to avoid Tkinter errors
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import balanced_accuracy_score

# Dica de performance (setar antes de rodar o Python para evitar over-subscription de threads):
# No PowerShell, antes de executar:
# $env:MKL_NUM_THREADS="1"; $env:OPENBLAS_NUM_THREADS="1"; $env:NUMEXPR_NUM_THREADS="1"; $env:OMP_NUM_THREADS=[Environment]::ProcessorCount

from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preditivos.integracao_rais_pnad_features import augment_pnad_with_rais
from preditivos.metrics_validacao_rais import validar_com_rais
from scripts.utils import classificar_faixa_etaria  # <-- novo import

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, learning_curve, GroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve, average_precision_score, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

def _make_one_hot():
    try:
        # Para versões mais recentes do scikit-learn
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        # Para versões mais antigas do scikit-learn
        return OneHotEncoder(sparse=False, handle_unknown="ignore")

# ------------------ Ajuste UF para linkagem RAIS × PNAD (usar códigos numéricos) ------------------
IBGE_UF = {
    11: 'RO', 12: 'AC', 13: 'AM', 14: 'RR', 15: 'PA', 16: 'AP', 17: 'TO',
    21: 'MA', 22: 'PI', 23: 'CE', 24: 'RN', 25: 'PB', 26: 'PE', 27: 'AL', 28: 'SE', 29: 'BA',
    31: 'MG', 32: 'ES', 33: 'RJ', 35: 'SP',
    41: 'PR', 42: 'SC', 43: 'RS',
    50: 'MS', 51: 'MT', 52: 'GO', 53: 'DF'
}


def _faixa_idade_series(idade: pd.Series) -> pd.Series:
    bins = [0, 29, 34, 44, 55, 120]
    labels = ["<=29", "30-34", "35-44", "45-55", ">=56"]
    return pd.cut(idade, bins=bins, labels=labels, right=True)

def salvar_heatmaps_idade_estudo(df_base: pd.DataFrame, y_proba: np.ndarray, alvo_nome: str, out_dir: str):
    """Gera mapas de calor de probabilidade média de estar ocupado por faixa etária × anos de estudo, separados por TI/Não-TI (dados de teste)."""
    os.makedirs(out_dir, exist_ok=True)
    dfh = df_base.copy()
    dfh['proba'] = y_proba
    dfh['faixa_idade'] = _faixa_idade_series(dfh['idade'])
    dfh['anos_estudo'] = pd.to_numeric(dfh.get('anos_estudo'), errors='coerce').astype('Int64')
    dfh = dfh.dropna(subset=['faixa_idade','anos_estudo','proba','eh_ti'])

    for flag, nome in [(1, 'TI'), (0, 'NaoTI')]:
        sub = dfh[dfh['eh_ti'] == flag]
        if sub.empty: 
            continue
        pv = sub.pivot_table(index='faixa_idade', columns='anos_estudo', values='proba', aggfunc='mean')
        # ordenar colunas
        pv = pv.reindex(index=['<=29','30-34','35-44','45-55','>=56'])
        plt.figure(figsize=(16, 6))
        sns.heatmap(pv, cmap='Greens', vmin=0, vmax=1, cbar_kws={'label': 'Probabilidade média de estar ocupado'})
        plt.title(f'Probabilidade de estar ocupado por faixa etária × anos de estudo ({nome})')
        plt.xlabel('Anos de estudo'); plt.ylabel('Faixa etária')
        plt.tight_layout()
        caminho = os.path.join(out_dir, f'{alvo_nome}_heatmap_idade_estudo_{nome}.png')
        plt.savefig(caminho, dpi=300)
        plt.close()
        # também salvar CSV
        pv.to_csv(os.path.join(out_dir, f'{alvo_nome}_heatmap_idade_estudo_{nome}.csv'), sep=';')

# ...inside rf_ti_6sm, logo após salvar pred_df/pred_path...
    # Heatmaps: prob. de estar ocupado por idade × anos de estudo (apenas alvo=ocupado)
    if alvo == "ocupado":
        try:
            df_base = df_scope.reset_index(drop=True).iloc[X_test.index][['idade','anos_estudo','eh_ti']]
            salvar_heatmaps_idade_estudo(
                df_base=df_base,
                y_proba=y_proba,
                alvo_nome=prefix,
                out_dir=str(graficos)
            )
            print("Heatmaps por idade × anos de estudo salvos.")
        except Exception as e:
            print("[aviso] Heatmaps idade×estudo não gerados:", e)

def rais_extract_uf_code(df_rais, mun_col='mun_trab'):
    """Extrai código UF (2 primeiros dígitos do código IBGE municipal) e cria uf_ibge_code (numérico)."""
    if mun_col not in df_rais.columns:
        raise KeyError(f"Coluna RAIS esperada '{mun_col}' ausente.")
    s = df_rais[mun_col].astype(str).str.strip().str.zfill(6)
    df = df_rais.copy()
    # dois primeiros dígitos -> código UF IBGE (numérico)
    df['uf_ibge_code'] = pd.to_numeric(s.str[:2], errors='coerce').astype('Int64')
    df['uf_sigla'] = df['uf_ibge_code'].map(IBGE_UF)
    return df

def pnad_force_uf_code(df_pnad, uf_col='uf'):
    """Garante coluna uf_ibge_code numérica na PNAD (PNAD já usa códigos numéricos)."""
    p = df_pnad.copy()
    if uf_col not in p.columns:
        p['uf_ibge_code'] = pd.NA
        return p
    # força numérico (ex.: '35' -> 35); mantém NA quando inválido
    p['uf_ibge_code'] = pd.to_numeric(p[uf_col], errors='coerce').astype('Int64')
    return p

def _plot_roc_pr(y_true, y_proba, outdir: Path, prefix: str):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    p1 = outdir / f"{prefix}_roc.png"
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}", color='darkgreen')
    plt.plot([0,1], [0,1], "k--", alpha=0.5)
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close()

    p2 = outdir / f"{prefix}_pr.png"
    plt.figure(figsize=(6,5))
    plt.plot(rec, prec, label=f"AP={pr_auc:.3f}", color='darkgreen')
    plt.xlabel("Recall")
    plt.ylabel("Precisão")
    plt.title("Curva Precisão-Recall")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close()
    return [str(p1.resolve()), str(p2.resolve())]

def _plot_calibration(y_true, y_proba, outdir: Path, prefix: str):
    # Aumentar n_bins e usar strategy="uniform" para cobrir melhor o intervalo [0,1]
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=20, strategy="uniform")
    p = outdir / f"{prefix}_calibration.png"
    plt.figure(figsize=(6,5))
    plt.plot(prob_pred, prob_true, marker="o", color='darkgreen')
    plt.plot([0,1], [0,1], "k--", alpha=0.5)
    plt.xlabel("Probabilidade prevista")
    plt.ylabel("Fração positiva real")
    plt.title("Curva de Calibração")
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(p, dpi=300)
    plt.close()
    return [str(p.resolve())]


def _plot_confusion(cm, outdir: Path, prefix: str):
    p1 = outdir / f"{prefix}_cm_counts.png"
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title("Matriz de Confusão")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(p1, dpi=300)
    plt.close()

    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    p2 = outdir / f"{prefix}_cm_norm.png"
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens")
    plt.title("Matriz de Confusão (normalizada)")
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(p2, dpi=300)
    plt.close()
    return [str(p1.resolve()), str(p2.resolve())]


def _threshold_report(y_true, y_proba, outdir: Path, prefix: str, sample_weight=None):
    rows = []
    for thr in np.linspace(0.05, 0.95, 19):
        y_pred = (y_proba >= thr).astype(int)
        metrics = {
            "threshold": thr,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0)  # Recall para classe negativa
        }
        if sample_weight is not None:
            metrics.update({
                "precision_w": precision_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight),
                "recall_w": recall_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight),
                "f1_w": f1_score(y_true, y_pred, zero_division=0, sample_weight=sample_weight),
                "accuracy_w": accuracy_score(y_true, y_pred, sample_weight=sample_weight)
            })
        rows.append(metrics)
    df_thr = pd.DataFrame(rows)
    path = outdir / f"{prefix}_threshold_metrics.csv"
    df_thr.to_csv(path, index=False, sep=";")
    return path, df_thr

def _subgroup_metrics(df_meta_test: pd.DataFrame, y_true: pd.Series, y_proba: np.ndarray, outdir: Path, prefix: str):
    y_true = y_true.reset_index(drop=True)
    y_pred = (y_proba >= 0.5).astype(int)

    sg = []
    if "idade" in df_meta_test.columns:
        try:
            df_meta_test["faixa_etaria"] = classificar_faixa_etaria(df_meta_test["idade"])
            for faixa, grp in df_meta_test.groupby("faixa_etaria"):
                idx = grp.index
                if len(idx) < 10 or y_true.loc[idx].nunique() <= 1:
                    continue
                sg.append({
                    "grupo": f"idade:{faixa}",
                    "n": len(idx),
                    "pos_rate": float(y_true.loc[idx].mean()),
                    "acc": accuracy_score(y_true.loc[idx], y_pred[idx]),
                    "prec": precision_score(y_true.loc[idx], y_pred[idx], zero_division=0),
                    "rec": recall_score(y_true.loc[idx], y_pred[idx], zero_division=0),
                    "f1": f1_score(y_true.loc[idx], y_pred[idx], zero_division=0)
                })
        except Exception as e:
            print(f"Erro em faixa_etaria: {e}")

    if "sexo" in df_meta_test.columns:
        for sexo, grp in df_meta_test.groupby("sexo"):
            idx = grp.index
            if len(idx) < 10 or y_true.loc[idx].nunique() <= 1:
                continue
            sg.append({
                "grupo": f"sexo:{sexo}",
                "n": len(idx),
                "pos_rate": float(y_true.loc[idx].mean()),
                "acc": accuracy_score(y_true.loc[idx], y_pred[idx]),
                "prec": precision_score(y_true.loc[idx], y_pred[idx], zero_division=0),
                "rec": recall_score(y_true.loc[idx], y_pred[idx], zero_division=0),
                "f1": f1_score(y_true.loc[idx], y_pred[idx], zero_division=0)
            })

    if "nivel_instrucao" in df_meta_test.columns:
        for ni, grp in df_meta_test.groupby("nivel_instrucao"):
            idx = grp.index
            if len(idx) < 10 or y_true.loc[idx].nunique() <= 1:
                continue
            sg.append({
                "grupo": f"nivel_instrucao:{ni}",
                "n": len(idx),
                "pos_rate": float(y_true.loc[idx].mean()),
                "acc": accuracy_score(y_true.loc[idx], y_pred[idx]),
                "prec": precision_score(y_true.loc[idx], y_pred[idx], zero_division=0),
                "rec": recall_score(y_true.loc[idx], y_pred[idx], zero_division=0),
                "f1": f1_score(y_true.loc[idx], y_pred[idx], zero_division=0)
            })

    df_sg = pd.DataFrame(sg)
    out_path = None
    if not df_sg.empty:
        out_path = outdir / f"{prefix}_subgroup_metrics.csv"
        df_sg.to_csv(out_path, index=False, sep=";")
    return out_path, df_sg


def _learning_curve(pipe, X, y, outdir: Path, prefix: str):
    sizes = np.linspace(0.1, 1.0, 5)
    train_sizes, train_scores, test_scores = learning_curve(
        pipe, X, y, cv=3, scoring="f1", n_jobs=-1, train_sizes=sizes, shuffle=True, random_state=42
    )
    p = outdir / f"{prefix}_learning_curve.png"
    plt.figure(figsize=(6,5))
    plt.plot(train_sizes, train_scores.mean(axis=1), marker="o", label="Treino (F1)", color='darkgreen')
    plt.plot(train_sizes, test_scores.mean(axis=1), marker="o", label="Validação (F1)", color='green')
    plt.xlabel("Tamanho de treino")
    plt.ylabel("Pontuação")
    plt.title("Curva de Aprendizado (F1)")
    plt.legend()
    plt.grid(True, ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(p, dpi=300)
    plt.close()
    return [str(p.resolve())]


def _aggregate_categorical_importances(encoder, features_num, features_cat, importances, outdir: Path, prefix: str):
    try:
        if features_cat:
            cat_cols_out = encoder.get_feature_names_out(features_cat)
        else:
            cat_cols_out = np.array([], dtype=str)
        final_names = list(features_num) + list(cat_cols_out)
        imp = pd.Series(importances, index=final_names)
        rows = [{"feature": fn, "importance": float(imp[fn])} for fn in features_num if fn in imp.index]
        for cat in features_cat:
            cat_cols = [c for c in imp.index if c.startswith(cat + "_")]
            rows.append({"feature": cat, "importance": float(imp[cat_cols].sum()) if cat_cols else 0.0})
        df_imp = pd.DataFrame(rows).sort_values("importance", ascending=False)
        path = outdir / f"{prefix}_importances_aggregated.csv"
        df_imp.to_csv(path, sep=";", index=False)
        return df_imp, str(path.resolve())
    except Exception as e:
        print(f"[aviso] Falha ao agregar importâncias: {e}")
        return pd.DataFrame(), None


# ---------- Amostragem estratificada por múltiplos estratos (ano, trimestre, y) ----------
def _allocate_quotas(group_sizes: np.ndarray, max_rows: int):
    total = int(group_sizes.sum())
    if total <= max_rows:
        return group_sizes.astype(int)
    proportions = group_sizes / total
    raw = proportions * max_rows
    quotas = np.floor(raw).astype(int)
    short = max_rows - quotas.sum()
    if short > 0:
        residuals = raw - quotas
        add_idx = np.argsort(-residuals)[:short]
        quotas[add_idx] += 1
    return quotas


def _stratified_downsample_multi(df: pd.DataFrame, strata_cols, y_col: str, max_rows: int, random_state: int = 42):
    if len(df) <= max_rows:
        return df, None, None
    
    # resumo antes
    resumo_antes = df.groupby(strata_cols + [y_col]).size().reset_index(name="n")
    
    # cotas por estrato temporal (sem y)
    g = df.groupby(strata_cols)
    sizes = g.size().to_numpy()
    quotas = _allocate_quotas(sizes, max_rows)
    
    sampled_parts = []
    rng = np.random.RandomState(random_state)
    for (keys, grp), q in zip(g, quotas):
        if len(grp) <= q:
            # pegar tudo
            sampled_parts.append(grp)
        else:
            # amostragem estratificada por y
            grp_y = grp.groupby(y_col)
            y_sizes = grp_y.size().to_numpy()
            y_quotas = _allocate_quotas(y_sizes, q)
            y_parts = []
            for (y_val, y_grp), y_q in zip(grp_y, y_quotas):
                if len(y_grp) <= y_q:
                    y_parts.append(y_grp)
                else:
                    y_parts.append(y_grp.sample(n=int(y_q), random_state=rng))
            sampled_parts.append(pd.concat(y_parts, axis=0))
    
    df_out = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    resumo_depois = df_out.groupby(strata_cols + [y_col]).size().reset_index(name="n")
    
    return df_out, resumo_antes, resumo_depois


# ----------------------------------------------------------------------
def modelo_aprimorado(caminho_csv=r"C:\TCC\dados\pnad\dados_pnad_consolidados.csv"):
    # mantido (sem alterações relevantes)
    pass


# ----------------------------------------------------------------------
def rf_ti_6sm(
    caminho_csv=r"C:\TCC\dados\pnad\dados_pnad_consolidados.csv",
    n_amostra=None,
    alvo="salario_6sm",
    usar_peso=True,
    usar_subpastas_por_alvo=True,
    rf_verbose=0,
    escopo_ocupado="auto",
    limite_auto_pop_ocupado=2_000_000,
    rf_n_estimators=500,
    rf_max_depth=None,
    rf_min_samples_leaf=5,
    skip_cv_threshold=1_500_000,
    ajuste_auto_grande_porte=True,
    amostragem_por_tempo=True,
    usar_variaveis_tempo=True,
    split_temporal=False,
    ano_teste=None,
    validar_grade=False,
    usar_rais_features=False,                          # desativado por padrão (TCC sem integração RAIS)
    rais_dir=r"C:\TCC\dados\rais\preprocessados"
):
    print("\n[rf_ti_6sm] Iniciando...")
    t0 = time.perf_counter()

    base_graficos = Path("graficos"); base_graficos.mkdir(exist_ok=True, parents=True)
    base_resultados = Path("resultados"); base_resultados.mkdir(exist_ok=True, parents=True)
    base_modelos = Path("modelos"); base_modelos.mkdir(exist_ok=True, parents=True)

    if usar_subpastas_por_alvo:
        graficos = base_graficos / alvo; graficos.mkdir(exist_ok=True, parents=True)
        resultados = base_resultados / alvo; resultados.mkdir(exist_ok=True, parents=True)
        modelos = base_modelos / alvo; modelos.mkdir(exist_ok=True, parents=True)
    else:
        graficos, resultados, modelos = base_graficos, base_resultados, base_modelos

    print(f"Diretório de gráficos: {graficos.resolve()}")
    print(f"Diretório de resultados: {resultados.resolve()}")
    print(f"CPU cores (os.cpu_count): {os.cpu_count()}")

    # Carregar dados
    try:
        df = pd.read_csv(caminho_csv, sep=";")
        print(f"Dados carregados: {caminho_csv}")
    except FileNotFoundError:
        print("Arquivo principal não encontrado. Carregando 'amostra_pnad.csv'.")
        df = pd.read_csv("amostra_pnad.csv", sep=";")

    # --- Normalizar UF na PNAD (cria coluna numérica uf_ibge_code) ---
    try:
        df = pnad_force_uf_code(df, uf_col='uf')
        print("PNAD: coluna uf_ibge_code criada (numérica).")
    except Exception as e:
        print("Aviso ao normalizar UF na PNAD:", e)
    
    total_lidos = len(df)
    print(f"Registros totais lidos (todas as ocupações): {total_lidos:,}")

    # Tipos
    if df["eh_ti"].dtype == "object":
        df["eh_ti"] = df["eh_ti"].astype(str).str.upper().map({"TRUE": True, "FALSE": False})
    df["eh_ti"] = df["eh_ti"].fillna(False).astype(bool)

    to_num = [
        "ano","trimestre","idade","anos_estudo","rendimento_trabalho_principal","rendimento_bruto_mensal",
        "horas_trabalhadas_semana","sexo","cor_raca","ocupado","forca_trabalho","peso_populacional",
        "nivel_instrucao","uf","carteira_assinada","tipo_area"
    ]
    for c in to_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "ano" in df.columns:
        df = df[df["ano"].between(2001, 2100, inclusive="both")]

    # Escopo TI
    df_ti = df[df["eh_ti"] == True].copy()
    ti_filtrados = len(df_ti)
    print(f"Registros TI: {ti_filtrados:,} (de {total_lidos:,})")

    # Auxiliares PNAD (SM por ano e renda_uso)
    salario_minimo_por_ano = {2012: 622, 2013: 678, 2014: 724, 2015: 788, 2016: 880, 2017: 937,
                              2018: 954, 2019: 998, 2020: 1045, 2021: 1100, 2022: 1212, 2023: 1320, 2024: 1412}
    if "ano" in df_ti.columns:
        df_ti["salario_minimo_ano"] = df_ti["ano"].map(salario_minimo_por_ano)
    else:
        df_ti["salario_minimo_ano"] = 1212

    df_ti["renda_uso"] = np.nan
    for rc in ["rendimento_trabalho_principal","rendimento_bruto_mensal"]:
        if rc in df_ti.columns:
            df_ti["renda_uso"] = df_ti["renda_uso"].fillna(df_ti[rc])

    # --- CRIAR faixa_etaria (necessário para analises/heatmaps posteriores) ---
    if "idade" in df_ti.columns:
        df_ti["idade"] = pd.to_numeric(df_ti["idade"], errors="coerce")
        df_ti["faixa_etaria"] = df_ti["idade"].apply(lambda x: classificar_faixa_etaria(x) if pd.notnull(x) else "Outros")
    else:
        df_ti["faixa_etaria"] = "Outros"

    # Definição do escopo e alvo
    scope = "ti"
    if alvo == "salario_6sm":
        cobertura = df_ti[["renda_uso","salario_minimo_ano"]].notna().all(axis=1)
        print(f"Cobertura para cálculo 6SM (renda e SM por ano presentes): {cobertura.mean():.1%}")
        df_scope = df_ti[cobertura].copy()
        df_scope["y"] = (df_scope["renda_uso"] >= 6 * df_scope["salario_minimo_ano"]).astype(int)
    elif alvo == "ocupado":
        if escopo_ocupado in ("auto", "ti"):
            df_scope = df_ti.dropna(subset=["ocupado"]).copy()
            df_scope["y"] = (df_scope["ocupado"] == 1).astype(int)
            if escopo_ocupado == "auto" and df_scope["y"].nunique() < 2:
                print("[aviso] alvo='ocupado' no escopo TI ficou com uma única classe. Alternando para escopo POPULAÇÃO.")
                scope = "populacao"
        if scope == "populacao":
            df_scope = df.dropna(subset=["ocupado"]).copy()
            df_scope["y"] = (df_scope["ocupado"] == 1).astype(int)
    else:
        raise ValueError("alvo deve ser 'salario_6sm' ou 'ocupado'.")

    print(f"Registros {('TI' if scope=='ti' else 'População')} com alvo definido: {len(df_scope):,}")

    # Amostragem automática para alvo='ocupado' em população
    if alvo == "ocupado" and scope == "populacao" and limite_auto_pop_ocupado is not None:
        if len(df_scope) > limite_auto_pop_ocupado:
            antes = len(df_scope)
            if amostragem_por_tempo and {"ano","trimestre"}.issubset(df_scope.columns):
                df_scope, res_a, res_d = _stratified_downsample_multi(
                    df_scope, strata_cols=["ano","trimestre"], y_col="y",
                    max_rows=limite_auto_pop_ocupado, random_state=42
                )
                # salvar resumos
                if res_a is not None:
                    res_a.to_csv(resultados / "amostragem_tempo_resumo_antes.csv", index=False, sep=";")
                if res_d is not None:
                    res_d.to_csv(resultados / "amostragem_tempo_resumo_depois.csv", index=False, sep=";")
            else:
                # fallback: estratificada só por y (função simples)
                frac = limite_auto_pop_ocupado / len(df_scope)
                df_scope = df_scope.groupby("y", group_keys=False).apply(
                    lambda g: g.sample(frac=frac, random_state=42) if len(g) > 0 else g
                ).sample(frac=1.0, random_state=42).reset_index(drop=True)
            print(f"[info] Amostragem aplicada: {antes:,} -> {len(df_scope):,} (limite_auto_pop_ocupado={limite_auto_pop_ocupado:,})")

    # Seleção de features
    features_num = [c for c in ["idade","anos_estudo","horas_trabalhadas_semana"] if c in df_scope.columns]
    features_cat = [c for c in ["sexo","cor_raca","uf","nivel_instrucao","carteira_assinada","tipo_area"] if c in df_scope.columns]
    if usar_variaveis_tempo:
        if "ano" in df_scope.columns and "ano" not in features_num:
            features_num.append("ano")         # ano numérico (generaliza melhor para anos não vistos)
        if "trimestre" in df_scope.columns and "trimestre" not in features_cat:
            features_cat.append("trimestre")   # trimestre categórico (sazonalidade)
    if alvo == "ocupado" and scope == "populacao" and "eh_ti" in df_scope.columns and "eh_ti" not in features_cat:
        features_cat.append("eh_ti")
    features = features_num + features_cat

    if usar_rais_features and alvo == "ocupado" and augment_pnad_with_rais is not None:
        try:
            df_scope = augment_pnad_with_rais(df_scope, rais_dir=rais_dir)
            rais_feats = [c for c in ['taxa_ativo_3112_rais','n_vinculos_rais','salario_median_rais'] if c in df_scope.columns]
            for c in rais_feats:
                if c not in features_num:
                    features_num.append(c)
            features = features_num + features_cat
            print(f"[RAIS] Features adicionadas: {rais_feats}")
        except Exception as e:
            print(f"[RAIS] Aviso: falha ao integrar features da RAIS ({e}). Prosseguindo sem RAIS.")
    else:
        if usar_rais_features and alvo != "ocupado":
            print("[RAIS] Integração pulada (apenas para alvo='ocupado').")

    # Amostragem manual (se solicitada) após a automática
    if n_amostra is not None and len(df_scope) > n_amostra:
        if amostragem_por_tempo and {"ano","trimestre"}.issubset(df_scope.columns):
            df_scope, _, _ = _stratified_downsample_multi(df_scope, ["ano","trimestre"], "y", n_amostra, random_state=42)
        else:
            frac = n_amostra / len(df_scope)
            df_scope = df_scope.groupby("y", group_keys=False).apply(
                lambda g: g.sample(frac=frac, random_state=42) if len(g) > 0 else g
            ).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"[info] Amostragem estratificada manual aplicada: {len(df_scope):,} registros")

    # Preparar X, y, pesos
    X_all = df_scope[features].reset_index(drop=True).copy()
    y_all = df_scope["y"].astype(int).reset_index(drop=True).copy()
    amostra_utilizada = len(X_all)
    print(f"Registros efetivos no modelo (após filtros/amostra): {amostra_utilizada:,}")

    sample_weight_all = None
    if usar_peso and "peso_populacional" in df_scope.columns:
        sw = pd.to_numeric(df_scope["peso_populacional"], errors="coerce").replace([np.inf,-np.inf], np.nan)
        if (sw > 0).any():
            sw = sw.fillna(0.0)
            sw = sw.where(sw > 0, other=sw[sw > 0].median())
            sw = sw / sw.mean()
            sample_weight_all = sw.reset_index(drop=True)

    # Define preprocessor
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

    # Define standard RF parameters
    rf_params = {
        "n_estimators": rf_n_estimators,
        "max_depth": rf_max_depth,
        "min_samples_leaf": rf_min_samples_leaf,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": rf_verbose
    }
    rf = RandomForestClassifier(**rf_params)
    
    # Create basic pipeline first (we'll update it after train/test split if needed)
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", rf)])

    # Split the data
    if usar_peso and sample_weight_all is not None:
        X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
            X_all, y_all, sample_weight_all, test_size=0.25, random_state=42, 
            stratify=y_all if y_all.nunique() > 1 else None
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.25, random_state=42, 
            stratify=y_all if y_all.nunique() > 1 else None
        )
        sw_train, sw_test = None, None
        
    # NOW we can check for imbalance after y_train is defined
    if alvo == "ocupado" and y_train.mean() > 0.75:  # Se desbalanceado (>75% positivos)
        print(f"[info] Aplicando técnicas de balanceamento para lidar com classes desbalanceadas ({y_train.mean():.1%} positivos)")
        
        # Defina uma estratégia de balanceamento
        sampling_strategy = 0.5  # Proporção desejada de minoria:maioria
        
        if amostra_utilizada > 1_000_000:
            # Para datasets muito grandes, usar apenas under-sampling da maioria
            print("[info] Dataset grande: usando apenas under-sampling")
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
            
            if sw_train is not None:
                print("[aviso] Pesos amostrais serão recalculados após balanceamento")
            
            # Pipeline com balanceamento integrado
            pipe = ImbPipeline([
                ('prep', preprocessor),
                ('sampler', sampler),
                ('clf', rf)
            ])
        else:
            # Para datasets menores, usar combinação de over e under sampling
            print("[info] Dataset médio: usando over e under sampling combinados")
            # Primeiro SMOTE para aumentar a minoria, depois under-sampling para reduzir maioria
            sampler = ImbPipeline([
                ('over', SMOTE(sampling_strategy=0.3, random_state=42)),
                ('under', RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42))
            ])
            
            pipe = ImbPipeline([
                ('prep', preprocessor),
                ('sampler', sampler),
                ('clf', rf)
            ])
    # Split (estratificado ou temporal)
    if split_temporal and "ano" in df_scope.columns:
        ano_max = int(df_scope["ano"].max()) if ano_teste is None else int(ano_teste)
        mask_test = (df_scope["ano"] == ano_max)
        X_train = X_all.loc[~mask_test].reset_index(drop=True)
        X_test = X_all.loc[mask_test].reset_index(drop=True)
        y_train = y_all.loc[~mask_test].reset_index(drop=True)
        y_test = y_all.loc[mask_test].reset_index(drop=True)
        if sample_weight_all is not None:
            sw_train = sample_weight_all.loc[~mask_test].reset_index(drop=True)
            sw_test = sample_weight_all.loc[mask_test].reset_index(drop=True)
        else:
            sw_train = sw_test = None
        print(f"[info] Split temporal ativado. Ano de teste: {ano_max}. Treino={len(X_train):,}, Teste={len(X_test):,}")
    else:
        if sample_weight_all is not None:
            X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
                X_all, y_all, sample_weight_all, test_size=0.25, stratify=y_all if y_all.nunique()>1 else None, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all, y_all, test_size=0.25, stratify=y_all if y_all.nunique()>1 else None, random_state=42
            )
            sw_train, sw_test = None, None

    def counts(y_arr):
        vals, cnts = np.unique(y_arr, return_counts=True)
        return {int(k): int(v) for k, v in zip(vals, cnts)}

    print("\nDistribuição de classes:")
    print("Treino:", counts(y_train))
    print("Teste :", counts(y_test))
    print(f"Taxa positiva (prevalência) no teste: {y_test.mean():.2%}")

    # Diagnóstico RF e reuso
    print(f"Configuração do RandomForest: n_estimators={rf.n_estimators}, max_depth={rf.max_depth}, min_samples_leaf={rf.min_samples_leaf}, verbose={rf_verbose}")
    modelo_path = modelos / f"rf_ti_{alvo}.joblib"
    if modelo_path.exists():
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modelo_path.stat().st_mtime))
        print(f"[diag] Arquivo de modelo já existe ({modelo_path.name}, mtime={ts}). Não será reutilizado; será sobrescrito após novo treino.")

    # Fit
    t_fit = time.perf_counter()
    if alvo == "ocupado" and y_train.mean() > 0.75:
        # Don't pass sample_weight directly when using resampling
        pipe.fit(X_train, y_train)  # Remove the clf__sample_weight parameter
    else:
        # Original code for when not using resampling
        if sw_train is not None:
            pipe.fit(X_train, y_train, clf__sample_weight=sw_train.to_numpy())
        else:
            pipe.fit(X_train, y_train)
    print(f"Tempo de treino: {time.perf_counter()-t_fit:.2f}s")

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
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc = float("nan")
    try:
        pr_auc = average_precision_score(y_test, y_proba)
    except Exception:
        pr_auc = float("nan")

    # Adicionar balanced_accuracy para melhor avaliação com classes desbalanceadas
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    print("\n--- MÉTRICAS (não ponderadas) ---")
    print(f"Acurácia: {acc:.4f}")
    print(f"Acurácia Balanceada: {bal_acc:.4f}  <-- Melhor para classes desbalanceadas")
    print(f"Precisão: {prec:.4f}")
    print(f"Recall:   {rec:.4f}")
    print(f"F1-score: {f1:.4f}")

    if sw_test is not None:
        print("\n--- MÉTRICAS (ponderadas por peso_populacional) ---")
        try:
            print(f"[W] Acurácia: {accuracy_score(y_test, y_pred, sample_weight=sw_test):.4f}")
            print(f"[W] Precisão: {precision_score(y_test, y_pred, zero_division=0, sample_weight=sw_test):.4f}")
            print(f"[W] Recall:   {recall_score(y_test, y_pred, zero_division=0, sample_weight=sw_test):.4f}")
            print(f"[W] F1-score: {f1_score(y_test, y_pred, zero_division=0, sample_weight=sw_test):.4f}")
            if not np.isnan(roc):
                print(f"[W] ROC AUC:  {roc_auc_score(y_test, y_proba, sample_weight=sw_test):.4f}")
            if not np.isnan(pr_auc):
                print(f"[W] PR AUC:   {average_precision_score(y_test, y_proba, sample_weight=sw_test):.4f}")
        except Exception as e:
            print(f"[aviso] Falha métricas ponderadas: {e}")

    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de confusão:")
    print(cm)

    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred, zero_division=0, digits=4))

    # Thresholds e limiares ótimos
    prefix = f"rf_{alvo}"
    thr_csv_path, thr_df = _threshold_report(y_test, y_proba, resultados, prefix=prefix, sample_weight=sw_test)
    
    # Escolha threshold otimizando balanced_accuracy em vez de apenas F1 para lidar com desbalanceamento
    thr_opt_row = thr_df.iloc[int(thr_df["balanced_accuracy"].values.argmax())]
    thr_opt = float(thr_opt_row["threshold"])
    
    # Também calcular threshold ótimo para F1 padrão como referência
    thr_opt_f1 = float(thr_df.iloc[int(thr_df["f1"].values.argmax())]["threshold"])
    
    if "f1_w" in thr_df.columns and thr_df["f1_w"].notna().any():
        thr_opt_w = float(thr_df.loc[thr_df["f1_w"].idxmax(), "threshold"])
    else:
        thr_opt_w = thr_opt
        
    y_pred_opt = (y_proba >= thr_opt).astype(int)
    y_pred_opt_f1 = (y_proba >= thr_opt_f1).astype(int)
    y_pred_opt_w = (y_proba >= thr_opt_w).astype(int)
    
    print(f"\nLimiar ótimo (balanced_accuracy): {thr_opt:.4f} | BalAcc@opt={balanced_accuracy_score(y_test, y_pred_opt):.4f}")
    print(f"Limiar ótimo (F1): {thr_opt_f1:.4f} | F1@opt={f1_score(y_test, y_pred_opt_f1, zero_division=0):.4f}")
    
    # Verificar desempenho por classe com o limiar balanceado
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt).ravel()
    class0_acc = tn / (tn + fp) if (tn + fp) > 0 else 0
    class1_acc = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"Acurácia da classe negativa (limiar balanceado): {class0_acc:.4f}")
    print(f"Acurácia da classe positiva (limiar balanceado): {class1_acc:.4f}")
    
    if sw_test is not None:
        print(f"Limiar ótimo (F1) ponderado: {thr_opt_w:.4f} | [W]F1@opt={f1_score(y_test, y_pred_opt_w, zero_division=0, sample_weight=sw_test):.4f}")

    # Curvas e calibração (se possível)
    generated_imgs = []
    try:
        generated_imgs += _plot_roc_pr(y_test, y_proba, graficos, prefix=prefix)
    except Exception:
        print("[aviso] ROC/PR não geradas (classe única no teste).")
    generated_imgs += _plot_confusion(cm, graficos, prefix=prefix)
    try:
        generated_imgs += _plot_calibration(y_test, y_proba, graficos, prefix=prefix)
    except Exception as e:
        print(f"[aviso] Calibração não gerada: {e}")

    # Validação cruzada — pula se dataset muito grande
    if y_all.nunique() > 1 and amostra_utilizada <= skip_cv_threshold and not split_temporal:
        print("\nValidação cruzada (StratifiedKFold=3) - ROC AUC e F1 (não ponderada):")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scoring = {"roc_auc": "roc_auc", "f1": "f1", "accuracy": "accuracy"}
        t_cv = time.perf_counter()
        cv_res = cross_validate(pipe, X_all, y_all, cv=cv, scoring=scoring, n_jobs=-1)
        print(f"Tempo CV: {time.perf_counter()-t_cv:.2f}s")
        print("ROC AUC (CV):", f"{cv_res['test_roc_auc'].mean():.4f} ± {cv_res['test_roc_auc'].std():.4f}")
        print("F1      (CV):", f"{cv_res['test_f1'].mean():.4f} ± {cv_res['test_f1'].std():.4f}")
        print("Acc     (CV):", f"{cv_res['test_accuracy'].mean():.4f} ± {cv_res['test_accuracy'].std():.4f}")
    elif y_all.nunique() <= 1:
        print("\n[aviso] CV não executada: y tem classe única no escopo atual.")
    elif split_temporal:
        print("\n[info] CV padrão suprimida (modo de avaliação temporal ativado).")
    else:
        print(f"\n[info] CV suprimida por tamanho (amostra={amostra_utilizada:,} > {skip_cv_threshold:,}).")

    # Validação de grade (opcional) — compara poucas combinações de RF
    if validar_grade:
        print("\n[grid] Validação de grade (pequena) iniciada...")
        # Subamostra para grid
        df_for_grid = df_scope.copy()
        if len(df_for_grid) > grid_val_sample_max:
            if amostragem_por_tempo and {"ano","trimestre"}.issubset(df_for_grid.columns):
                df_for_grid, _, _ = _stratified_downsample_multi(df_for_grid, ["ano","trimestre"], "y", grid_val_sample_max, random_state=123)
            else:
                frac = grid_val_sample_max / len(df_for_grid)
                df_for_grid = df_for_grid.groupby("y", group_keys=False).apply(
                    lambda g: g.sample(frac=frac, random_state=123) if len(g) > 0 else g
                ).sample(frac=1.0, random_state=123).reset_index(drop=True)
        Xg = df_for_grid[features].reset_index(drop=True)
        yg = df_for_grid["y"].astype(int).reset_index(drop=True)
        swg = None
        if usar_peso and "peso_populacional" in df_for_grid.columns:
            swg = pd.to_numeric(df_for_grid["peso_populacional"], errors="coerce").replace([np.inf,-np.inf], np.nan)
            if (swg > 0).any():
                swg = swg.fillna(0.0)
                swg = swg.where(swg > 0, other=swg[swg > 0].median())
                swg = swg / swg.mean()
        grid = [
            {"n_estimators": 200, "max_depth": None},
            {"n_estimators": 300, "max_depth": 12},
            {"n_estimators": 500, "max_depth": None}
        ]
        rows = []
        cv2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=123)
        for gparams in grid:
            rf_g = RandomForestClassifier(
                n_estimators=gparams["n_estimators"],
                max_depth=gparams["max_depth"],
                min_samples_leaf=rf_min_samples_leaf,
                class_weight=None if swg is not None else "balanced",
                random_state=42, n_jobs=-1, verbose=rf_verbose
            )
            pipe_g = Pipeline(steps=[("prep", preprocessor), ("clf", rf_g)])
            scoring = {"roc_auc": "roc_auc", "f1": "f1", "average_precision": "average_precision"}
            if swg is not None:
                cv_res = cross_validate(pipe_g, Xg, yg, cv=cv2, scoring=scoring, n_jobs=-1, fit_params={"clf__sample_weight": swg})
            else:
                cv_res = cross_validate(pipe_g, Xg, yg, cv=cv2, scoring=scoring, n_jobs=-1)
            rows.append({
                "n_estimators": gparams["n_estimators"],
                "max_depth": gparams["max_depth"],
                "roc_auc_mean": np.mean(cv_res["test_roc_auc"]),
                "roc_auc_std": np.std(cv_res["test_roc_auc"]),
                "f1_mean": np.mean(cv_res["test_f1"]),
                "f1_std": np.std(cv_res["test_f1"]),
                "ap_mean": np.mean(cv_res["test_average_precision"]),
                "ap_std": np.std(cv_res["test_average_precision"])
            })
        df_grid = pd.DataFrame(rows).sort_values(["roc_auc_mean","f1_mean","ap_mean"], ascending=False)
        path_grid = resultados / f"{prefix}_grid_validacao.csv"
        df_grid.to_csv(path_grid, index=False, sep=";")
        print("[grid] Resultados salvos em:", path_grid.resolve())
        print(df_grid.to_string(index=False))

    # Importâncias
    try:
        encoder = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["oh"] if features_cat else None
        if encoder is not None:
            cat_cols_out = encoder.get_feature_names_out(features_cat)
        else:
            cat_cols_out = np.array([], dtype=str)
        final_feature_names = list(features_num) + list(cat_cols_out)
        importances = pipe.named_steps["clf"].feature_importances_
        top_idx = np.argsort(importances)[::-1][:20]
        print("\nTop 20 importâncias de variáveis (colunas transformadas):")
        for i in top_idx:
            print(f"  {final_feature_names[i]}: {importances[i]:.4f}")
        _aggregate_categorical_importances(encoder, features_num, features_cat, importances, resultados, prefix=prefix)
    except Exception as e:
        print(f"[aviso] Não foi possível extrair importâncias: {e}")

    # Permutation importance
    try:
        X_test_pi = X_test.reset_index(drop=True).copy()
        y_test_pi = pd.Series(y_test).reset_index(drop=True)
        features_in = list(X_all.columns)
        t_pi = time.perf_counter()
        pi = permutation_importance(
            pipe, X_test_pi, y_test_pi,
            n_repeats=5, scoring="f1",
            n_jobs=-1, random_state=42
        )
        print(f"\nPermutation importance (F1) — tempo {time.perf_counter()-t_pi:.2f}s")
        pi_df = pd.DataFrame({
            "feature": features_in,
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std
        }).sort_values("importance_mean", ascending=False)
        pi_path = resultados / f"{prefix}_permutation_importance.csv"
        pi_df.to_csv(pi_path, index=False, sep=";")
        print(pi_df.head(10).to_string(index=False))
    except Exception as e:
        print(f"[aviso] Falha permutation_importance: {e}")

    # Subgrupos
    try:
        df_meta_test = df_scope.reset_index(drop=True).iloc[X_test.index]
        sg_df, sg_path = _subgroup_metrics(df_meta_test, y_test, y_proba, resultados, prefix=prefix)
        if sg_df is not None and not sg_df.empty:
            print("\nMétricas por subgrupo (amostra do teste) — primeiras linhas:")
            print(sg_df.head(10).to_string(index=False))
            foco = sg_df[sg_df["grupo"].str.contains("idade:45-55|idade:35-44", regex=True)]
            if not foco.empty:
                print("\nFoco etário (35-55):")
                print(foco.to_string(index=False))
    except Exception as e:
        print(f"[aviso] Falha métricas por subgrupo: {e}")

    # Learning curve
    try:
        generated_imgs += _learning_curve(pipe, X_all, y_all, graficos, prefix=prefix)
    except Exception as e:
        print(f"[aviso] Falha learning curve: {e}")

    # Predições do teste
    try:
        cols_keep = [c for c in ["ano","trimestre","uf","idade","sexo","nivel_instrucao","anos_estudo","horas_trabalhadas_semana","eh_ti"] if c in df_meta_test.columns]
        pred_df = df_meta_test[cols_keep].reset_index(drop=True).copy()
        pred_df["y_true"] = pd.Series(y_test).reset_index(drop=True)
        pred_df["y_proba"] = y_proba
        pred_df["y_pred_0_5"] = (y_proba >= 0.5).astype(int)
        pred_df["y_pred_opt"] = y_pred_opt
        pred_df["threshold_opt"] = thr_opt
        pred_df["y_pred_opt_w"] = y_pred_opt_w
        pred_df["threshold_opt_w"] = thr_opt_w
        pred_path = resultados / f"{prefix}_predicoes_teste.csv"
        pred_df.to_csv(pred_path, index=False, sep=";")
        print(f"\nPredições do teste salvas em: {pred_path.resolve()}")
    except Exception as e:
        print(f"[aviso] Falha ao salvar predições do teste: {e}")
        pred_path = None

    if usar_rais_features and alvo == "ocupado":
        try:
            met = validar_com_rais(pred_df, rais_dir=rais_dir, resultados_dir=resultados)
            print(f"[RAIS] Validação externa por grupos — chaves: {met.get('chaves_usadas','')}")
            print(f"[RAIS] Pearson: {met.get('corr_pearson'):.4f} | Spearman: {met.get('corr_spearman'):.4f} | MAE: {met.get('mae'):.4f} | RMSE: {met.get('rmse'):.4f}")
            print(f"[RAIS] CSV detalhado: {str((resultados / 'validacao_rais_por_grupo.csv').resolve())}")
        except Exception as e:
            print(f"[RAIS] Aviso: falha ao gerar métricas de validação com RAIS: {e}")

    # Salvar modelo e resumo
    joblib.dump(pipe, modelo_path)
    print(f"\nModelo salvo em: {modelo_path.resolve()}")

    resumo_txt = resultados / f"resumo_rf_ti_{alvo}.txt"
    with open(resumo_txt, "w", encoding="utf-8") as f:
        f.write(f"Alvo: {alvo}\n")
        f.write(f"Escopo: {scope}\n")
        f.write(f"Amostra: {len(X_all):,}\n")
        f.write("Distribuição de classes (treino/teste):\n")
        vals, cnts = np.unique(y_train, return_counts=True)
        f.write(f"  Treino: {dict(zip(map(int, vals), map(int, cnts)))}\n")
        vals, cnts = np.unique(y_test, return_counts=True)
        f.write(f"  Teste : {dict(zip(map(int, vals), map(int, cnts)))}\n\n")
        f.write("Métricas (não ponderadas):\n")
        f.write(f"  Acc: {accuracy_score(y_test, y_pred):.4f}  BalAcc: {balanced_accuracy_score(y_test, y_pred):.4f}  ")
        f.write(f"Prec: {precision_score(y_test, y_pred, zero_division=0):.4f}  Rec: {recall_score(y_test, y_pred, zero_division=0):.4f}  F1: {f1_score(y_test, y_pred, zero_division=0):.4f}  ")
        if not (np.isnan(roc) or np.isnan(pr_auc)):
            f.write(f"ROC: {roc:.4f}  PR AUC: {pr_auc:.4f}\n")
        else:
            f.write("ROC/PR AUC: n/a (classe única no teste)\n")
        if sw_test is not None:
            f.write("Métricas (ponderadas por peso_populacional):\n")
            try:
                f.write(f"  [W]Acc: {accuracy_score(y_test, y_pred, sample_weight=sw_test):.4f}  ")
                f.write(f"[W]Prec: {precision_score(y_test, y_pred, zero_division=0, sample_weight=sw_test):.4f}  ")
                f.write(f"[W]Rec: {recall_score(y_test, y_pred, zero_division=0, sample_weight=sw_test):.4f}  ")
                f.write(f"[W]F1: {f1_score(y_test, y_pred, zero_division=0, sample_weight=sw_test):.4f}\n")
            except Exception:
                f.write("  (n/a)\n")
        f.write(f"  Limiar ótimo (F1, não ponderado): {thr_opt:.4f} | F1@opt: {f1_score(y_test, y_pred_opt, zero_division=0):.4f}\n")
        if sw_test is not None:
            f.write(f"  Limiar ótimo (F1, ponderado):    {thr_opt_w:.4f} | [W]F1@opt: {f1_score(y_test, y_pred_opt_w, zero_division=0, sample_weight=sw_test):.4f}\n")
        if validar_grade:
            f.write(f"\nValidação de grade salva em: {path_grid.resolve()}\n")
        if split_temporal and "ano" in df_scope.columns:
            f.write(f"\nAvaliação temporal: ano de teste = {int(df_scope['ano'].max() if ano_teste is None else ano_teste)}\n")
        f.write("- Consulte graficos/ para ROC, PR, calibração, matriz de confusão e learning curve.\n")
        f.write("- Consulte resultados/*_threshold_metrics.csv para trade-offs de limiar.\n")
        f.write("- Consulte resultados/*_importances_aggregated.csv para importâncias por variável original.\n")
    print(f"Resumo salvo em: {resumo_txt.resolve()}")

    info_txt = resultados / f"info_leitura_{prefix}.txt"
    with open(info_txt, "w", encoding="utf-8") as f:
        f.write(f"Arquivo lido: {caminho_csv}\n")
        f.write(f"Total de registros lidos (todas as ocupações): {total_lidos}\n")
        f.write(f"Registros de TI: {ti_filtrados}\n")
        f.write(f"Registros {('TI' if scope=='ti' else 'População')} com alvo definido: {len(df_scope)}\n")
        f.write(f"Registros efetivos no modelo: {amostra_utilizada}\n")
        f.write(f"Diretório de gráficos: {graficos.resolve()}\n")
        f.write("Figuras geradas:\n")
        for p in sorted(set(generated_imgs)):
            f.write(f"  {p}\n")
        f.write(f"Tabela de thresholds: {thr_csv_path}\n")
        if pred_path:
            f.write(f"Predições do teste: {pred_path.resolve()}\n")
    print(f"Resumo de leitura salvo em: {info_txt.resolve()}")

    if generated_imgs:
        print("\nFiguras geradas:")
        for p in sorted(set(generated_imgs)):
            print(" -", p)

    print(f"\nTempo total: {time.perf_counter()-t0:.2f}s")

    return {
        "alvo": alvo,
        "graficos_dir": str(graficos.resolve()),
        "resultados_dir": str(resultados.resolve()),
        "modelos_dir": str(modelos.resolve()),
        "modelo_path": str(modelo_path.resolve()),
        "resumo_path": str(resumo_txt.resolve()),
        "info_path": str(info_txt.resolve())
    }


def run_alvo_unico(
    caminho_csv=r"C:\TCC\dados\pnad\dados_pnad_consolidados.csv",
    alvo="salario_6sm",
    n_amostra=None,
    usar_peso=True,
    rf_verbose=0,
    limite_auto_pop_ocupado=2_000_000,
    rf_n_estimators=500,
    rf_max_depth=None,
    rf_min_samples_leaf=5,
    amostragem_por_tempo=True,
    usar_variaveis_tempo=True,
    split_temporal=False,
    validar_grade=False
):
    print(f"\n[run_alvo_unico] Rodando alvo único: {alvo}...")
    return rf_ti_6sm(
        caminho_csv=caminho_csv,
        n_amostra=n_amostra,
        alvo=alvo,
        usar_peso=usar_peso,
        usar_subpastas_por_alvo=True,
        rf_verbose=rf_verbose,
        escopo_ocupado="auto",
        limite_auto_pop_ocupado=limite_auto_pop_ocupado,
        rf_n_estimators=rf_n_estimators,
        rf_max_depth=rf_max_depth,
        rf_min_samples_leaf=rf_min_samples_leaf,
        amostragem_por_tempo=amostragem_por_tempo,
        usar_variaveis_tempo=usar_variaveis_tempo,
        split_temporal=split_temporal,
        validar_grade=validar_grade
    )


def run_dois_alvos(*args, **kwargs):
    """Compat: wrapper antigo. Agora apenas encaminha para run_alvo_unico."""
    print("[aviso] run_dois_alvos agora executa apenas um alvo. Use run_alvo_unico.")
    return run_alvo_unico(*args, **kwargs)


if __name__ == "__main__":
    # Ajuste rf_verbose=1 para ver “building tree X of Y”
    run_alvo_unico(
        caminho_csv=r"C:\TCC\dados\pnad\dados_pnad_consolidados.csv",
        alvo="salario_6sm",
        n_amostra=None,
        usar_peso=True,
        rf_verbose=1,
        limite_auto_pop_ocupado=2_000_000,
        rf_n_estimators=500,
        rf_max_depth=None,
        rf_min_samples_leaf=5,
        amostragem_por_tempo=True,
        usar_variaveis_tempo=True,
        split_temporal=False,
        validar_grade=False
    )


