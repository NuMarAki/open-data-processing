# -*- coding: utf-8 -*-
"""
Predição com base em amostra PNAD (amostra_pnad.csv):
- alvo in {'ocupado','salario_6sm','carteira','horas44','eh_ti_demografico'}

Saídas em resultados/pnad/{alvo}/:
- modelo.joblib
- pnad_{alvo}_predicoes_teste.csv
- resumo.txt
"""
import argparse
from pathlib import Path
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import joblib

from typing import Tuple, List, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier

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

def _carregar_salarios_minimos(caminho: Optional[str]) -> dict:
    if caminho and Path(caminho).exists():
        sm = pd.read_csv(caminho)
        anos = pd.to_numeric(sm['ano'], errors='coerce')
        vals = pd.to_numeric(sm['salario'], errors='coerce')
        return dict(zip(anos, vals))
    # fallback mínimo (ajuste conforme necessário)
    return {2012: 622, 2013: 678, 2014: 724, 2015: 788, 2016: 880, 2017: 937,
            2018: 954, 2019: 998, 2020: 1045, 2021: 1100, 2022: 1212, 2023: 1320, 2024: 1412}

def carregar_pnad(amostra_path: Optional[str] = None, pnad_dir: Optional[str] = None, sample_frac: Optional[float] = None) -> pd.DataFrame:
    """Carrega PNAD de arquivo único OU diretório de arquivos preprocessados"""
    
    # Modo 1: Arquivo único (amostra)
    if amostra_path and Path(amostra_path).exists():
        print(f"[pnad] Carregando de arquivo único: {amostra_path}")
        df = pd.read_csv(amostra_path, sep=';', dtype=str, low_memory=False, encoding='utf-8')
    
    # Modo 2: Diretório de preprocessados (igual RAIS)
    elif pnad_dir and Path(pnad_dir).exists():
        print(f"[pnad] Carregando de diretório: {pnad_dir}")
        import glob
        paths = sorted(glob.glob(os.path.join(pnad_dir, "PNADC_*_preprocessado.csv")))
        if not paths:
            raise FileNotFoundError(f"Nenhum arquivo PNADC_*_preprocessado.csv encontrado em {pnad_dir}")
        
        print(f"[pnad] Encontrados {len(paths)} arquivos")
        df_parts = []
        for i, p in enumerate(paths, 1):
            print(f"[{i}/{len(paths)}] Lendo: {os.path.basename(p)}")
            raw = pd.read_csv(p, sep=';', dtype=str, low_memory=False, encoding='utf-8')
            print(f"    linhas: {len(raw):,}")
            df_parts.append(raw)
        
        print(f"[pnad] Concatenando {len(df_parts)} arquivos...")
        df = pd.concat(df_parts, ignore_index=True)
        print(f"[pnad] Total: {len(df):,} linhas")
        del df_parts
        import gc
        gc.collect()
        
        # Amostragem opcional (igual RAIS)
        if sample_frac and 0.0 < sample_frac < 1.0:
            original = len(df)
            df = df.sample(frac=sample_frac, random_state=42)
            print(f"[pnad] Amostragem: {original:,} -> {len(df):,} linhas ({sample_frac*100:.1f}%)")
    else:
        raise FileNotFoundError(f"Nem amostra ({amostra_path}) nem diretório ({pnad_dir}) encontrados")
    
    # Normalização de tipos (comum a ambos os modos)
    to_num = [
        "ano","trimestre","idade","anos_estudo","rendimento_trabalho_principal","rendimento_bruto_mensal",
        "horas_trabalhadas_semana","sexo","cor_raca","ocupado","forca_trabalho","peso_populacional",
        "nivel_instrucao","uf","carteira_assinada"
    ]
    for c in to_num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    
    # eh_ti -> 0/1
    if "eh_ti" in df.columns:
        if df["eh_ti"].dtype == "object":
            df["eh_ti"] = df["eh_ti"].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
        df["eh_ti"] = pd.to_numeric(df["eh_ti"], errors="coerce").fillna(0).astype(int)
    
    return df

def _renda_uso(df: pd.DataFrame) -> pd.Series:
    # preferência: rendimento_trabalho_principal; fallback: rendimento_bruto_mensal
    s = pd.Series(np.nan, index=df.index, dtype='Float64')
    if "rendimento_trabalho_principal" in df.columns:
        s = s.fillna(pd.to_numeric(df["rendimento_trabalho_principal"], errors="coerce"))
    if "rendimento_bruto_mensal" in df.columns:
        s = s.fillna(pd.to_numeric(df["rendimento_bruto_mensal"], errors="coerce"))
    return s

def preparar_xy(df: pd.DataFrame, alvo: str, salarios_minimos_map: Optional[dict], n_sm: int) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    df = df.copy()
    y = None

    if alvo == "ocupado":
        if "ocupado" not in df.columns:
            raise ValueError("Coluna 'ocupado' não encontrada.")
        y = (df["ocupado"] == 1).astype(int)

    elif alvo == "salario_6sm":
        sm_map = salarios_minimos_map or _carregar_salarios_minimos(None)
        if "ano" in df.columns:
            df["salario_minimo_ano"] = df["ano"].map(sm_map)
        else:
            df["salario_minimo_ano"] = np.nan
        renda = _renda_uso(df)
        # Criar máscara válida: onde ambos renda e salario_minimo_ano não são NaN
        mask_valido = renda.notna() & df["salario_minimo_ano"].notna()
        # Inicializar y com 0 (não atende critério)
        y = pd.Series(0, index=df.index, dtype=int)
        # Onde válido, verificar se atende critério de >= n_sm * salario_minimo
        y[mask_valido] = (renda[mask_valido] >= n_sm * df.loc[mask_valido, "salario_minimo_ano"]).astype(int)

    elif alvo == "carteira":
        if "carteira_assinada" not in df.columns:
            raise ValueError("Coluna 'carteira_assinada' não encontrada.")
        # opcional: restringir a ocupados
        # df = df[df["ocupado"] == 1] if "ocupado" in df.columns else df
        y = (df["carteira_assinada"] == 1).astype(int)

    elif alvo == "horas44":
        if "horas_trabalhadas_semana" not in df.columns:
            raise ValueError("Coluna 'horas_trabalhadas_semana' não encontrada.")
        y = (df["horas_trabalhadas_semana"] >= 44).astype(int)

    elif alvo == "eh_ti_demografico":
        if "eh_ti" not in df.columns:
            raise ValueError("Coluna 'eh_ti' não encontrada.")
        y = pd.to_numeric(df["eh_ti"], errors="coerce").fillna(0).astype(int)

    else:
        raise ValueError("alvo inválido")

    # Seleção de features
    features_num = [c for c in ["idade","anos_estudo","horas_trabalhadas_semana","ano"] if c in df.columns]
    features_cat = [c for c in ["sexo","cor_raca","uf","nivel_instrucao","tipo_area","trimestre","cbo_familia","grupo_amostra"] if c in df.columns]

    # Evitar vazamentos
    drops = set()
    if alvo == "salario_6sm":
        drops |= {"rendimento_trabalho_principal","rendimento_bruto_mensal","salario_minimo_ano"}
    if alvo == "carteira":
        drops |= {"carteira_assinada"}
    if alvo == "eh_ti_demografico":
        drops |= {"eh_ti","cbo_ocupacao","cbo_familia"}
    # nunca incluir a própria variável alvo se estiver na lista
    drops |= {alvo}

    features_num = [c for c in features_num if c not in drops]
    features_cat = [c for c in features_cat if c not in drops]

    X = df[features_num + features_cat].copy()
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    return X, y.astype(int), features_num, features_cat

def treinar_avaliar(X, y, features_num, features_cat, out_dir: Path, prefix: str, sample_weight: Optional[pd.Series] = None, fast: bool = False, n_boot: int = 50, n_repeats: int = 5):
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

    # Modo rápido: menos árvores, profundidade limitada, leaf maior
    if fast:
        n_estimators = 100
        max_depth = 15
        min_samples_leaf = 10
    else:
        n_estimators = 500
        max_depth = None
        min_samples_leaf = 5

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1,
        random_state=42
    )

    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    if sample_weight is not None:
        X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
            X, y, sample_weight, test_size=0.25, random_state=42, stratify=y if y.nunique()>1 else None
        )
        pipe.fit(X_train, y_train, clf__sample_weight=sw_train.to_numpy())
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y if y.nunique()>1 else None
        )
        sw_test = None
        pipe.fit(X_train, y_train)

    classes_ = getattr(pipe.named_steps["clf"], "classes_", np.array([0,1]))
    if len(classes_) == 1:
        cls = int(classes_[0])
        y_proba = np.full(len(X_test), 1.0 if cls == 1 else 0.0, dtype=float)
    else:
        col1 = int(np.where(classes_ == 1)[0][0]) if 1 in classes_ else int(np.argmax(classes_))
        y_proba = pipe.predict_proba(X_test)[:, col1]
    y_pred = (y_proba >= 0.5).astype(int)

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

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "modelo.joblib")

    # Salvar predições com colunas meta básicas se existirem
    meta_cols = [c for c in ["ano","trimestre","uf","idade","sexo","nivel_instrucao","anos_estudo","horas_trabalhadas_semana","eh_ti"] if c in X_test.columns]
    pred_df = X_test[meta_cols].reset_index(drop=True).copy()
    pred_df["y_true"] = y_test.reset_index(drop=True)
    pred_df["y_proba"] = y_proba
    pred_df["y_pred_0_5"] = y_pred
    pred_df.to_csv(out_dir / f"{prefix}_predicoes_teste.csv", index=False, sep=";")

    with open(out_dir / "resumo.txt", "w", encoding="utf-8") as f:
        f.write(f"Amostra total: {len(X):,}\n")
        f.write(f"Treino/Teste: {len(X_train):,}/{len(X_test):,}\n")
        vals, cnts = np.unique(y_train, return_counts=True)
        f.write(f"Distribuição treino: {dict(zip(map(int, vals), map(int, cnts)))}\n")
        vals, cnts = np.unique(y_test, return_counts=True)
        f.write(f"Distribuição teste: {dict(zip(map(int, vals), map(int, cnts)))}\n\n")
        f.write("Métricas (não ponderadas):\n")
        f.write(f"  Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}\n")
        if not np.isnan(roc):
            f.write(f"  ROC AUC: {roc:.4f}\n")
        if not np.isnan(pr_auc):
            f.write(f"  PR AUC: {pr_auc:.4f}\n")

        if sw_test is not None:
            try:
                f.write("\nMétricas (ponderadas por peso_populacional):\n")
                f.write(f"  [W]Acc: {accuracy_score(y_test, y_pred, sample_weight=sw_test):.4f}  ")
                f.write(f"[W]Prec: {precision_score(y_test, y_pred, zero_division=0, sample_weight=sw_test):.4f}  ")
                f.write(f"[W]Rec: {recall_score(y_test, y_pred, zero_division=0, sample_weight=sw_test):.4f}  ")
                f.write(f"[W]F1: {f1_score(y_test, y_pred, zero_division=0, sample_weight=sw_test):.4f}\n")
            except Exception:
                pass

    # Passos pesados controlados por flags: importância por permutação e bootstrap
    try:
        from sklearn.inspection import permutation_importance
        if not fast:
            # Importância por permutação no espaço transformado é pesada; usamos apenas features numéricas+categorias nomeadas
            oh = pipe.named_steps["prep"].named_transformers_["cat"].named_steps["oh"]
            num_cols = features_num
            print("[pnad] Importância por permutação (", n_repeats, " repetições )...")
            perm = permutation_importance(pipe.named_steps["clf"], pipe.named_steps["prep"].transform(X_test), y_test, n_repeats=n_repeats, random_state=42, n_jobs=-1)
            imp_mean = getattr(perm, "importances_mean", None)
            if imp_mean is None and isinstance(perm, dict):
                imp_mean = perm.get("importances_mean")
            imp_arr = np.asarray(imp_mean) if imp_mean is not None else np.empty(0)
            # Tentativa de nomes: pode falhar se ohe não fornecer nomes; salvamos índices mesmo assim
            try:
                feat_names = oh.get_feature_names_out(features_cat).tolist() + num_cols
            except Exception:
                feat_names = [f"f{i}" for i in range(len(imp_arr))]
            imp = pd.DataFrame({"feature": feat_names, "importance_mean": imp_arr})
            imp.sort_values("importance_mean", ascending=False).to_csv(out_dir / "feature_importance_permutation.csv", index=False)
            print("[pnad] Importância por permutação salva.")
        else:
            print("[pnad] Modo rápido: pulando importância por permutação.")
    except Exception as e:
        print(f"[pnad] Aviso: falha na importância por permutação: {e}")

    if not fast:
        print("[pnad] Validação por bootstrap (", n_boot, " amostras )...")
        try:
            rng = np.random.default_rng(42)
            aucs = []
            for i in range(n_boot):
                idx = rng.choice(len(y_test), size=len(y_test), replace=True)
                aucs.append(roc_auc_score(y_test.iloc[idx], pd.Series(y_proba).iloc[idx]))
            pd.Series(aucs).to_csv(out_dir / "bootstrap_auc.csv", index=False)
            print("[pnad] Bootstrap AUC salvo.")
        except Exception as e:
            print(f"[pnad] Aviso: falha no bootstrap: {e}")
    else:
        print("[pnad] Modo rápido: pulando bootstrap.")

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "roc": roc, "pr_auc": pr_auc}

def _normalizar_pesos(df: pd.DataFrame) -> Optional[pd.Series]:
    if "peso_populacional" not in df.columns:
        return None
    sw = pd.to_numeric(df["peso_populacional"], errors="coerce").replace([np.inf,-np.inf], np.nan)
    if (sw > 0).any():
        sw = sw.fillna(0.0)
        sw = sw.where(sw > 0, other=sw[sw > 0].median())
        sw = sw / sw.mean()
        return sw
    return None

def main():
    ap = argparse.ArgumentParser(description="Predição PNAD - arquivo único ou diretório de preprocessados")
    ap.add_argument("--amostra", default=None, help="Caminho para amostra_pnad.csv (sep=';') - OPCIONAL")
    ap.add_argument("--pnad-dir", default=os.path.join("dados", "pnad", "preprocessados"), help="Diretório com PNADC_*_preprocessado.csv")
    ap.add_argument("--sample-frac", type=float, default=None, help="Fração de amostragem (ex: 0.5 = 50%%). Use None para todos os dados. Padrão: None (100%%)")
    ap.add_argument("--alvo", choices=["ocupado","salario_6sm","carteira","horas44","eh_ti_demografico"], default="salario_6sm")
    ap.add_argument("--salarios_minimos", default=None, help="CSV com colunas: ano,salario (opcional)")
    ap.add_argument("--n_sm", type=int, default=6, help="Múltiplos do SM para alvo salario_6sm")
    ap.add_argument("--usar_peso", action="store_true", help="Usar peso_populacional se disponível", default=True)
    ap.add_argument("--fast", action="store_true", help="Modo rápido: desativa passos pesados (permutation importance e bootstrap)")
    ap.add_argument("--n_boot", type=int, default=50, help="Número de bootstraps (modo completo)")
    ap.add_argument("--n_repeats", type=int, default=5, help="Repetições na importância por permutação (modo completo)")
    args = ap.parse_args()

    sample_msg = f"{args.sample_frac*100:.0f}%" if args.sample_frac else "100%"
    print(f"[pnad] Alvo: {args.alvo} | Peso: {args.usar_peso} | Sample: {sample_msg}")
    
    df = carregar_pnad(
        amostra_path=args.amostra,
        pnad_dir=args.pnad_dir,
        sample_frac=args.sample_frac
    )
    
    sm_map = _carregar_salarios_minimos(args.salarios_minimos) if args.alvo == "salario_6sm" else None
    X, y, features_num, features_cat = preparar_xy(df, alvo=args.alvo, salarios_minimos_map=sm_map, n_sm=args.n_sm)
    sw = _normalizar_pesos(df) if args.usar_peso else None
    if sw is not None:
        sw = sw.loc[X.index].reset_index(drop=True)

    out_dir = Path("resultados") / "pnad" / args.alvo
    prefix = f"pnad_{args.alvo}"
    metrics = treinar_avaliar(X, y, features_num, features_cat, out_dir, prefix, sample_weight=sw, fast=args.fast, n_boot=args.n_boot, n_repeats=args.n_repeats)

    print(f"[ok] Treinamento concluído. Resultados em: {out_dir.resolve()}")
    print(f"[resumo] Acc={metrics['acc']:.3f} Prec={metrics['prec']:.3f} Rec={metrics['rec']:.3f} F1={metrics['f1']:.3f} ROC={metrics['roc']:.3f} PR-AUC={metrics['pr_auc']:.3f}")

if __name__ == "__main__":
    main()

"""
Como rodar rapidamente

    Ocupado:
        python preditivo_pnad_alvos.py --amostra amostra_pnad.csv --alvo ocupado --usar_peso
    Salário alto (6SM, com salário mínimo externo opcional):
        python preditivo_pnad_alvos.py --amostra amostra_pnad.csv --alvo salario_6sm --n_sm 6 --salarios_minimos salarios_minimos.csv --usar_peso
    Carteira assinada:
        python preditivo_pnad_alvos.py --amostra amostra_pnad.csv --alvo carteira --usar_peso
    Jornada alta:
        python preditivo_pnad_alvos.py --amostra amostra_pnad.csv --alvo horas44 --usar_peso
    Eh_ti (sem CBO nas features):
        python preditivo_pnad_alvos.py --amostra amostra_pnad.csv --alvo eh_ti_demografico --usar_peso

"""
