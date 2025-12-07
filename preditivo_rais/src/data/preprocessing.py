from typing import Tuple, Optional

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.feature_processing import build_preprocessor, _clean_col

def _clean_feature_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.replace(to_replace=r".*\{.*", value=np.nan, regex=True, inplace=True)
    obj_cols = df.select_dtypes(include=[object]).columns.tolist()
    for c in obj_cols:
        try:
            df[c] = df[c].astype(str).str.strip()
        except Exception:
            continue

    numeric_like = []
    pattern = re.compile(r'^[0-9\.,\-\s]+$')
    for c in obj_cols:
        ser = df[c].dropna().astype(str)
        if ser.empty:
            continue
        sample = ser.sample(n=min(len(ser), 1000), random_state=0)
        frac_numeric_like = sample.apply(lambda x: bool(pattern.match(x))).mean()
        if frac_numeric_like >= 0.6:
            numeric_like.append(c)

    for c in numeric_like:
        new = df[c].astype(str).str.replace(r'\s+', '', regex=True)
        new = new.str.replace(',', '.', regex=False)
        new = new.str.replace(r'^0+(\d)', r'\1', regex=True)
        df[c] = pd.to_numeric(new.replace({'': np.nan}), errors='coerce')

    return df

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    cleaned_data = data.drop_duplicates()
    leakage_features = [
        'mes_desligamento', 'motivo_desligamento', 'causa_afastamento_1',
        'causa_afastamento_2', 'causa_afastamento_3', 'qtd_dias_afastamento',
        'tipo_admissao',
        'vl_remun_dezembro_nom', 'vl_remun_dezembro__sm_', 'faixa_remun_dezem__sm_',
        'vl_remun_media_nom', 'vl_remun_media__sm_', 'faixa_remun_media__sm_',
        'vl_rem_janeiro_cc', 'vl_rem_fevereiro_cc', 'vl_rem_marco_cc', 'vl_rem_abril_cc',
        'vl_rem_maio_cc', 'vl_rem_junho_cc', 'vl_rem_julho_cc', 'vl_rem_agosto_cc',
        'vl_rem_setembro_cc', 'vl_rem_outubro_cc', 'vl_rem_novembro_cc'
    ]
    existing_leakage = [col for col in leakage_features if col in cleaned_data.columns]
    if existing_leakage:
        print(f"Removing {len(existing_leakage)} potential leakage features: {existing_leakage}")
        cleaned_data = cleaned_data.drop(columns=existing_leakage)
    rem_patterns = ['vl_rem', 'vl_remun', 'faixa_remun', 'vl_salario', 'faixa_remun']
    to_drop = [c for c in cleaned_data.columns if any(p in c.lower() for p in rem_patterns)]
    if to_drop:
        print(f"Also removing {len(to_drop)} remuneration-like columns: {to_drop}")
        cleaned_data = cleaned_data.drop(columns=to_drop)
    return cleaned_data

def preprocess_data(
    data: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: bool = True,
    sample_frac: Optional[float] = None,
    return_preprocessor: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Optional[object]]:
    if data is None or data.empty:
        raise ValueError("Empty input data")

    df = clean_data(data)
    df = _clean_feature_values(df)

    cleaned_target = _clean_col(target_column)
    if cleaned_target not in df.columns:
        raise KeyError(f"Target column '{target_column}' (cleaned '{cleaned_target}') not found in data")

    sex_col = 'sexo_trabalhador'
    if sex_col in df.columns:
        try:
            s = df[sex_col].astype(str).str.strip().str.lower()
            mapping = {
                'm': 1, 'masculino': 1, 'homem': 1, 'male': 1, 'masc': 1,
                'f': 0, 'feminino': 0, 'mulher': 0, 'female': 0, 'fem': 0,
                '1': 1, '01': 1, '2': 0, '02': 0, '0': 0
            }
            df['sexo_trabalhador_bin'] = s.map(mapping)
            if df['sexo_trabalhador_bin'].isna().any():
                coerced = pd.to_numeric(df[sex_col], errors='coerce')
                if coerced.notna().any():
                    df.loc[df['sexo_trabalhador_bin'].isna() & (coerced == 1), 'sexo_trabalhador_bin'] = 1
                    df.loc[df['sexo_trabalhador_bin'].isna() & (coerced == 2), 'sexo_trabalhador_bin'] = 0
            df = df.drop(columns=[sex_col])
        except Exception:
            pass

    ordinal_like = [
        'faixa_etaria',
        'faixa_tempo_emprego',
        'escolaridade_apos_2005',
    ]
    for col in ordinal_like:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

    initial_rows = len(df)
    df = df.dropna(subset=[cleaned_target])
    removed_rows = initial_rows - len(df)
    if removed_rows > 0:
        print(f"Removed {removed_rows} rows with NaN values in target column '{cleaned_target}'")

    if sample_frac is not None:
        if not (0.0 < sample_frac <= 1.0):
            raise ValueError("sample_frac must be in (0, 1]")
        df = df.sample(frac=sample_frac, random_state=random_state)

    X = df.drop(columns=[cleaned_target])
    y = df[cleaned_target].astype(int)

    if 'ano' in df.columns:
        stratify_col = df['ano'].astype(str) + '_' + y.astype(str)
    else:
        stratify_col = y

    stratify_param = stratify_col if stratify and stratify_col.nunique() > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    preprocessor, numeric_cols, low_card_cols, high_card_cols = build_preprocessor(X_train)
    preprocessor.fit(X_train)

    X_train_proc = preprocessor.transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    if return_preprocessor:
        return X_train_proc, y_train, X_test_proc, y_test, preprocessor
    return X_train_proc, y_train, X_test_proc, y_test, None
