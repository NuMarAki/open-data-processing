import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def _clean_col(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[\s\/:()\\]+", "_", name)
    name = re.sub(r"[^0-9a-zA-Z_]+", "", name)
    return name

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.maps = None

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        self.maps = []
        for col in df.columns:
            vc = df[col].value_counts(normalize=True).to_dict()
            self.maps.append(vc)
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        out_cols = []
        for i, col in enumerate(df.columns):
            s = df[col].map(self.maps[i]).fillna(0)
            out_cols.append(s.values)

        return np.vstack(out_cols).T

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features must be provided")
        return np.array(input_features)

def build_preprocessor(
    df: pd.DataFrame,
    numerical_threshold: int = None,
    low_cardinality_threshold: int = 20,
) -> Tuple[ColumnTransformer, List[str], List[str], List[str]]:
    df_cols = list(df.columns)
    numeric_candidates = df.select_dtypes(include=["number"]).columns.tolist()
    other_cols = [c for c in df_cols if c not in numeric_candidates]
    cat_cols = other_cols
    low_card_cols = [c for c in cat_cols if df[c].nunique(dropna=False) <= low_cardinality_threshold]
    high_card_cols = [c for c in cat_cols if c not in low_card_cols]
    numeric_cols = numeric_candidates

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    low_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", onehot),
    ])

    high_card_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("freq", FrequencyEncoder()),
    ])

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if low_card_cols:
        transformers.append(("low_cat", low_card_pipeline, low_card_cols))
    if high_card_cols:
        transformers.append(("high_cat", high_card_pipeline, high_card_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)
    return preprocessor, numeric_cols, low_card_cols, high_card_cols

def apply_preprocessor(preprocessor, X):
    return preprocessor.transform(X)
