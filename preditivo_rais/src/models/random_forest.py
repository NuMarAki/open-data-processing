class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        from sklearn.ensemble import RandomForestClassifier as skRF
        self.impl = skRF(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.is_cuml = False
        self.random_state = random_state

    def _to_numpy(self, X):
        try:
            import pandas as pd
        except Exception:
            pd = None
        if pd is not None and isinstance(X, (pd.DataFrame, pd.Series)):
            return X.values
        return X

    def fit(self, X, y):
        X_in = self._to_numpy(X)
        y_in = self._to_numpy(y)
        self.impl.fit(X_in, y_in)

    def predict(self, X):
        X_in = self._to_numpy(X)
        preds = self.impl.predict(X_in)
        try:
            import numpy as np
            return np.asarray(preds)
        except Exception:
            return preds

    def predict_proba(self, X):
        X_in = self._to_numpy(X)
        if hasattr(self.impl, 'predict_proba'):
            proba = self.impl.predict_proba(X_in)
            try:
                import numpy as np
                return np.asarray(proba)
            except Exception:
                return proba
        preds = self.predict(X)
        try:
            import numpy as np
            proba = np.zeros((len(preds), 2))
            proba[:, 0] = 1 - preds
            proba[:, 1] = preds
            return proba
        except Exception:
            return preds

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, state):
        self.__dict__.update(state)
