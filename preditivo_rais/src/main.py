import os
import sys
import time
import yaml

from src.data.loader import load_data
from src.data.preprocessing import preprocess_data
from src.models.random_forest import RandomForestModel
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, balanced_accuracy_score,
    roc_curve, precision_recall_curve
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLOTS_AVAILABLE = True
except ImportError:
    PLOTS_AVAILABLE = False

def _now():
    return time.strftime('%Y-%m-%d %H:%M:%S')

def main():
    print(f"[{_now()}] Starting main()", flush=True)
    
    with open(os.path.join('config', 'params.yaml'), 'r') as file:
        params = yaml.safe_load(file)
    
    data_path = params['data'].get('train_data_path') or params['data'].get('path')
    patterns = params['data'].get('patterns')
    sample_frac = params['data'].get('sample_frac')
    file_limit = params['data'].get('file_limit')

    print(f"[{_now()}] Loading data from {data_path}", flush=True)
    t_load0 = time.perf_counter()
    data = load_data(data_path, patterns=patterns, file_limit=file_limit)
    t_load = time.perf_counter() - t_load0
    print(f"[{_now()}] Data loaded rows={len(data)} cols={len(data.columns)} — time={t_load:.2f}s", flush=True)

    print(f"[{_now()}] Preprocessing data", flush=True)
    t0 = time.perf_counter()
    X_train, y_train, X_test, y_test, preprocessor = preprocess_data(
        data,
        target_column=params['data']['target_column'],
        test_size=params['data'].get('test_size', 0.2),
        random_state=params['model'].get('random_state'),
        stratify=params['data'].get('stratify', True),
        sample_frac=sample_frac,
        return_preprocessor=True,
    )
    t_pre = time.perf_counter() - t0
    print(f"[{_now()}] Preprocessing complete. X_train: {X_train.shape}, X_test: {X_test.shape} — time={t_pre:.2f}s", flush=True)

    model_path = os.path.join('artifacts', 'model.joblib')
    retrain_needed = True
    model = None
    if os.path.exists(model_path):
        try:
            loaded = joblib.load(model_path)
            impl = getattr(loaded, 'impl', None)
            importances = getattr(impl, 'feature_importances_', None)
            feature_names = None
            if preprocessor is not None:
                try:
                    feature_names = preprocessor.get_feature_names_out()
                except Exception:
                    feature_names = None

            if importances is not None and feature_names is not None:
                if len(importances) == len(feature_names):
                    model = loaded
                    retrain_needed = False
        except Exception:
            pass

    if retrain_needed:
        model = RandomForestModel(
            n_estimators=params['model']['n_estimators'],
            max_depth=params['model']['max_depth'],
            random_state=params['model'].get('random_state'),
        )

        if preprocessor is not None:
            os.makedirs('artifacts', exist_ok=True)
            joblib.dump(preprocessor, os.path.join('artifacts', 'preprocessor.joblib'))

        print(f"[{_now()}] Training model", flush=True)
        t_train0 = time.perf_counter()
        model.fit(X_train, y_train)
        t_train = time.perf_counter() - t_train0
        print(f"[{_now()}] Model training finished — time={t_train:.2f}s", flush=True)
        joblib.dump(model, model_path)

    if hasattr(model.impl, 'feature_importances_'):
        feature_names = preprocessor.get_feature_names_out()
        importances = model.impl.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        print(f"[{_now()}] Top 10 most important features:", flush=True)
        for idx, row in feature_importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}", flush=True)
        feature_importance_df.to_csv("artifacts/feature_importance.csv", index=False)

    print(f"[{_now()}] Computing metrics", flush=True)
    predictions = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else predictions

    acc = accuracy_score(y_test, predictions)
    bal_acc = balanced_accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, zero_division=0)
    rec = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = float('nan')
    try:
        pr_auc = average_precision_score(y_test, y_proba)
    except Exception:
        pr_auc = float('nan')
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"[{_now()}] Metrics:", flush=True)
    print(f"  Accuracy: {acc:.4f}", flush=True)
    print(f"  Balanced Accuracy: {bal_acc:.4f}", flush=True)
    print(f"  Precision: {prec:.4f}", flush=True)
    print(f"  Recall: {rec:.4f}", flush=True)
    print(f"  F1 Score: {f1:.4f}", flush=True)
    print(f"  ROC-AUC: {roc_auc:.4f}", flush=True)
    print(f"  PR-AUC: {pr_auc:.4f}", flush=True)
    print(f"  Specificity: {specificity:.4f}", flush=True)
    print(f"  NPV: {npv:.4f}", flush=True)
    print(f"  Sensitivity: {sensitivity:.4f}", flush=True)

    print(f"[{_now()}] Classification report:\n{classification_report(y_test, predictions, zero_division=0)}", flush=True)

    if PLOTS_AVAILABLE:
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            precision, recall_vals, _ = precision_recall_curve(y_test, y_proba)
            os.makedirs('artifacts', exist_ok=True)
            plt.figure()
            plt.plot(fpr, tpr, label='ROC curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc='lower right')
            plt.savefig('artifacts/roc_curve.png')
            plt.close()

            plt.figure()
            plt.plot(recall_vals, precision, label='PR curve')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc='lower left')
            plt.savefig('artifacts/pr_curve.png')
            plt.close()
            print(f"[{_now()}] Saved ROC and PR curves to artifacts/", flush=True)
        except Exception:
            pass

    print(f"[{_now()}] Finished main()", flush=True)

if __name__ == '__main__':
    main()
