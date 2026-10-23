"""Trenowanie i porównanie modeli: regresja logistyczna vs Random Forest (+ opcjonalnie XGBoost).

Wynik:
  models/<klucz>.joblib     – wytrenowane pipeline'y
  models/metrics.json       – metryki, krzywe ROC/PR, macierze pomyłek, ważność cech
  models/reference.json     – wartości referencyjne do wyjaśniania pojedynczych decyzji
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve, brier_score_loss)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import features as F

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "credit_risk_dataset.csv"
MODELS_DIR = ROOT / "models"

MODEL_NAMES = {
    "logreg": "Regresja logistyczna",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}


def _preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # starsze wersje scikit-learn
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", ohe)])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), F.MODEL_NUMERIC),
        ("cat", cat, F.MODEL_CATEGORICAL),
    ])


def build_models() -> dict[str, Pipeline]:
    models = {
        "logreg": Pipeline([
            ("preprocessor", _preprocessor(scale=True)),
            ("classifier", LogisticRegression(C=1.0, max_iter=2000)),
        ]),
        "rf": Pipeline([
            ("preprocessor", _preprocessor(scale=False)),
            ("classifier", RandomForestClassifier(
                n_estimators=300, max_depth=None, min_samples_leaf=3,
                max_features="sqrt", n_jobs=-1, random_state=42)),
        ]),
    }
    try:
        from xgboost import XGBClassifier
        # Parametry z GridSearch w notatniku trening_modelu.ipynb
        # (scale_pos_weight=1, aby prawdopodobieństwa były porównywalne z PD).
        models["xgb"] = Pipeline([
            ("preprocessor", _preprocessor(scale=False)),
            ("classifier", XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.1, min_child_weight=5,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
                objective="binary:logistic", eval_metric="logloss",
                random_state=42, n_jobs=-1)),
        ])
    except ImportError:
        pass
    return models


def _best_f1_threshold(y_true, proba) -> float:
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-9, None)
    return float(t[np.nanargmax(f1[:-1])])


def _metrics(y_true, proba, thr) -> dict:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(thr), 4),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "specificity": tn / (tn + fp),
        "f1": f1_score(y_true, pred),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _downsample(xs, ys, n=150):
    idx = np.unique(np.linspace(0, len(xs) - 1, min(n, len(xs))).astype(int))
    return [round(float(xs[i]), 4) for i in idx], [round(float(ys[i]), 4) for i in idx]


def raw_permutation_importance(model, X_raw: pd.DataFrame, y, n_repeats=3, seed=42) -> dict:
    """Permutacyjna ważność SUROWYCH cech wniosku (spadek ROC-AUC po wymieszaniu).

    Cechy pochodne są przeliczane po permutacji, więc wynik pokazuje wpływ całej
    informacji niesionej przez daną daną wejściową.
    """
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(F.add_features(X_raw))[:, 1])
    out = {}
    for col in F.RAW_FEATURES:
        drops = []
        for _ in range(n_repeats):
            Xp = X_raw.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            drops.append(base - roc_auc_score(y, model.predict_proba(F.add_features(Xp))[:, 1]))
        out[col] = float(np.mean(drops))
    return out


def train_all(data_path: Path = DATA_PATH, models_dir: Path = MODELS_DIR, log=print) -> dict:
    models_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(data_path)
    df = F.clean_training_data(raw)
    X_raw = df[F.RAW_FEATURES]
    y = df[F.TARGET].astype(int).values

    Xr_train, Xr_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=12, stratify=y)
    X_train, X_test = F.add_features(Xr_train), F.add_features(Xr_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "rows_raw": int(len(raw)), "rows_clean": int(len(df)),
            "train": int(len(y_train)), "test": int(len(y_test)),
            "default_rate": float(y.mean()),
        },
        "models": {},
    }

    for key, model in build_models().items():
        t0 = time.time()
        log(f"[{key}] walidacja krzyżowa (5-fold)…")
        oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        thr = _best_f1_threshold(y_train, oof)
        log(f"[{key}] trening na pełnym zbiorze uczącym…")
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        fpr_s, tpr_s = _downsample(fpr, tpr)
        rec_s, prec_s = _downsample(rec[::-1], prec[::-1])

        log(f"[{key}] ważność cech (permutacyjna)…")
        imp = raw_permutation_importance(model, Xr_test, y_test)

        entry = {
            "name": MODEL_NAMES[key],
            "cv_roc_auc": roc_auc_score(y_train, oof),
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "brier": brier_score_loss(y_test, proba),
            "at_050": _metrics(y_test, proba, 0.5),
            "at_best_f1": _metrics(y_test, proba, thr),
            "roc_curve": {"fpr": fpr_s, "tpr": tpr_s},
            "pr_curve": {"recall": rec_s, "precision": prec_s},
            "permutation_importance": imp,
            "train_seconds": round(time.time() - t0, 1),
        }
        if key == "logreg":
            pre = model.named_steps["preprocessor"]
            coefs = model.named_steps["classifier"].coef_[0]
            entry["coefficients"] = sorted(
                [{"feature": n, "coef": float(c)} for n, c in zip(pre.get_feature_names_out(), coefs)],
                key=lambda d: -abs(d["coef"]))
        results["models"][key] = entry
        joblib.dump(model, models_dir / f"{key}.joblib")
        log(f"[{key}] ROC-AUC={entry['roc_auc']:.3f}  ({entry['train_seconds']} s)")

    # wartości referencyjne: "typowy" wnioskodawca ze zbioru uczącego
    reference = {c: float(Xr_train[c].median()) for c in F.RAW_NUMERIC}
    reference.update({c: str(Xr_train[c].mode()[0]) for c in F.RAW_CATEGORICAL})
    (models_dir / "reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")

    best = max(results["models"], key=lambda k: results["models"][k]["roc_auc"])
    results["best_by_roc_auc"] = best
    (models_dir / "metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    log(f"Gotowe. Najlepszy model wg ROC-AUC: {MODEL_NAMES[best]}")
    return results
