"""Scoring wniosku, polityka decyzyjna i wyjaśnianie wyniku."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from . import features as F
from .training import MODEL_NAMES, MODELS_DIR

RISK_CLASSES = [  # (górna granica PD, klasa, opis)
    (0.05, "A", "bardzo niskie"),
    (0.10, "B", "niskie"),
    (0.20, "C", "umiarkowane"),
    (0.40, "D", "podwyższone"),
    (1.01, "E", "wysokie"),
]

RECOMMENDATIONS = {
    "AKCEPTACJA": "Rekomendacja: akceptacja",
    "WERYFIKACJA": "Wymagana weryfikacja analityka",
    "ODRZUCENIE": "Rekomendacja: odrzucenie",
}


def models_available() -> bool:
    return (MODELS_DIR / "metrics.json").exists() and any(MODELS_DIR.glob("*.joblib"))


def load_metrics() -> dict:
    return json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))


def load_reference() -> dict:
    return json.loads((MODELS_DIR / "reference.json").read_text(encoding="utf-8"))


def load_model(key: str):
    return joblib.load(MODELS_DIR / f"{key}.joblib")


def available_models() -> dict[str, str]:
    return {k: v for k, v in MODEL_NAMES.items() if (MODELS_DIR / f"{k}.joblib").exists()}


def risk_class(pd_value: float) -> tuple[str, str]:
    for upper, cls, desc in RISK_CLASSES:
        if pd_value < upper:
            return cls, desc
    return "E", "wysokie"


def monthly_installment(amount: float, annual_rate_pct: float, months: int) -> float:
    r = annual_rate_pct / 100 / 12
    if months <= 0:
        return 0.0
    if r == 0:
        return amount / months
    return amount * r / (1 - (1 + r) ** -months)


def policy_flags(app: dict) -> list[str]:
    """Twarde reguły polityki kredytowej (niezależne od modelu ML)."""
    flags = []
    income_m = app["person_income"] / 12
    inst = monthly_installment(app["loan_amnt"], app["loan_int_rate"], int(app.get("loan_term_months") or 36))
    dti = inst / income_m if income_m > 0 else 1
    if dti > 0.5:
        flags.append(f"Rata ({inst:,.0f}/mies.) przekracza 50% miesięcznego dochodu (DSTI={dti:.0%}).")
    elif dti > 0.4:
        flags.append(f"Wysoki wskaźnik DSTI={dti:.0%} (rata/dochód) – zalecana ostrożność.")
    if app["cb_person_default_on_file"] == "Y":
        flags.append("W BIK odnotowano wcześniejsze zaległości w spłacie.")
    if app["person_age"] + int(app.get("loan_term_months") or 36) / 12 > 80:
        flags.append("Wiek klienta w dniu spłaty ostatniej raty przekroczy 80 lat.")
    if app["person_emp_length"] < 1:
        flags.append("Staż pracy krótszy niż 1 rok.")
    return flags


def predict_pd(model, app: dict) -> float:
    return float(model.predict_proba(F.application_to_frame(app))[:, 1][0])


def explain(model, app: dict, reference: dict) -> list[dict]:
    """Wkład każdej danej wejściowej w PD (metoda „podmiany na wartość typową”).

    Dla każdej cechy liczymy, o ile zmieniłoby się PD, gdyby klient miał w tej
    cesze wartość typową (medianę / dominantę ze zbioru uczącego).
    delta > 0  -> cecha ZWIĘKSZA ryzyko względem typowego klienta.
    """
    base = predict_pd(model, app)
    rows = []
    for col in F.RAW_FEATURES:
        alt = dict(app)
        alt[col] = reference[col]
        delta = base - predict_pd(model, alt)
        rows.append({
            "feature": col,
            "label": F.FEATURE_LABELS[col],
            "value": display_value(col, app[col]),
            "reference": display_value(col, reference[col]),
            "delta": round(delta, 4),
        })
    return sorted(rows, key=lambda r: -abs(r["delta"]))


def display_value(col, v):
    if col == "person_home_ownership":
        return F.HOME_OWNERSHIP.get(v, v)
    if col == "loan_intent":
        return F.LOAN_INTENT.get(v, v)
    if col == "cb_person_default_on_file":
        return F.DEFAULT_ON_FILE.get(v, v)
    if col in ("person_income", "loan_amnt"):
        return f"{float(v):,.0f}".replace(",", " ")
    if col == "loan_int_rate":
        return f"{float(v):.2f}%"
    if col in ("person_emp_length", "cb_person_cred_hist_length", "person_age"):
        return f"{float(v):g} lat"
    return str(v)


def score_application(app: dict, settings: dict) -> dict:
    avail = available_models()
    if not avail:
        raise RuntimeError("Brak wytrenowanych modeli – uruchom: python train_models.py")
    key = settings["active_model"] if settings["active_model"] in avail else next(iter(avail))
    model = load_model(key)
    metrics = load_metrics()
    pd_value = predict_pd(model, app)
    cls, cls_desc = risk_class(pd_value)
    low, high = float(settings["threshold_low"]), float(settings["threshold_high"])
    if pd_value < low:
        rec = "AKCEPTACJA"
    elif pd_value >= high:
        rec = "ODRZUCENIE"
    else:
        rec = "WERYFIKACJA"

    if app["loan_amnt"] > float(settings["escalation_amount"]):
        status = "DO_AKCEPTACJI_KIEROWNIKA"
    elif rec == "AKCEPTACJA":
        status = "REKOMENDACJA_POZYTYWNA"
    elif rec == "ODRZUCENIE":
        status = "REKOMENDACJA_NEGATYWNA"
    else:
        status = "DO_WERYFIKACJI"

    return {
        "model_key": key,
        "model_version": metrics.get("trained_at", "?"),
        "pd": pd_value,
        "risk_class": cls,
        "risk_desc": cls_desc,
        "recommendation": rec,
        "status": status,
        "explanation": explain(model, app, load_reference()),
        "flags": policy_flags(app),
    }


def explanation_frame(expl: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(expl)
    if df.empty:
        return df
    df["kierunek"] = df["delta"].apply(lambda d: "↑ zwiększa ryzyko" if d > 0.001 else ("↓ zmniejsza ryzyko" if d < -0.001 else "≈ neutralny"))
    df["wpływ (pp)"] = (df["delta"] * 100).round(1)
    return df.rename(columns={"label": "czynnik", "value": "wartość klienta", "reference": "wartość typowa"})[
        ["czynnik", "wartość klienta", "wartość typowa", "wpływ (pp)", "kierunek"]]
