"""Wspólna inżynieria cech – używana zarówno przy trenowaniu, jak i w aplikacji.

Dzięki temu dane podawane do modelu w aplikacji są przetwarzane dokładnie tak
samo jak dane uczące (brak rozjazdu "training/serving skew").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "loan_status"

# Dane wprowadzane przez doradcę / pobierane z BIK (surowe wejście modelu)
RAW_NUMERIC = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "cb_person_cred_hist_length",
]
RAW_CATEGORICAL = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]
RAW_FEATURES = RAW_NUMERIC + RAW_CATEGORICAL

# Cechy pochodne
DERIVED_NUMERIC = [
    "log_income",
    "loan_percent_income",
    "loan_to_emp_length_ratio",
    "int_rate_to_loan_amt_ratio",
]
DERIVED_CATEGORICAL = ["age_group", "income_group", "loan_amount_group"]

MODEL_NUMERIC = RAW_NUMERIC + DERIVED_NUMERIC
MODEL_CATEGORICAL = RAW_CATEGORICAL + DERIVED_CATEGORICAL
MODEL_FEATURES = MODEL_NUMERIC + MODEL_CATEGORICAL

# Kolumna loan_grade jest wewnętrznym ratingiem nadawanym przez bank PO ocenie,
# więc jej użycie byłoby "przeciekiem" informacji – świadomie ją pomijamy.
EXCLUDED = ["loan_grade"]

# Polskie etykiety cech (do UI, wyjaśnień i dokumentacji)
FEATURE_LABELS = {
    "person_age": "Wiek",
    "person_income": "Roczny dochód",
    "person_emp_length": "Staż pracy",
    "loan_amnt": "Kwota kredytu",
    "loan_int_rate": "Oprocentowanie",
    "cb_person_cred_hist_length": "Długość historii kredytowej (BIK)",
    "person_home_ownership": "Status mieszkaniowy",
    "loan_intent": "Cel kredytu",
    "cb_person_default_on_file": "Wcześniejsze zaległości (BIK)",
    "log_income": "Log. dochodu",
    "loan_percent_income": "Kwota kredytu / dochód",
    "loan_to_emp_length_ratio": "Staż / kwota kredytu",
    "int_rate_to_loan_amt_ratio": "Oprocentowanie / kwota",
    "age_group": "Grupa wiekowa",
    "income_group": "Grupa dochodowa",
    "loan_amount_group": "Grupa kwoty kredytu",
}

HOME_OWNERSHIP = {
    "RENT": "Wynajem",
    "OWN": "Własność",
    "MORTGAGE": "Kredyt hipoteczny",
    "OTHER": "Inne",
}
LOAN_INTENT = {
    "PERSONAL": "Cele osobiste",
    "EDUCATION": "Edukacja",
    "MEDICAL": "Cele medyczne",
    "VENTURE": "Działalność gospodarcza",
    "HOMEIMPROVEMENT": "Remont / ulepszenie domu",
    "DEBTCONSOLIDATION": "Konsolidacja zadłużenia",
}
DEFAULT_ON_FILE = {"N": "Nie", "Y": "Tak"}


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Czyszczenie zbioru historycznego (jak w notatniku trening_modelu.ipynb)."""
    df = df.copy()
    df = df.drop_duplicates()
    df = df[df["person_age"] <= 80]
    df = df[(df["person_emp_length"].isna()) | (df["person_emp_length"] <= 60)]
    # braki w loan_int_rate / person_emp_length uzupełnia imputer w pipeline
    return df.reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje cechy pochodne do ramki z surowymi danymi wniosku."""
    out = df.copy()
    income = out["person_income"].astype(float).clip(lower=1)
    amount = out["loan_amnt"].astype(float).clip(lower=1)
    emp = out["person_emp_length"].astype(float)
    rate = out["loan_int_rate"].astype(float)

    out["log_income"] = np.log(income)
    out["loan_percent_income"] = amount / income
    out["loan_to_emp_length_ratio"] = emp / amount
    out["int_rate_to_loan_amt_ratio"] = rate / amount

    out["age_group"] = pd.cut(
        out["person_age"], bins=[0, 25, 35, 45, 55, 200],
        labels=["do 25", "26-35", "36-45", "46-55", "56+"],
    ).astype(str)
    out["income_group"] = pd.cut(
        income, bins=[0, 25_000, 50_000, 75_000, 100_000, float("inf")],
        labels=["niski", "niski-średni", "średni", "średni-wysoki", "wysoki"],
    ).astype(str)
    out["loan_amount_group"] = pd.cut(
        amount, bins=[0, 5_000, 10_000, 15_000, float("inf")],
        labels=["mała", "średnia", "duża", "bardzo duża"],
    ).astype(str)
    return out[MODEL_FEATURES]


def application_to_frame(app: dict) -> pd.DataFrame:
    """Zamienia słownik z danymi wniosku na ramkę gotową dla modelu."""
    row = {k: app.get(k) for k in RAW_FEATURES}
    return add_features(pd.DataFrame([row]))
