import streamlit as st

from src import db
from src import features as F
from src import scoring as S
from src.ui import render_score, require, status_label

user = require("wniosek.rejestracja")
st.title("📝 Rejestracja wniosku kredytowego")


def pesel_valid(p: str) -> bool:
    if len(p) != 11 or not p.isdigit():
        return False
    w = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
    return (10 - sum(int(p[i]) * w[i] for i in range(10)) % 10) % 10 == int(p[10])


with st.form("application"):
    st.subheader("Dane wnioskodawcy")
    c1, c2, c3 = st.columns(3)
    first_name = c1.text_input("Imię")
    last_name = c2.text_input("Nazwisko")
    pesel = c3.text_input("PESEL", max_chars=11)
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Wiek", 18, 100, 30)
    income = c2.number_input("Roczny dochód (USD)", 0, 10_000_000, 60_000, step=1000)
    emp = c3.number_input("Staż pracy (lata)", 0.0, 60.0, 5.0, step=0.5)
    home = c4.selectbox("Status mieszkaniowy", list(F.HOME_OWNERSHIP), format_func=F.HOME_OWNERSHIP.get)

    st.subheader("Parametry kredytu")
    c1, c2, c3, c4 = st.columns(4)
    intent = c1.selectbox("Cel kredytu", list(F.LOAN_INTENT), format_func=F.LOAN_INTENT.get)
    amount = c2.number_input("Kwota kredytu (USD)", 500, 35_000, 10_000, step=500)
    rate = c3.number_input("Oprocentowanie (%)", 0.0, 30.0, 11.0, step=0.1)
    term = c4.selectbox("Okres (mies.)", [12, 24, 36, 48, 60, 72, 84], index=2)

    st.subheader("Dane z Biura Informacji Kredytowej (BIK)")
    c1, c2 = st.columns(2)
    bik_default = c1.radio("Wcześniejsze zaległości w spłacie", ["N", "Y"], format_func=F.DEFAULT_ON_FILE.get, horizontal=True)
    bik_hist = c2.number_input("Długość historii kredytowej (lata)", 0, 60, 4)
    consent = st.checkbox("Klient wyraził zgodę na przetwarzanie danych i zapytanie do BIK")
    submitted = st.form_submit_button("Oceń ryzyko i zarejestruj wniosek", type="primary")

if submitted:
    errors = []
    if not first_name.strip() or not last_name.strip():
        errors.append("Podaj imię i nazwisko.")
    if not pesel_valid(pesel):
        errors.append("Nieprawidłowy numer PESEL (11 cyfr, poprawna suma kontrolna).")
    if income <= 0:
        errors.append("Dochód musi być większy od zera.")
    if emp > age - 14:
        errors.append("Staż pracy jest niespójny z wiekiem.")
    if not consent:
        errors.append("Wymagana zgoda klienta.")
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    app = {
        "first_name": first_name.strip(), "last_name": last_name.strip(), "pesel": pesel,
        "person_age": int(age), "person_income": float(income), "person_emp_length": float(emp),
        "person_home_ownership": home, "loan_intent": intent, "loan_amnt": float(amount),
        "loan_int_rate": float(rate), "loan_term_months": int(term),
        "cb_person_default_on_file": bik_default, "cb_person_cred_hist_length": float(bik_hist),
    }
    result = S.score_application(app, db.get_settings())
    record = {**app, **{k: result[k] for k in ("model_key", "model_version", "pd", "risk_class", "recommendation", "status")},
              "explanation_json": result["explanation"]}
    app_id = db.insert_application(record, user)
    saved = db.get_application(app_id)
    st.success(f"Zarejestrowano wniosek **{saved['number']}** – status: {status_label(saved['status'])}")
    render_score(result)
