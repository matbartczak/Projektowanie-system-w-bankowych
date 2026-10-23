import streamlit as st

from src import auth, db
from src.ui import current_user, status_label

user = current_user()
st.title("Pulpit")
st.caption(f"Witaj, {user['full_name']} ({auth.ROLE_NAMES[user['role']]})")

df = db.applications_frame(created_by=user["id"] if user["role"] == "doradca" else None)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Wnioski", len(df))
c2.metric("Oczekujące na decyzję", int((~df["status"].isin(db.FINAL_STATUSES)).sum()) if len(df) else 0)
c3.metric("Zaakceptowane", int((df["status"] == "ZAAKCEPTOWANY").sum()) if len(df) else 0)
c4.metric("Średnie PD", f"{df['pd'].mean():.1%}" if len(df) else "—")

st.subheader("Moje zadania")
if user["role"] == "doradca":
    st.write("Zarejestruj nowy wniosek w zakładce **Nowy wniosek** lub sprawdź status złożonych wniosków.")
    queue = df[~df["status"].isin(db.FINAL_STATUSES)]
elif user["role"] == "analityk":
    queue = df[df["status"].isin(["DO_WERYFIKACJI", "REKOMENDACJA_POZYTYWNA", "REKOMENDACJA_NEGATYWNA"])]
elif user["role"] == "kierownik":
    queue = df[df["status"] == "DO_AKCEPTACJI_KIEROWNIKA"]
else:
    st.write("Administrator: zarządzanie użytkownikami, modelami i dziennikiem zdarzeń.")
    queue = df.iloc[0:0]

if len(queue):
    view = queue[["number", "created_at", "klient", "loan_amnt", "pd", "risk_class", "status"]].copy()
    view["status"] = view["status"].map(status_label)
    view["pd"] = (view["pd"] * 100).round(1)
    st.dataframe(view.rename(columns={"number": "Nr", "created_at": "Data", "klient": "Klient",
                                      "loan_amnt": "Kwota", "pd": "PD [%]", "risk_class": "Klasa", "status": "Status"}),
                 hide_index=True)
    st.caption("Szczegóły i decyzja – zakładka **Rejestr wniosków**.")
elif user["role"] != "admin":
    st.success("Brak oczekujących wniosków.")
