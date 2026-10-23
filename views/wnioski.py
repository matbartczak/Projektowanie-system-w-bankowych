import streamlit as st

from src import db
from src import features as F
from src.ui import current_user, render_application, status_label

user = current_user()
st.title("📂 Rejestr wniosków")

df = db.applications_frame(created_by=user["id"] if user["role"] == "doradca" else None)
if df.empty:
    st.info("Brak wniosków w rejestrze.")
    st.stop()

c1, c2, c3 = st.columns(3)
statuses = c1.multiselect("Status", list(db.STATUSES), format_func=status_label, placeholder="Wszystkie")
classes = c2.multiselect("Klasa ryzyka", list("ABCDE"), placeholder="Wszystkie")
search = c3.text_input("Szukaj (nr wniosku / klient)")
if statuses:
    df = df[df["status"].isin(statuses)]
if classes:
    df = df[df["risk_class"].isin(classes)]
if search:
    s = search.lower()
    df = df[df["number"].str.lower().str.contains(s) | df["klient"].str.lower().str.contains(s)]

view = df.assign(
    status=df["status"].map(status_label),
    loan_intent=df["loan_intent"].map(F.LOAN_INTENT),
    pd=(df["pd"] * 100).round(1),
)[["number", "created_at", "klient", "doradca", "loan_intent", "loan_amnt", "pd", "risk_class", "status"]]
st.dataframe(view.rename(columns={"number": "Nr", "created_at": "Data", "klient": "Klient", "doradca": "Doradca",
                                  "loan_intent": "Cel", "loan_amnt": "Kwota", "pd": "PD [%]",
                                  "risk_class": "Klasa", "status": "Status"}), hide_index=True)
st.download_button("⬇️ Eksport do CSV", df.to_csv(index=False).encode("utf-8-sig"),
                   file_name="rejestr_wnioskow.csv", mime="text/csv")

st.divider()
if len(df):
    options = dict(zip(df["number"] + " – " + df["klient"], df["id"]))
    choice = st.selectbox("Wybierz wniosek, aby zobaczyć szczegóły", list(options))
    render_application(int(options[choice]))
