import altair as alt
import pandas as pd
import streamlit as st

from src import db
from src import features as F
from src.ui import require, status_label

require("raporty")
st.title("📊 Raporty portfela wniosków")

df = db.applications_frame()
if df.empty:
    st.info("Brak danych – zarejestruj wnioski.")
    st.stop()

df["data"] = df["created_at"].str[:10]
decided = df[df["status"].isin(db.FINAL_STATUSES)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Liczba wniosków", len(df))
c2.metric("Wnioskowana kwota", f"{df['loan_amnt'].sum():,.0f}".replace(",", " "))
c3.metric("Wskaźnik akceptacji", f"{(decided['status'] == 'ZAAKCEPTOWANY').mean():.0%}" if len(decided) else "—")
c4.metric("Średnie PD portfela", f"{df['pd'].mean():.1%}")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Wnioski wg klasy ryzyka**")
    st.altair_chart(alt.Chart(df).mark_bar().encode(
        x=alt.X("risk_class:N", title="Klasa", sort=list("ABCDE")), y=alt.Y("count():Q", title="Liczba")))
with col2:
    st.markdown("**Wnioski wg statusu**")
    s = df.assign(status=df["status"].map(status_label))
    st.altair_chart(alt.Chart(s).mark_bar().encode(
        y=alt.Y("status:N", title=None), x=alt.X("count():Q", title="Liczba")))

st.markdown("**Średnie PD i kwota wg celu kredytu**")
g = df.groupby("loan_intent").agg(liczba=("id", "count"), srednie_pd=("pd", "mean"), suma_kwot=("loan_amnt", "sum")).reset_index()
g["loan_intent"] = g["loan_intent"].map(F.LOAN_INTENT)
g["srednie_pd"] = (g["srednie_pd"] * 100).round(1)
st.dataframe(g.rename(columns={"loan_intent": "Cel", "liczba": "Liczba", "srednie_pd": "Średnie PD [%]", "suma_kwot": "Suma kwot"}),
             hide_index=True)

st.markdown("**Wnioski wg oddziału**")
b = df.groupby("branch").agg(liczba=("id", "count"), srednie_pd=("pd", "mean")).reset_index()
b["srednie_pd"] = (b["srednie_pd"] * 100).round(1)
st.dataframe(b.rename(columns={"branch": "Oddział", "liczba": "Liczba", "srednie_pd": "Średnie PD [%]"}), hide_index=True)

st.download_button("⬇️ Raport (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "raport_portfela.csv", "text/csv")
