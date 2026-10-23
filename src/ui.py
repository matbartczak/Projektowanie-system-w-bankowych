"""Wspólne elementy interfejsu Streamlit."""
from __future__ import annotations

import html
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from . import auth, db
from . import features as F
from . import scoring as S
from .training import MODEL_NAMES

RISK_COLORS = {"A": "#1a7f37", "B": "#4c9a2a", "C": "#b58900", "D": "#d9480f", "E": "#c92a2a"}


def current_user() -> dict | None:
    return st.session_state.get("user")


def require(permission: str) -> dict:
    user = current_user()
    if not auth.can(user, permission):
        st.error("Brak uprawnień do tej funkcji.")
        st.stop()
    return user


def fmt_money(v) -> str:
    return f"{float(v):,.0f}".replace(",", " ") + " USD"


def status_label(code: str) -> str:
    return db.STATUSES.get(code, code)


def explanation_chart(expl: list[dict]):
    df = pd.DataFrame(expl)
    df["pp"] = df["delta"] * 100
    df["kierunek"] = df["pp"].apply(lambda v: "zwiększa ryzyko" if v > 0 else "zmniejsza ryzyko")
    df["opis"] = df["label"] + " = " + df["value"]
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("pp:Q", title="Wpływ na PD względem typowego klienta (punkty proc.)"),
            y=alt.Y("opis:N", sort=alt.EncodingSortField(field="pp", op="sum", order="descending"), title=None,
                    axis=alt.Axis(labelLimit=360)),
            color=alt.Color("kierunek:N", scale=alt.Scale(domain=["zwiększa ryzyko", "zmniejsza ryzyko"],
                                                            range=["#c92a2a", "#1a7f37"]), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=[alt.Tooltip("label:N", title="Czynnik"), alt.Tooltip("value:N", title="Klient"),
                     alt.Tooltip("reference:N", title="Typowo"), alt.Tooltip("pp:Q", title="Wpływ [pp]", format=".1f")],
        )
        .properties(height=34 * len(df) + 40)
    )


def render_score(result: dict):
    """Panel z wynikiem scoringu (PD, klasa, rekomendacja, flagi, wyjaśnienie)."""
    pd_value = result["pd"]
    cls = result["risk_class"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Prawdopodobieństwo niewypłacalności (PD)", f"{pd_value:.1%}")
    c2.metric("Klasa ryzyka", f"{cls} – {S.risk_class(pd_value)[1]}")
    c3.metric("Model", MODEL_NAMES.get(result["model_key"], result["model_key"]))
    st.progress(min(max(pd_value, 0.0), 1.0))

    rec = result["recommendation"]
    msg = S.RECOMMENDATIONS[rec]
    {"AKCEPTACJA": st.success, "WERYFIKACJA": st.warning, "ODRZUCENIE": st.error}[rec](msg)

    flags = result.get("flags") or []
    if flags:
        st.markdown("**Reguły polityki kredytowej:**")
        for f in flags:
            st.markdown(f"- ⚠️ {f}")

    st.markdown("**Czynniki wpływające na wynik**")
    st.altair_chart(explanation_chart(result["explanation"]))
    with st.expander("Tabela czynników i metoda wyjaśniania"):
        st.dataframe(S.explanation_frame(result["explanation"]), hide_index=True)
        st.caption(
            "Wpływ = o ile punktów procentowych zmieniłoby się PD, gdyby klient miał w danej cesze "
            "wartość typową dla portfela (mediana / dominanta ze zbioru uczącego). "
            "Wynik modelu jest rekomendacją – ostateczną decyzję podejmuje uprawniony pracownik banku.")


def application_card_html(app: dict) -> str:
    """Dokument wynikowy: „Karta oceny ryzyka kredytowego” (HTML do wydruku / PDF)."""
    e = html.escape
    rows = "".join(
        f"<tr><td>{e(r['label'])}</td><td>{e(str(r['value']))}</td><td>{e(str(r['reference']))}</td>"
        f"<td style='text-align:right;color:{'#c92a2a' if r['delta'] > 0 else '#1a7f37'}'>{r['delta']*100:+.1f}</td></tr>"
        for r in app["explanation"])
    flags = S.policy_flags(app)
    flags_html = "".join(f"<li>{e(f)}</li>" for f in flags) or "<li>Brak</li>"
    inst = S.monthly_installment(app["loan_amnt"], app["loan_int_rate"], int(app.get("loan_term_months") or 36))
    decision = (f"{e(status_label(app['status']))} – {e(app.get('decided_by_name') or '')}, {e(app.get('decided_at') or '')}"
                f"<br><i>{e(app.get('decision_comment') or '')}</i>") if app["status"] in db.FINAL_STATUSES else e(status_label(app["status"]))
    return f"""<!doctype html><html lang="pl"><head><meta charset="utf-8">
<title>Karta oceny {e(app['number'])}</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;color:#222}}h1{{font-size:20px;margin:0}}
h2{{font-size:15px;border-bottom:1px solid #999;padding-bottom:4px;margin-top:24px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #ccc;padding:5px 8px;text-align:left}}
.big{{font-size:28px;font-weight:bold;color:{RISK_COLORS.get(app['risk_class'], '#333')}}}.muted{{color:#666;font-size:12px}}</style></head>
<body>
<h1>Bank Spółdzielczy w Nowym Mieście – Karta oceny ryzyka kredytowego</h1>
<p class="muted">Wniosek nr {e(app['number'])} · zarejestrowany {e(app['created_at'])} · {e(app.get('branch') or '')} · doradca: {e(app['created_by_name'])}</p>
<h2>1. Wnioskodawca</h2>
<table><tr><th>Imię i nazwisko</th><td>{e(app['first_name'])} {e(app['last_name'])}</td><th>PESEL</th><td>{e(app['pesel'][:6])}*****</td></tr>
<tr><th>Wiek</th><td>{app['person_age']}</td><th>Roczny dochód</th><td>{fmt_money(app['person_income'])}</td></tr>
<tr><th>Status mieszkaniowy</th><td>{e(F.HOME_OWNERSHIP.get(app['person_home_ownership'], ''))}</td><th>Staż pracy</th><td>{app['person_emp_length']:g} lat</td></tr>
<tr><th>Zaległości w BIK</th><td>{e(F.DEFAULT_ON_FILE.get(app['cb_person_default_on_file'], ''))}</td><th>Historia kredytowa</th><td>{app['cb_person_cred_hist_length']:g} lat</td></tr></table>
<h2>2. Wnioskowany kredyt</h2>
<table><tr><th>Cel</th><td>{e(F.LOAN_INTENT.get(app['loan_intent'], ''))}</td><th>Kwota</th><td>{fmt_money(app['loan_amnt'])}</td></tr>
<tr><th>Oprocentowanie</th><td>{app['loan_int_rate']:.2f}%</td><th>Okres / rata</th><td>{app.get('loan_term_months') or 36} mies. / {inst:,.0f} USD</td></tr></table>
<h2>3. Wynik oceny</h2>
<p>PD: <span class="big">{app['pd']:.1%}</span> &nbsp; Klasa ryzyka: <span class="big">{e(app['risk_class'])}</span></p>
<p>{e(S.RECOMMENDATIONS.get(app['recommendation'], ''))} · model: {e(MODEL_NAMES.get(app['model_key'], ''))} (wersja {e(app.get('model_version') or '')})</p>
<p><b>Reguły polityki kredytowej:</b></p><ul>{flags_html}</ul>
<h2>4. Czynniki wpływające na wynik</h2>
<table><tr><th>Czynnik</th><th>Klient</th><th>Wartość typowa</th><th>Wpływ na PD [pp]</th></tr>{rows}</table>
<h2>5. Decyzja</h2><p>{decision}</p>
<p class="muted">Wygenerowano {datetime.now():%Y-%m-%d %H:%M}. Ocena modelu ma charakter wspierający; decyzję kredytową podejmuje uprawniony pracownik banku.</p>
</body></html>"""


def render_application(app_id: int):
    user = current_user()
    app = db.get_application(app_id)
    if not app:
        st.warning("Nie znaleziono wniosku.")
        return
    if user["role"] == "doradca" and app["created_by"] != user["id"]:
        st.error("Możesz przeglądać tylko wnioski, które zarejestrowałeś.")
        return

    st.subheader(f"Wniosek {app['number']}")
    st.caption(f"Status: **{status_label(app['status'])}** · zarejestrowany {app['created_at']} przez {app['created_by_name']} ({app.get('branch') or '-'})")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Klient**")
        st.write(f"{app['first_name']} {app['last_name']}, wiek {app['person_age']}, PESEL {app['pesel'][:6]}*****")
        st.write(f"Dochód roczny: {fmt_money(app['person_income'])} · staż: {app['person_emp_length']:g} lat")
        st.write(f"Status mieszkaniowy: {F.HOME_OWNERSHIP.get(app['person_home_ownership'])}")
        st.write(f"BIK: zaległości – {F.DEFAULT_ON_FILE.get(app['cb_person_default_on_file'])}, historia {app['cb_person_cred_hist_length']:g} lat")
    with c2:
        st.markdown("**Kredyt**")
        st.write(f"Cel: {F.LOAN_INTENT.get(app['loan_intent'])}")
        st.write(f"Kwota: {fmt_money(app['loan_amnt'])} · oprocentowanie {app['loan_int_rate']:.2f}%")
        months = int(app.get("loan_term_months") or 36)
        st.write(f"Okres: {months} mies. · rata ≈ {S.monthly_installment(app['loan_amnt'], app['loan_int_rate'], months):,.0f} USD")

    st.divider()
    render_score({**app, "flags": S.policy_flags(app)})

    st.download_button("📄 Pobierz kartę oceny (HTML – do druku/PDF)", application_card_html(app),
                       file_name=f"karta_oceny_{app['number'].replace('/', '_')}.html", mime="text/html")

    if app["status"] in db.FINAL_STATUSES:
        st.info(f"Decyzja: **{status_label(app['status'])}** – {app.get('decided_by_name')}, {app.get('decided_at')}. "
                f"Uzasadnienie: {app.get('decision_comment') or '—'}")
        return

    # --- panel decyzyjny ---
    needs_manager = app["status"] == "DO_AKCEPTACJI_KIEROWNIKA"
    allowed = auth.can(user, "wniosek.decyzja_eskalowana") if needs_manager else auth.can(user, "wniosek.decyzja")
    if not allowed:
        if needs_manager and auth.can(user, "wniosek.decyzja"):
            st.info("Wniosek przekracza limit kwotowy – decyzję podejmuje kierownik.")
        return

    st.divider()
    st.markdown("### Decyzja kredytowa")
    with st.form(f"decision_{app_id}"):
        choice = st.radio("Decyzja", ["Akceptuj", "Odrzuć"] + (["Przekaż do kierownika"] if user["role"] == "analityk" else []),
                          horizontal=True)
        comment = st.text_area("Uzasadnienie (wymagane)")
        ok = st.form_submit_button("Zapisz decyzję", type="primary")
    if ok:
        if not comment.strip():
            st.error("Uzasadnienie decyzji jest wymagane.")
            return
        if choice == "Przekaż do kierownika":
            db.escalate(app_id, comment, user)
        else:
            db.decide(app_id, "ZAAKCEPTOWANY" if choice == "Akceptuj" else "ODRZUCONY", comment, user)
        st.success("Zapisano.")
        st.rerun()
