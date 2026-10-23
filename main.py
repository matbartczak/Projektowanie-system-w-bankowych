"""System oceny ryzyka kredytowego – Bank Spółdzielczy w Nowym Mieście.

Uruchomienie:   streamlit run main.py
Konta demo:     doradca/doradca123, analityk/analityk123, kierownik/kierownik123, admin/admin123
"""
import streamlit as st

from src import auth, db
from src import scoring as S
from src.training import train_all

st.set_page_config(page_title="System oceny ryzyka kredytowego", page_icon="💳", layout="wide")


@st.cache_resource
def bootstrap():
    db.init_db()
    return True


bootstrap()

# Pierwsze uruchomienie: brak modeli -> trening (ok. 1 min)
if not S.models_available():
    with st.status("Pierwsze uruchomienie – trenowanie modeli (ok. 1 min)…", expanded=True) as s:
        train_all(log=st.write)
        s.update(label="Modele wytrenowane", state="complete")


def login_page():
    st.title("💳 System oceny ryzyka kredytowego")
    st.caption("Bank Spółdzielczy w Nowym Mieście · Pion Kredytowy")
    col, _ = st.columns([1, 2])
    with col:
        with st.form("login"):
            login = st.text_input("Login")
            password = st.text_input("Hasło", type="password")
            ok = st.form_submit_button("Zaloguj", type="primary")
        if ok:
            user = auth.authenticate(login, password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Nieprawidłowy login lub hasło.")
        with st.expander("Konta demonstracyjne"):
            st.markdown(
                "| Login | Hasło | Rola |\n|---|---|---|\n"
                "| doradca | doradca123 | Doradca kredytowy |\n"
                "| analityk | analityk123 | Analityk ryzyka |\n"
                "| kierownik | kierownik123 | Kierownik Działu Kredytów |\n"
                "| admin | admin123 | Administrator |")


def logout():
    db.log(st.session_state.user["id"], "WYLOGOWANIE")
    del st.session_state["user"]
    st.rerun()


user = st.session_state.get("user")
if not user:
    st.navigation([st.Page(login_page, title="Logowanie", icon="🔐")]).run()
    st.stop()

with st.sidebar:
    st.markdown(f"**{user['full_name']}**  \n{auth.ROLE_NAMES[user['role']]}  \n{user.get('branch') or ''}")
    if st.button("Wyloguj"):
        logout()

pages = [st.Page("views/pulpit.py", title="Pulpit", icon="🏠", default=True)]
if auth.can(user, "wniosek.rejestracja"):
    pages.append(st.Page("views/nowy_wniosek.py", title="Nowy wniosek", icon="📝"))
pages.append(st.Page("views/wnioski.py", title="Rejestr wniosków", icon="📂"))
if auth.can(user, "raporty"):
    pages.append(st.Page("views/raporty.py", title="Raporty", icon="📊"))
if auth.can(user, "model.porownanie"):
    pages.append(st.Page("views/modele.py", title="Modele i metryki", icon="🧠"))
if auth.can(user, "ustawienia.progi") or auth.can(user, "uzytkownicy.zarzadzanie") or auth.can(user, "audyt.podglad"):
    pages.append(st.Page("views/administracja.py", title="Administracja", icon="⚙️"))

st.navigation(pages).run()
