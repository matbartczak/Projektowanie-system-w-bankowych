import streamlit as st

from src import auth, db
from src.ui import current_user

user = current_user()
st.title("⚙️ Administracja")

tabs = []
if auth.can(user, "ustawienia.progi"):
    tabs.append("Polityka decyzyjna")
if auth.can(user, "uzytkownicy.zarzadzanie"):
    tabs.append("Użytkownicy")
if auth.can(user, "audyt.podglad"):
    tabs.append("Dziennik zdarzeń")
if not tabs:
    st.error("Brak uprawnień.")
    st.stop()

for name, tab in zip(tabs, st.tabs(tabs)):
    with tab:
        if name == "Polityka decyzyjna":
            s = db.get_settings()
            st.write("Progi PD określają rekomendację systemu. Wnioski powyżej limitu kwotowego trafiają do kierownika.")
            with st.form("policy"):
                low = st.slider("PD poniżej → rekomendacja akceptacji", 0.01, 0.5, float(s["threshold_low"]), 0.01)
                high = st.slider("PD od → rekomendacja odrzucenia", 0.1, 0.95, float(s["threshold_high"]), 0.01)
                esc = st.number_input("Limit kwoty wymagający decyzji kierownika", 1000, 100000, int(float(s["escalation_amount"])), 1000)
                if st.form_submit_button("Zapisz", type="primary"):
                    if low >= high:
                        st.error("Próg akceptacji musi być niższy niż próg odrzucenia.")
                    else:
                        db.set_setting("threshold_low", low, user["id"])
                        db.set_setting("threshold_high", high, user["id"])
                        db.set_setting("escalation_amount", esc, user["id"])
                        st.success("Zapisano politykę decyzyjną.")

        elif name == "Użytkownicy":
            users = db.users_frame()
            st.dataframe(users.assign(role=users["role"].map(auth.ROLE_NAMES)), hide_index=True)
            st.markdown("**Dodaj użytkownika**")
            with st.form("add_user", clear_on_submit=True):
                c1, c2 = st.columns(2)
                login = c1.text_input("Login")
                name_ = c2.text_input("Imię i nazwisko")
                role = c1.selectbox("Rola", list(auth.ROLE_NAMES), format_func=auth.ROLE_NAMES.get)
                branch = c2.text_input("Oddział", "Oddział Nowe Miasto")
                pwd = st.text_input("Hasło początkowe (min. 8 znaków)", type="password")
                if st.form_submit_button("Dodaj"):
                    if len(pwd) < 8 or not login or not name_:
                        st.error("Uzupełnij dane; hasło min. 8 znaków.")
                    elif db.get_user_by_login(login):
                        st.error("Login zajęty.")
                    else:
                        db.add_user(login, name_, role, branch, pwd, user["id"])
                        st.success("Dodano użytkownika.")
                        st.rerun()
            st.markdown("**Blokowanie / odblokowanie konta**")
            c1, c2 = st.columns([3, 1])
            uid = c1.selectbox("Użytkownik", users["id"], format_func=lambda i: users.set_index("id").loc[i, "login"])
            if c2.button("Przełącz status") and uid != user["id"]:
                active = bool(users.set_index("id").loc[uid, "active"])
                db.set_user_active(int(uid), not active, user["id"])
                st.rerun()

        elif name == "Dziennik zdarzeń":
            st.dataframe(db.audit_frame(), hide_index=True)
