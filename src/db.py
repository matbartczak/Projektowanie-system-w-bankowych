"""Warstwa danych – SQLite (plik data/kredyty.db)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "kredyty.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    login TEXT UNIQUE NOT NULL,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('doradca','analityk','kierownik','admin')),
    branch TEXT,
    password_hash TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    number TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL,
    created_by INTEGER NOT NULL REFERENCES users(id),
    branch TEXT,
    -- dane klienta
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    pesel TEXT NOT NULL,
    person_age INTEGER NOT NULL,
    person_income REAL NOT NULL,
    person_home_ownership TEXT NOT NULL,
    person_emp_length REAL NOT NULL,
    -- dane kredytu
    loan_intent TEXT NOT NULL,
    loan_amnt REAL NOT NULL,
    loan_int_rate REAL NOT NULL,
    loan_term_months INTEGER,
    -- dane z BIK
    cb_person_default_on_file TEXT NOT NULL,
    cb_person_cred_hist_length REAL NOT NULL,
    -- wynik scoringu
    model_key TEXT,
    model_version TEXT,
    pd REAL,
    risk_class TEXT,
    recommendation TEXT,
    explanation_json TEXT,
    -- decyzja
    status TEXT NOT NULL,
    decided_by INTEGER REFERENCES users(id),
    decided_at TEXT,
    decision_comment TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    user_id INTEGER,
    action TEXT NOT NULL,
    object TEXT,
    details TEXT
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

DEFAULT_SETTINGS = {
    "active_model": "rf",
    "threshold_low": "0.15",      # PD poniżej -> rekomendacja akceptacji
    "threshold_high": "0.50",     # PD od -> rekomendacja odrzucenia
    "escalation_amount": "25000", # kwota, powyżej której decyzję podejmuje kierownik
}

STATUSES = {
    "DO_WERYFIKACJI": "Do weryfikacji (analityk)",
    "DO_AKCEPTACJI_KIEROWNIKA": "Do akceptacji kierownika",
    "REKOMENDACJA_POZYTYWNA": "Rekomendacja pozytywna – czeka na decyzję",
    "REKOMENDACJA_NEGATYWNA": "Rekomendacja negatywna – czeka na decyzję",
    "ZAAKCEPTOWANY": "Zaakceptowany",
    "ODRZUCONY": "Odrzucony",
}
FINAL_STATUSES = {"ZAAKCEPTOWANY", "ODRZUCONY"}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@contextmanager
def connect(path: Path | None = None):
    path = path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(seed_users=True):
    from .auth import hash_password
    with connect() as con:
        con.executescript(SCHEMA)
        for k, v in DEFAULT_SETTINGS.items():
            con.execute("INSERT OR IGNORE INTO settings(key, value) VALUES (?, ?)", (k, v))
        if seed_users and con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            demo = [
                ("doradca", "Anna Nowak", "doradca", "Oddział Nowe Miasto"),
                ("doradca2", "Piotr Zieliński", "doradca", "Filia Stare Pole"),
                ("analityk", "Jan Kowalski", "analityk", "Centrala"),
                ("kierownik", "Maria Wiśniewska", "kierownik", "Centrala"),
                ("admin", "Tomasz Lewandowski", "admin", "Centrala"),
            ]
            for login, name, role, branch in demo:
                con.execute(
                    "INSERT INTO users(login, full_name, role, branch, password_hash, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (login, name, role, branch, hash_password(login + "123"), now()))


# ---------- ustawienia ----------
def get_settings() -> dict:
    with connect() as con:
        return {r["key"]: r["value"] for r in con.execute("SELECT key, value FROM settings")}


def set_setting(key: str, value, user_id: int | None):
    with connect() as con:
        con.execute("INSERT OR REPLACE INTO settings(key, value) VALUES (?, ?)", (key, str(value)))
    log(user_id, "ZMIANA_USTAWIEN", key, str(value))


# ---------- audyt ----------
def log(user_id, action, obj=None, details=None):
    with connect() as con:
        con.execute("INSERT INTO audit_log(ts, user_id, action, object, details) VALUES (?,?,?,?,?)",
                    (now(), user_id, action, obj, details))


def audit_frame(limit=500) -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query(
            "SELECT a.ts AS czas, u.login AS uzytkownik, a.action AS akcja, a.object AS obiekt, "
            "a.details AS szczegoly FROM audit_log a LEFT JOIN users u ON u.id = a.user_id "
            "ORDER BY a.id DESC LIMIT ?", con, params=(limit,))


# ---------- użytkownicy ----------
def get_user_by_login(login: str):
    with connect() as con:
        r = con.execute("SELECT * FROM users WHERE login = ?", (login,)).fetchone()
        return dict(r) if r else None


def users_frame() -> pd.DataFrame:
    with connect() as con:
        return pd.read_sql_query(
            "SELECT id, login, full_name, role, branch, active, created_at FROM users ORDER BY id", con)


def add_user(login, full_name, role, branch, password, by_user):
    from .auth import hash_password
    with connect() as con:
        con.execute(
            "INSERT INTO users(login, full_name, role, branch, password_hash, created_at) VALUES (?,?,?,?,?,?)",
            (login, full_name, role, branch, hash_password(password), now()))
    log(by_user, "DODANIE_UZYTKOWNIKA", login, role)


def set_user_active(user_id: int, active: bool, by_user):
    with connect() as con:
        con.execute("UPDATE users SET active = ? WHERE id = ?", (int(active), user_id))
    log(by_user, "ZMIANA_STATUSU_UZYTKOWNIKA", str(user_id), "aktywny" if active else "zablokowany")


# ---------- wnioski ----------
def next_number(con) -> str:
    year = datetime.now().year
    n = con.execute("SELECT COUNT(*) FROM applications WHERE number LIKE ?", (f"WK/{year}/%",)).fetchone()[0]
    return f"WK/{year}/{n + 1:05d}"


def insert_application(data: dict, user: dict) -> int:
    with connect() as con:
        data = dict(data)
        data["number"] = next_number(con)
        data["created_at"] = now()
        data["created_by"] = user["id"]
        data["branch"] = user.get("branch")
        if isinstance(data.get("explanation_json"), (list, dict)):
            data["explanation_json"] = json.dumps(data["explanation_json"], ensure_ascii=False)
        cols = ", ".join(data)
        qs = ", ".join("?" for _ in data)
        cur = con.execute(f"INSERT INTO applications({cols}) VALUES ({qs})", list(data.values()))
        app_id = cur.lastrowid
    log(user["id"], "REJESTRACJA_WNIOSKU", data["number"], f"PD={data.get('pd'):.3f}")
    return app_id


def get_application(app_id: int) -> dict | None:
    with connect() as con:
        r = con.execute(
            "SELECT a.*, u.full_name AS created_by_name, d.full_name AS decided_by_name "
            "FROM applications a JOIN users u ON u.id = a.created_by "
            "LEFT JOIN users d ON d.id = a.decided_by WHERE a.id = ?", (app_id,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["explanation"] = json.loads(d["explanation_json"]) if d.get("explanation_json") else []
        return d


def applications_frame(created_by: int | None = None) -> pd.DataFrame:
    q = ("SELECT a.id, a.number, a.created_at, u.full_name AS doradca, a.branch, "
         "a.first_name || ' ' || a.last_name AS klient, a.loan_intent, a.loan_amnt, a.person_income, "
         "a.pd, a.risk_class, a.recommendation, a.status, a.model_key, a.decided_at "
         "FROM applications a JOIN users u ON u.id = a.created_by")
    params: tuple = ()
    if created_by is not None:
        q += " WHERE a.created_by = ?"
        params = (created_by,)
    q += " ORDER BY a.id DESC"
    with connect() as con:
        return pd.read_sql_query(q, con, params=params)


def decide(app_id: int, status: str, comment: str, user: dict):
    with connect() as con:
        con.execute(
            "UPDATE applications SET status=?, decided_by=?, decided_at=?, decision_comment=? WHERE id=?",
            (status, user["id"], now(), comment, app_id))
        number = con.execute("SELECT number FROM applications WHERE id=?", (app_id,)).fetchone()[0]
    log(user["id"], "DECYZJA_" + status, number, comment)


def escalate(app_id: int, comment: str, user: dict):
    with connect() as con:
        con.execute("UPDATE applications SET status='DO_AKCEPTACJI_KIEROWNIKA', decision_comment=? WHERE id=?",
                    (comment, app_id))
        number = con.execute("SELECT number FROM applications WHERE id=?", (app_id,)).fetchone()[0]
    log(user["id"], "ESKALACJA", number, comment)
