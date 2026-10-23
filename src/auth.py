"""Uwierzytelnianie i autoryzacja (role i uprawnienia)."""
from __future__ import annotations

import hashlib
import hmac
import os

ROLE_NAMES = {
    "doradca": "Doradca kredytowy",
    "analityk": "Analityk ryzyka",
    "kierownik": "Kierownik Działu Kredytów",
    "admin": "Administrator systemu",
}

# Macierz uprawnień: uprawnienie -> role, które je posiadają
PERMISSIONS = {
    "wniosek.rejestracja":        {"doradca"},
    "wniosek.podglad_wlasne":     {"doradca"},
    "wniosek.podglad_wszystkie":  {"analityk", "kierownik"},
    "wniosek.decyzja":            {"analityk", "kierownik"},
    "wniosek.decyzja_eskalowana": {"kierownik"},
    "model.porownanie":           {"analityk", "kierownik", "admin"},
    "model.wybor_aktywnego":      {"admin"},
    "model.trening":              {"admin"},
    "raporty":                    {"kierownik", "analityk"},
    "ustawienia.progi":           {"kierownik"},
    "uzytkownicy.zarzadzanie":    {"admin"},
    "audyt.podglad":              {"admin", "kierownik"},
}


def can(user: dict | None, permission: str) -> bool:
    return bool(user) and user["role"] in PERMISSIONS.get(permission, set())


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return salt.hex() + "$" + dk.hex()


def verify_password(password: str, stored: str) -> bool:
    salt_hex, dk_hex = stored.split("$")
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), 200_000)
    return hmac.compare_digest(dk.hex(), dk_hex)


def authenticate(login: str, password: str) -> dict | None:
    from . import db
    user = db.get_user_by_login(login.strip())
    if user and user["active"] and verify_password(password, user["password_hash"]):
        db.log(user["id"], "LOGOWANIE")
        return user
    db.log(user["id"] if user else None, "NIEUDANE_LOGOWANIE", login)
    return None
