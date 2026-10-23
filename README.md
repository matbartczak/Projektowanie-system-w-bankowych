# System Oceny Ryzyka Kredytowego (SORK)

Aplikacja szacująca ryzyko niespłacenia kredytu dla (fikcyjnego) Banku Spółdzielczego w Nowym Mieście.
Pełna dokumentacja projektu (punkty 1–7 wymagań zaliczenia): `docs/Dokumentacja_SORK.docx`.

## Funkcje
- logowanie i role: doradca, analityk ryzyka, kierownik, administrator (RBAC),
- rejestracja i analiza wniosków (SQLite), walidacja PESEL, reguły polityki kredytowej (DSTI, BIK),
- PD + klasa ryzyka A–E + rekomendacja, ścieżka decyzyjna z eskalacją do kierownika,
- porównanie: regresja logistyczna vs Random Forest (+ XGBoost) – precision, recall, F1, ROC-AUC, PR-AUC, krzywe ROC/PR, macierze pomyłek,
- wyjaśnienie czynników: globalne (permutacyjne) i dla pojedynczego wniosku,
- raporty portfela, karta oceny (HTML do druku), eksport CSV, dziennik zdarzeń.

## Uruchomienie
```bash
pip install -r requirements.txt
python train_models.py      # opcjonalnie – przy 1. starcie aplikacji dzieje się automatycznie
python seed_demo.py 40      # opcjonalnie – przykładowe wnioski
streamlit run main.py
```
Konta demo: `doradca/doradca123`, `analityk/analityk123`, `kierownik/kierownik123`, `admin/admin123`.

## Struktura
```
main.py              logowanie + menu wg roli
views/               ekrany (pulpit, nowy wniosek, rejestr, raporty, modele, administracja)
src/features.py      wspólna inżynieria cech (trening = aplikacja)
src/training.py      trening i porównanie modeli, metryki -> models/metrics.json
src/scoring.py       PD, klasy ryzyka, polityka decyzyjna, wyjaśnienia
src/auth.py, db.py   role/uprawnienia, hasła (PBKDF2), baza SQLite
trening_modelu.ipynb analiza eksploracyjna i strojenie XGBoost
```
