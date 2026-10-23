"""Wypełnia bazę przykładowymi wnioskami (do prezentacji). Uruchom: python seed_demo.py [liczba]"""
import random
import sys

import pandas as pd

from src import db
from src import features as F
from src import scoring as S
from src.training import train_all

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
FIRST = ["Adam", "Ewa", "Krzysztof", "Magdalena", "Paweł", "Katarzyna", "Michał", "Agnieszka", "Łukasz", "Joanna"]
LAST = ["Mazur", "Krawczyk", "Kaczmarek", "Wójcik", "Dąbrowski", "Kozłowska", "Jankowski", "Pawlak", "Michalska", "Król"]


def fake_pesel(rng):
    d = [rng.randint(0, 9) for _ in range(10)]
    w = [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]
    return "".join(map(str, d)) + str((10 - sum(a * b for a, b in zip(d, w)) % 10) % 10)


def main():
    db.init_db()
    if not S.models_available():
        print("Brak wytrenowanych modeli – trenuję (ok. 1 min)…")
        train_all()
    rng = random.Random(7)
    data = F.clean_training_data(pd.read_csv("credit_risk_dataset.csv")).dropna().sample(N, random_state=7)
    users = {u["login"]: u for u in (db.get_user_by_login(x) for x in ["doradca", "doradca2", "analityk", "kierownik"])}
    settings = db.get_settings()
    for _, r in data.iterrows():
        app = {c: r[c] for c in F.RAW_FEATURES}
        app.update(first_name=rng.choice(FIRST), last_name=rng.choice(LAST), pesel=fake_pesel(rng),
                   loan_term_months=rng.choice([24, 36, 48, 60]))
        app["person_age"] = int(app["person_age"])
        res = S.score_application(app, settings)
        rec = {**app, **{k: res[k] for k in ("model_key", "model_version", "pd", "risk_class", "recommendation", "status")},
               "explanation_json": res["explanation"]}
        app_id = db.insert_application(rec, users[rng.choice(["doradca", "doradca2"])])
        # część wniosków od razu z decyzją
        if rng.random() < 0.6 and res["status"] != "DO_AKCEPTACJI_KIEROWNIKA":
            ok = res["pd"] < float(settings["threshold_high"]) and rng.random() > 0.15
            db.decide(app_id, "ZAAKCEPTOWANY" if ok else "ODRZUCONY",
                      "Zgodnie z rekomendacją systemu." if ok == (res["recommendation"] != "ODRZUCENIE") else "Decyzja po weryfikacji dokumentów.",
                      users["analityk"])
    print(f"Dodano {N} wniosków demonstracyjnych.")


if __name__ == "__main__":
    main()
