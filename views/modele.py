import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src import auth, db
from src import features as F
from src import scoring as S
from src.training import MODEL_NAMES, train_all
from src.ui import require

user = require("model.porownanie")
st.title("🧠 Porównanie modeli")

m = S.load_metrics()
settings = db.get_settings()
models = m["models"]
st.caption(f"Wersja modeli: {m['trained_at']} · zbiór: {m['dataset']['rows_clean']} wniosków historycznych "
           f"(uczący {m['dataset']['train']}, testowy {m['dataset']['test']}), odsetek niewypłacalnych "
           f"{m['dataset']['default_rate']:.1%} · aktywny model: **{MODEL_NAMES[settings['active_model']]}**")

thr_mode = st.radio("Próg klasyfikacji dla metryk", ["0,5", "optymalny (max F1, wyznaczony w CV)"], horizontal=True)
key_thr = "at_050" if thr_mode == "0,5" else "at_best_f1"

rows = []
for k, v in models.items():
    t = v[key_thr]
    rows.append({"Model": v["name"], "ROC-AUC": v["roc_auc"], "ROC-AUC (CV)": v["cv_roc_auc"], "PR-AUC": v["pr_auc"],
                 "Precision": t["precision"], "Recall": t["recall"], "F1": t["f1"], "Specificity": t["specificity"],
                 "Accuracy": t["accuracy"], "Brier": v["brier"], "Próg": t["threshold"]})
table = pd.DataFrame(rows).set_index("Model")
st.dataframe(table.style.format("{:.3f}").highlight_max(axis=0, subset=["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1"], color="#d3f9d8"))
with st.expander("Jak czytać metryki?"):
    st.markdown(
        "- **Precision** – jaki odsetek klientów oznaczonych jako ryzykowni rzeczywiście nie spłacił kredytu "
        "(niska precyzja = odrzucamy dobrych klientów, tracimy przychód).\n"
        "- **Recall** – jaki odsetek faktycznie niewypłacalnych klientów model wychwycił "
        "(niski recall = udzielamy kredytów, które nie zostaną spłacone – strata kapitału).\n"
        "- **ROC-AUC** – zdolność modelu do rozróżniania klientów dobrych i złych niezależnie od progu "
        "(0,5 = losowo, 1,0 = idealnie).\n"
        "- **PR-AUC** – pole pod krzywą precision-recall; bardziej miarodajne przy niezrównoważonych klasach.\n"
        "- **Brier** – błąd kalibracji prawdopodobieństw (im mniej, tym lepiej).")

c1, c2 = st.columns(2)
roc = pd.concat([pd.DataFrame({"FPR": v["roc_curve"]["fpr"], "TPR": v["roc_curve"]["tpr"],
                               "Model": f"{v['name']} (AUC={v['roc_auc']:.3f})"}) for v in models.values()])
diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
with c1:
    st.markdown("**Krzywe ROC**")
    st.altair_chart(diag + alt.Chart(roc).mark_line().encode(
        x=alt.X("FPR:Q", title="False Positive Rate"), y=alt.Y("TPR:Q", title="True Positive Rate (Recall)"),
        color=alt.Color("Model:N", legend=alt.Legend(orient="bottom", columns=1, labelLimit=400))).properties(height=380))
pr = pd.concat([pd.DataFrame({"Recall": v["pr_curve"]["recall"], "Precision": v["pr_curve"]["precision"],
                              "Model": v["name"]}) for v in models.values()])
with c2:
    st.markdown("**Krzywe Precision–Recall**")
    st.altair_chart(alt.Chart(pr).mark_line().encode(
        x="Recall:Q", y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("Model:N", legend=alt.Legend(orient="bottom", columns=1, labelLimit=400))).properties(height=380))

st.markdown("**Macierze pomyłek (zbiór testowy)**")
cols = st.columns(len(models))
for col, v in zip(cols, models.values()):
    cm = v[key_thr]["confusion_matrix"]
    col.markdown(f"*{v['name']}* (próg {v[key_thr]['threshold']:.2f})")
    col.markdown(
        "| | prognoza: spłaci | prognoza: niewypł. |\n|---|---:|---:|\n"
        f"| **spłacił** | {cm['tn']} (TN) | {cm['fp']} (FP) |\n"
        f"| **niewypłacalny** | {cm['fn']} (FN) | {cm['tp']} (TP) |")

st.markdown("**Globalna ważność czynników (permutacyjna – spadek ROC-AUC po wymieszaniu cechy)**")
imp = pd.concat([pd.DataFrame({"Cecha": [F.FEATURE_LABELS[c] for c in v["permutation_importance"]],
                               "Spadek AUC": list(v["permutation_importance"].values()), "Model": v["name"]})
                 for v in models.values()])
st.altair_chart(alt.Chart(imp).mark_bar().encode(
    y=alt.Y("Cecha:N", sort="-x", title=None, axis=alt.Axis(labelLimit=300)), x=alt.X("Spadek AUC:Q"), color="Model:N", yOffset="Model:N",
    tooltip=["Model", "Cecha", alt.Tooltip("Spadek AUC:Q", format=".4f")]).properties(height=420))

if "coefficients" in models.get("logreg", {}):
    with st.expander("Współczynniki regresji logistycznej (po standaryzacji)"):
        coef = pd.DataFrame(models["logreg"]["coefficients"])
        coef["iloraz szans exp(β)"] = np.exp(coef["coef"]).round(3)
        st.dataframe(coef.head(25), hide_index=True)
        st.caption("β > 0 – cecha zwiększa szansę niewypłacalności; β < 0 – zmniejsza.")

st.markdown("**Wnioski z porównania**")
best = m["best_by_roc_auc"]
lr, rf = models.get("logreg"), models.get("rf")
if lr and rf:
    st.markdown(
        f"- Random Forest osiąga ROC-AUC **{rf['roc_auc']:.3f}** wobec **{lr['roc_auc']:.3f}** dla regresji logistycznej – "
        "lepiej wychwytuje nieliniowe zależności i interakcje (np. dochód × kwota × status mieszkaniowy).\n"
        "- Regresja logistyczna jest w pełni interpretowalna (współczynniki = wpływ cech na log-szanse), "
        "co ułatwia uzasadnienie decyzji klientowi i nadzorowi (KNF, AI Act).\n"
        f"- Najlepszy model wg ROC-AUC: **{MODEL_NAMES[best]}**.")

if auth.can(user, "model.wybor_aktywnego"):
    st.divider()
    st.subheader("Zarządzanie modelem (administrator)")
    avail = S.available_models()
    new = st.selectbox("Aktywny model scoringowy", list(avail), index=list(avail).index(settings["active_model"]),
                       format_func=avail.get)
    if st.button("Ustaw jako aktywny") and new != settings["active_model"]:
        db.set_setting("active_model", new, user["id"])
        st.success(f"Aktywny model: {avail[new]}")
        st.rerun()
    if st.button("Wytrenuj modele ponownie"):
        with st.status("Trenowanie…", expanded=True):
            train_all(log=st.write)
        db.log(user["id"], "TRENING_MODELI")
        st.rerun()
