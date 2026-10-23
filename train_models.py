"""Uruchom:  python train_models.py

Trenuje regresję logistyczną, Random Forest (i XGBoost, jeśli zainstalowany),
zapisuje modele oraz metryki w katalogu models/.
"""
from src.training import train_all

if __name__ == "__main__":
    res = train_all()
    print("\nPodsumowanie (zbiór testowy):")
    print(f"{'Model':<24}{'ROC-AUC':>9}{'Precision':>11}{'Recall':>9}{'F1':>7}  (próg max-F1)")
    for m in res["models"].values():
        b = m["at_best_f1"]
        print(f"{m['name']:<24}{m['roc_auc']:>9.3f}{b['precision']:>11.3f}{b['recall']:>9.3f}{b['f1']:>7.3f}  ({b['threshold']:.2f})")
