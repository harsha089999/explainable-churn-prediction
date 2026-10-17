"""Explainable churn model with profit-based threshold and per-customer reason codes."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
CALL_COST, SAVE_VALUE, SAVE_RATE = 15, 400, 0.3  # NZD per call, value of a saved customer, success rate

NUM = ["tenure_months", "monthly_charges", "support_tickets"]
CAT = ["contract", "payment_method", "internet_service"]


def make_data(n=6000, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "tenure_months": rng.integers(1, 72, n),
        "monthly_charges": rng.uniform(20, 120, n).round(2),
        "support_tickets": rng.poisson(1.2, n),
        "contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2]),
        "payment_method": rng.choice(["Card", "Bank transfer", "Invoice"], n),
        "internet_service": rng.choice(["Fibre", "VDSL", "None"], n, p=[0.5, 0.35, 0.15]),
    })
    logit = (-1.5 + 1.4 * (df.contract == "Month-to-month") - 0.04 * df.tenure_months
             + 0.015 * df.monthly_charges + 0.35 * df.support_tickets
             + 0.4 * (df.payment_method == "Invoice"))
    df["churned"] = rng.random(n) < 1 / (1 + np.exp(-logit))
    return df


def build(model):
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("model", model)])


def main():
    df = make_data()
    X, y = df[NUM + CAT], df["churned"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

    models = {"logistic": LogisticRegression(max_iter=1000),
              "gradient_boosting": HistGradientBoostingClassifier(max_depth=4, random_state=0)}
    fitted, best_name, best_auc = {}, None, 0
    for name, m in models.items():
        pipe = build(m).fit(Xtr, ytr)
        p = pipe.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p)
        print(f"{name:18s} ROC-AUC {auc:.3f}  PR-AUC {average_precision_score(yte, p):.3f}")
        fitted[name] = pipe
        if auc > best_auc:
            best_name, best_auc = name, auc
    pipe = fitted[best_name]
    p = pipe.predict_proba(Xte)[:, 1]

    # pick threshold by expected profit of calling everyone above it
    thresholds = np.linspace(0.05, 0.9, 86)
    profit = [((p >= t) & yte).sum() * SAVE_RATE * SAVE_VALUE - (p >= t).sum() * CALL_COST for t in thresholds]
    t_best = thresholds[int(np.argmax(profit))]
    print(f"Chosen model: {best_name}, call threshold {t_best:.2f}, expected profit ${max(profit):,.0f}")

    imp = permutation_importance(pipe, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=0)
    drivers = pd.DataFrame({"feature": NUM + CAT, "importance": imp.importances_mean}).sort_values(
        "importance", ascending=False)
    drivers.to_csv(OUT / "drivers.csv", index=False)

    # simple reason codes: how much each feature moves the score when set to a "typical" value
    scores = df.loc[Xte.index, ["customer_id"]].copy()
    scores["churn_prob"] = p.round(3)
    typical = {c: X[c].median() if c in NUM else X[c].mode()[0] for c in NUM + CAT}
    deltas = {}
    for col in NUM + CAT:
        Xmod = Xte.copy()
        Xmod[col] = typical[col]
        deltas[col] = p - pipe.predict_proba(Xmod)[:, 1]
    deltas = pd.DataFrame(deltas, index=Xte.index)
    scores["top_reasons"] = deltas.apply(lambda r: ", ".join(r.nlargest(3).index), axis=1)
    scores["risk_band"] = pd.cut(scores.churn_prob, [0, t_best / 2, t_best, 1], labels=["Low", "Watch", "Call"])
    scores["value_at_risk"] = (scores.churn_prob * SAVE_VALUE).round(0)
    scores.sort_values("value_at_risk", ascending=False).to_csv(OUT / "customer_scores.csv", index=False)
    print(drivers.to_string(index=False))


if __name__ == "__main__":
    main()
