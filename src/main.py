"""Explainable churn model on the real IBM Telco dataset, with a profit-based call threshold and reason codes.

Data: IBM Telco Customer Churn sample (7,043 customers), also published on Kaggle as blastchar/telco-customer-churn.
Stored in data/telco_churn.csv.gz.
"""
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
DATA = ROOT / "data" / "telco_churn.csv.gz"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
# business assumptions (not in the data): cost of a retention call, value of a saved customer, success rate
CALL_COST, SAVE_VALUE, SAVE_RATE = 15, 400, 0.3

NUM = ["tenure", "MonthlyCharges", "TotalCharges"]
CAT = ["Contract", "PaymentMethod", "InternetService", "OnlineSecurity", "TechSupport",
       "PaperlessBilling", "SeniorCitizen", "Dependents"]


def load():
    df = pd.read_csv(DATA)
    # TotalCharges is text with 11 blanks, all for customers with tenure 0 (not billed yet)
    df["TotalCharges"] = pd.to_numeric(df.TotalCharges, errors="coerce").fillna(0.0)
    df["SeniorCitizen"] = df.SeniorCitizen.map({0: "No", 1: "Yes"})
    df["churned"] = (df.Churn == "Yes").astype(int)
    return df


def build(model):
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("model", model)])


def main():
    df = load()
    print(f"{len(df):,} customers, churn rate {df.churned.mean():.1%}\n")
    X, y = df[NUM + CAT], df["churned"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

    models = {"logistic": LogisticRegression(max_iter=1000),
              "gradient_boosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                  max_iter=200, random_state=0)}
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
    profit = [((p >= t) & (yte == 1)).sum() * SAVE_RATE * SAVE_VALUE - (p >= t).sum() * CALL_COST
              for t in thresholds]
    i = int(np.argmax(profit))
    t_best = thresholds[i]
    called = (p >= t_best)
    print(f"\nChosen model: {best_name}. Call threshold {t_best:.2f}: call {called.sum()} of {len(p)} test customers, "
          f"catching {(called & (yte == 1)).sum()} of {yte.sum()} churners. Expected profit ${profit[i]:,.0f} "
          f"(vs ${profit[0]:,.0f} calling almost everyone)")

    # a retention team can only make so many calls: how good is the top of the list?
    order = np.argsort(-p)
    base = yte.mean()
    cap_rows = []
    for k in [100, 250, 500]:
        hit = yte.to_numpy()[order[:k]].mean()
        cap_rows.append({"calls": k, "churn_rate_in_list": round(hit, 3), "lift_vs_random": round(hit / base, 2)})
    cap = pd.DataFrame(cap_rows)
    cap.to_csv(OUT / "capacity_view.csv", index=False)
    print("\nIf the team can only make N calls (top N by score):\n" + cap.to_string(index=False))

    imp = permutation_importance(pipe, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=0)
    drivers = pd.DataFrame({"feature": NUM + CAT, "importance": imp.importances_mean.round(4)}).sort_values(
        "importance", ascending=False)
    drivers.to_csv(OUT / "drivers.csv", index=False)
    print("\nDrivers (drop in ROC-AUC when shuffled):\n" + drivers.to_string(index=False))

    # reason codes: how much each feature moves the score when reset to a typical value
    scores = df.loc[Xte.index, ["customerID"]].copy()
    scores["churn_prob"] = p.round(3)
    scores["actual_churn"] = yte.values
    typical = {c: X[c].median() if c in NUM else X[c].mode()[0] for c in NUM + CAT}
    deltas = {}
    for col in NUM + CAT:
        Xmod = Xte.copy()
        Xmod[col] = typical[col]
        deltas[col] = p - pipe.predict_proba(Xmod)[:, 1]
    deltas = pd.DataFrame(deltas, index=Xte.index)
    scores["top_reasons"] = deltas.apply(lambda r: ", ".join(r[r > 0.01].nlargest(3).index) or "none", axis=1)
    scores["risk_band"] = pd.cut(scores.churn_prob, [-0.01, t_best / 2, t_best, 1], labels=["Low", "Watch", "Call"])
    scores["value_at_risk"] = (scores.churn_prob * SAVE_VALUE).round(0)
    scores.sort_values("value_at_risk", ascending=False).to_csv(OUT / "customer_scores.csv", index=False)
    band = scores.groupby("risk_band", observed=True).actual_churn.agg(["size", "mean"]).round(3)
    print("\nActual churn rate by risk band (test set):\n" + band.to_string())


if __name__ == "__main__":
    main()
