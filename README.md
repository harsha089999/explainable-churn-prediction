# Explainable Churn Prediction

[![Run project](https://github.com/harsha089999/explainable-churn-prediction/actions/workflows/run.yml/badge.svg)](https://github.com/harsha089999/explainable-churn-prediction/actions/workflows/run.yml)

**Predict which telecom customers will leave, explain why for each one, and pick who to call based on cost and team capacity.**

> **About the data:** this project runs on a real public dataset, **IBM Telco Customer Churn sample dataset**, included in `data/` so it runs straight after cloning. Sources and licence are listed below and in [`data/README.md`](data/README.md).

## The business question

Which customers are most likely to cancel, why, and who should the retention team call first?

## What this project does

A churn score on its own isn't enough. The retention team needs to know who to call, what to talk about, and whether the calls are worth making. This project trains two models on the real IBM Telco dataset, chooses a call threshold by expected profit, shows how good the top of the list is when calls are limited, and gives every customer their top three reasons.

## The data

7,043 real (anonymised) telecom customers from IBM's sample dataset, with a 26.5% churn rate. The model uses tenure, monthly and total charges, contract type, payment method, internet service, online security, tech support, paperless billing, senior citizen and dependents. `TotalCharges` is stored as text with 11 blanks, all for brand new customers with zero tenure, so they're set to 0.

**Source:**
- [Kaggle: Telco Customer Churn (blastchar)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [IBM source repository](https://github.com/IBM/telco-customer-churn-on-icp4d)

**Licence:** IBM sample data, redistributed on Kaggle and GitHub. See the source repository for terms.

## How it works

1. Clean the raw file and do a stratified 75/25 split.
2. Train logistic regression and gradient boosting in the same scikit-learn pipeline (scaling plus one-hot encoding).
3. Compare on ROC-AUC and PR-AUC; PR-AUC matters because churners are the minority.
4. Choose a call threshold by expected profit. The call cost ($15), value of a saved customer ($400) and save rate (30%) are business assumptions, not data.
5. Capacity view: if the team can only make 100, 250 or 500 calls, how many of those people actually churn?
6. Permutation importance for global drivers, then per-customer reason codes: reset each feature to a typical value and see how far the score moves.

## Run it

```bash
git clone https://github.com/harsha089999/explainable-churn-prediction
cd explainable-churn-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

Outputs are written to `outputs/`. Python 3.10 or newer is recommended.

## Results from the included run

This is the actual console output when the script is run as shipped:

```text
7,043 customers, churn rate 26.5%

logistic           ROC-AUC 0.841  PR-AUC 0.625
gradient_boosting  ROC-AUC 0.842  PR-AUC 0.648

Chosen model: gradient_boosting. Call threshold 0.10: call 1089 of 1761 test customers, catching 442 of 467 churners. Expected profit $36,705 (vs $35,685 calling almost everyone)

If the team can only make N calls (top N by score):
 calls  churn_rate_in_list  lift_vs_random
   100               0.800            3.02
   250               0.708            2.67
   500               0.590            2.22

Drivers (drop in ROC-AUC when shuffled):
         feature  importance
        Contract      0.0730
          tenure      0.0459
  MonthlyCharges      0.0223
 InternetService      0.0108
  OnlineSecurity      0.0069
     TechSupport      0.0056
   PaymentMethod      0.0035
    TotalCharges      0.0029
PaperlessBilling      0.0021
   SeniorCitizen      0.0015
      Dependents      0.0004

Actual churn rate by risk band (test set):
           size   mean
risk_band             
Low         479  0.019
Watch       194  0.082
Call       1088  0.406
```

Both models land around 0.84 ROC-AUC. Gradient boosting has the better PR-AUC (0.65 vs 0.63), so it's used. Contract type is the biggest driver by a distance, followed by tenure and monthly charges, which matches what most telcos see. With the assumed economics a call is cheap next to a saved customer, so the profit maths says call almost everyone above 10% risk. That's a real result, but no team has that capacity. The capacity view is more useful: the top 100 scored customers churn at 80%, three times the base rate. The risk bands hold up on unseen data too, with actual churn of 2% in Low, 8% in Watch and 41% in Call.

## Output files

| File | What it contains |
|---|---|
| `outputs/customer_scores.csv` | Per test customer: churn probability, actual outcome, top three reasons, risk band and value at risk. |
| `outputs/drivers.csv` | Permutation importance (drop in ROC-AUC when a feature is shuffled). |
| `outputs/capacity_view.csv` | Churn rate and lift in the top 100, 250 and 500 customers by score. |

## Limitations and honest caveats

- This is a single snapshot, so it can't show when a customer will leave. Survival analysis would need dated history.
- Call cost, customer value and save rate are assumptions. The threshold changes a lot if they change.
- Reason codes are a simple perturbation method, not SHAP. They're easy to explain, but they ignore interactions.
- A customer's reasons show what the model leans on. They aren't proof of what caused that customer to leave.

## Turning it into a dashboard

A ranked call list filtered by risk band with reason codes in plain words, a drivers bar chart, and a slider for call capacity that updates the expected churners reached.

## Ideas to extend it

- Add SHAP values and compare them with the simple reason codes.
- Calibrate probabilities (isotonic) before using them for value at risk.
- Uplift modelling: target people a call would change, not just people likely to leave.

## Skills demonstrated

Classification, imbalanced metrics, cost-based thresholds, explainability, turning a model into a call list.

**Tools:** Python, pandas, scikit-learn

## Project structure

```text
explainable-churn-prediction/
├── src/main.py                 # the whole pipeline
├── data/                       # real source data (gzipped) + data/README.md with source and licence
├── outputs/                    # files from the run shown above
├── requirements.txt
├── .github/workflows/run.yml   # CI: installs and runs the project on every push
└── LICENSE
```

## Licence

Code: MIT, see [LICENSE](LICENSE). The data in `data/` keeps its original licence (see above).
