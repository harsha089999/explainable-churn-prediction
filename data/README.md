# Data

Real data, not simulated. Files are the original CSVs, gzip-compressed to keep the repo small. pandas and DuckDB read
`.csv.gz` directly, so nothing needs unzipping.

**Dataset:** IBM Telco Customer Churn sample dataset

- [Kaggle: Telco Customer Churn (blastchar)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [IBM source repository](https://github.com/IBM/telco-customer-churn-on-icp4d)

**Licence:** IBM sample data, redistributed on Kaggle and GitHub. See the source repository for terms.

| File | Contents |
|---|---|
| `telco_churn.csv.gz` | 7,043 customers, 21 columns: demographics, services, contract, billing, tenure and whether they churned |

To refresh from Kaggle instead, download the dataset with the Kaggle CLI (`kaggle datasets download <owner/dataset>`),
then gzip the needed files into this folder with the same names.
