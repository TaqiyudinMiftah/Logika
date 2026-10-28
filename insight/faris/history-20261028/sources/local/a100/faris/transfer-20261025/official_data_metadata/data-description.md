## Overview

This competition provides a **multi-table dataset** representing a consumer lending environment in Indonesia. The goal is to predict the probability that a selected loan will experience **serious repayment difficulty**, defined as reaching **90 or more days past due (DPD) within 12 calendar months after origination**. All monetary amounts are expressed in **Indonesian Rupiah (IDR)**. Timestamps use **Asia/Jakarta local time**.

---

## Data

The dataset consists of several interconnected tables:

| File                            | Description                                                       |
| ------------------------------- | ----------------------------------------------------------------- |
| `train_applications.csv`     | Loan applications with the target variable                        |
| `test_applications.csv`      | Loan applications for which predictions must be submitted         |
| `customer_profiles.csv`      | Customer demographic, employment, income, and housing information |
| `previous_applications.csv`  | Historical loan applications                                      |
| `repayment_history.csv`      | Historical installment and repayment behavior                     |
| `account_behavior.csv`       | Monthly account balance and cash-flow behavior                    |
| `bureau_observations.csv`    | Credit bureau information and credit-risk indicators              |
| `ecommerce_transactions.csv` | Historical e-commerce transaction activity                        |
| `sample_submission.csv`         | Example submission format                                         |

The tables can be connected primarily using id. Detailed column definitions are provided in **DATA_DICTIONARY.md**

---

## Target

The target is available only in `train_applications.csv`.

* `1` → the loan **reaches at least 90 DPD** within 12 calendar months
* `0` → the loan **does not reach 90 DPD** within 12 calendar months

The target represents future repayment behavior and should not be reconstructed using information occurring after the prediction cutoff.

---

## Important Notes

`decision_datetime` represents the information cutoff for the selected application. Participants should only use information that would have been available before this cutoff when creating features. Participants should avoid using information that occurs after the prediction cutoff or information derived from the future target period. No external data is required.



## Submission

Submit exactly `application_id,default_probability`, with every test application once and a finite probability in `[0,1]`. Higher values indicate greater predicted risk. Missing, extra, or duplicate IDs or invalid probabilities are rejected. Row order does not matter.
