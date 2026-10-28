**Credit risk** is an important challenge in consumer lending. Financial institutions need to identify loan applications that may lead to future repayment difficulties using information available at the time of the loan decision.

In this competition, participants are challenged to build a model that **predicts the probability** of serious repayment difficulty in an Indonesian consumer lending environment.

A loan is considered to experience serious repayment difficulty when its end-of-day **Days Past Due (DPD**) reaches 90 or more within 12 calendar months after origination.

---

# Competition Objective

The objective is to develop a predictive model that can effectively distinguish between loans that experience serious repayment difficulty and those that do not.

The competition uses the **Kolmogorov–Smirnov (KS) statistic** to evaluate how well the predicted risk scores separate the two classes. A higher KS score indicates better separation between higher-risk and lower-risk loans.

Participants are encouraged to explore the data, combine relevant information from multiple sources, and develop meaningful features that capture customer financial and behavioral patterns while avoiding data leakage.

---

# Dataset Summary

The dataset represents a consumer lending environment in Indonesia. All monetary values are expressed in **Indonesian Rupiah (IDR)**, and timestamps use **Asia/Jakarta local time**.

The dataset consists of several interconnected sources:

- **Loan Applications** — selected loan application information and characteristics.
- **Customer Profiles** — demographic, employment, income, dependency, and housing information.
- **Previous Applications** — historical loan application records.
- **Repayment History** — historical installment and repayment behavior.
- **Account Behavior** — monthly account balances, inflows, outflows, and other account activity.
- **Credit Bureau Observations** — credit scores, outstanding balances, credit utilization, inquiries, and delinquency information.
- **E-commerce Transactions** — historical purchasing activity, including product categories, payment methods, and order information.

The training dataset contains the target variable, while the test dataset contains the applications for which participants must generate predictions.

Detailed column definitions, table relationships, and data rules are available in the **Data** tab.

---

# Important Considerations

Participants must use only information that would have been available at the time of the loan decision. The `decision_datetime` represents the information cutoff for the selected application.

Participants should carefully avoid data leakage from future information and ensure that all features are constructed in accordance with the documented dataset rules.

The target variable is defined as follows:

- `1` — the loan **reaches at least 90 DPD**  within 12 calendar months after origination.
- `0` — the loan **does not reach 90 DPD**  within 12 calendar months after origination.
