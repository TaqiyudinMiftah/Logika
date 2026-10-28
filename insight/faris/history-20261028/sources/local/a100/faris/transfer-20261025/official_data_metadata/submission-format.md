## Submission Format

Participants must upload a submission file in **CSV format** using the required format. The `sample_submission.csv` file is available in the **Data** tab on Kaggle and can be used as a reference for preparing your submission.

The submission must contain the following columns:

```text
application_id,default_probability
```

### Sample Submission

```csv
application_id,default_probability
A_000001,0.1523
A_000002,0.7812
A_000003,0.0437
A_000004,0.3265
A_000005,0.9142
```

The `application_id` must correspond to an application in the test dataset, while `default_probability` represents the predicted probability of serious repayment difficulty. Values must be valid numeric probabilities between **0 and 1**, where higher values indicate greater predicted risk.

Please ensure that:

- Every `application_id` from the test dataset appears exactly once.
- No application IDs are missing, duplicated, or outside the test dataset.
- The number of rows matches the test dataset.
- All `default_probability` values are finite numbers between 0 and 1.

The order of the rows does not matter.
