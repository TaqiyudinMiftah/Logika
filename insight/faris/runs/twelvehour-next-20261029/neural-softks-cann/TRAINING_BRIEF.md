# Continuous training brief — logika-next-20261029

This sidecar records the campaign; it supplements the project's `AGENTS.md`.
Include its full content in every worker prompt and read applicable project
instructions before executing the assignment.

- Competition root and worker working directory: /mnt/home/faris/logika-twelvehour-next-20261029
- Metric: pooled absolute binary KS
- Direction: higher
- Train data, read only: /mnt/home/faris/logika-faris-exploration/exploration-12h-20261017/raw/train_applications.csv
- Test for prediction, read only: /mnt/home/faris/logika-faris-exploration/exploration-12h-20261017/raw/test_applications.csv
- Compute: 11explicit Colab profiles; lab2001 CPU4slots, lab2003 CPU4slots and2idleRTX3060, host119 CPU6slots and persistent control;2threads perfit
- Campaign start UTC: 2026-10-29T07:18:48.643995+00:00
- Campaign end UTC: 2026-10-29T19:18:48.643995+00:00
- Original hours: 12.0
- Live deadline check: `DEADLINE.md`
- Supplied rules file: /mnt/home/faris/logika-twelvehour-next-20261029/evidence/rules.md
- Rules SHA-256: <rules_sha256>

## Supplied competition rules

## 1. Eligibility

- The competition is open to active undergraduate students (S1/D4/D3/D2/D1) enrolled at an Indonesia university during the odd semester of the 2026/2027 academic year.
- Each team must consist of **2–3 members**.
- Team members are allowed to come from different universities.
- Each participant may only be registered in **one team** for this competition.
- Participants must complete the official registration process according to the competition guidelines.
- The team name used on the Kaggle platform must correspond to the team name registered with the competition committee.

---

## 2. Competition Participation

- Participants must follow all competition procedures and regulations established by the DSC LOGIKA UI 2026 committee.
- Participants are responsible for ensuring that all information submitted during registration and throughout the competition is accurate.
- Teams that qualify for the final round must complete the required re-registration process.
- Participants who qualify for the final round but fail to complete the required re-registration process may be considered to have withdrawn from the competition.
- All participants are expected to follow official announcements and instructions provided by the committee throughout the competition.

---

## 3. Dataset and Modeling Rules

- Participants may only use the datasets provided by the competition committee.
- The use of **external datasets, external data sources, or additional data not provided by the committee is prohibited**.
- Participants may perform data preprocessing, feature engineering, data aggregation, visualization, and model development using the provided datasets.
- Participants must follow the documented relationships between the provided tables.
- Participants must respect the information cutoff defined in the dataset documentation.
- Features must only use information that would have been available at the `decision_datetime` of the selected application.
- Participants must not use future information related to the target period or otherwise introduce data leakage.
- Participants are responsible for understanding the provided `DATA_DICTIONARY.md` before developing their models.
- No external data is required for this competition.

---

## 4. Programming Language

- Participants are required to use **Python** for the competition.
- Participants may use appropriate Python libraries for data processing, feature engineering, visualization, and machine learning.

---

## 5. Submission Rules

- Participants must submit their predictions through the designated **Kaggle competition platform**.
- Each submission must follow the required submission format.
- The submission must contain exactly the following columns:

```text
application_id,default_probability
```

- Every test application must have exactly one prediction.
- `default_probability` must be a valid numeric value between `0` and `1`.
- Missing, duplicated, or additional `application_id` values are not allowed.
- The number of predictions must match the number of applications in the test dataset.
- Participants are responsible for checking their submission before submitting it.

---

## 6. Evaluation and Elimination Round

The elimination round consists of **Kaggle evaluation and notebook evaluation**.

### Kaggle Score — 90%

The Kaggle score contributes **90%** of the total elimination-round assessment.

The Kaggle score is divided into:

- **Public Score — 45%**
- **Private Score — 45%**

The competition leaderboard uses the **Kolmogorov-Smirnov (KS) Statistic** as the evaluation metric.

A higher KS score indicates better separation between loans that experience serious repayment difficulty and those that do not.

### Notebook Assessment — 10%

The submitted notebook contributes **10%** of the total elimination-round assessment.

The notebook will be evaluated solely based on:

- **Reproducibility and code execution — 10%**


Therefore, the total elimination-round score is:

```text
Public Score       45%
Private Score      45%
Notebook           10%
-----------------------
Total             100%
```

---

## 7. Notebook Submission

- Each team must submit a notebook containing the code used during the competition.
- The notebook must be reproducible and executable.
- The notebook should be able to run successfully in the designated environment without requiring unauthorized external data.
- The submitted notebook must correspond to the team's competition approach and submission.
- Participants must ensure that all required files, dependencies, and paths are correctly configured.
- The notebook must be submitted through the submission mechanism provided by the committee.
- The notebook must follow the required naming format:

```text
ParticipantNumber_TeamName_Notebook
```

Example:

```text
DSC0001_RandomSeed_Notebook
```

- The notebook submission deadline follows the official competition schedule. According to the competition guidebook, the elimination-round notebook submission deadline is **30 October 2026 at 23:59 WIB**.

---

## 8. Late Notebook Submission

Participants are responsible for submitting their notebook before the specified deadline.

For the elimination-round notebook submission:

- A late submission will receive a **1-point deduction per minute** after the deadline.
- Participants should ensure that their submission has been successfully uploaded before the deadline.
- The deduction applies according to the official competition guidelines.

---

## 9. Advancement to the Final Round

- The **eight teams with the highest qualifying scores** in the elimination round will advance to the final round.
- Finalists must complete the required re-registration process.
- Finalists must attend the Final Technical Meeting.
- Finalists must submit the required final report and presentation materials according to the schedule provided by the committee.
- The final round will be conducted offline at the **Universitas Indonesia, Depok**.
- Further technical requirements for the final round will be explained during the Final Technical Meeting.

---

## 10. Final Round

- The final round will consist of a presentation and submission of the required final report.
- Each finalist team will be given a designated presentation time.
- Finalists must follow the presentation format and time allocation determined by the committee.
- Final-round submission requirements and technical details will be communicated during the Final Technical Meeting.
- Final-round decisions made by the judges and committee are final and cannot be contested.

---

## 11. Prohibited Actions

The following actions are prohibited:

- Registering a participant in more than one team.
- Submitting false or misleading registration information.
- Using external datasets or data sources that are not provided by the competition committee.
- Using information that causes data leakage from the future target period.
- Manipulating or attempting to manipulate the competition scoring system or leaderboard.
- Submitting predictions generated from unauthorized sources or methods.
- Copying or plagiarizing another team's work without proper attribution.
- Engaging in any form of cheating, fraud, or other actions that provide an unfair advantage.
- Violating the competition rules or instructions provided by the committee.
- Any other action that is proven to provide an unfair advantage over other participants.

---

## 12. Disqualification and Sanctions

Participants or teams proven to have violated the competition rules may receive sanctions determined by the committee.

Violations may include, but are not limited to:

- Registering in more than one team.
- Failing to meet the required administrative conditions.
- Committing fraud or other forms of cheating.
- Violating competition rules during any stage of the competition.
- Providing false information.
- Using unauthorized data or resources.
- Engaging in actions that provide an unfair advantage.

Depending on the severity of the violation, sanctions may include:

- Score reduction.
- Submission invalidation.
- Removal from the leaderboard.
- Disqualification from the competition.
- Revocation of finalist status.
- Revocation of winner status and/or awards, where applicable.

The committee reserves the right to determine the appropriate sanction based on the nature and severity of the violation.

---

## 13. Data Privacy

Participants must not attempt to identify real individuals, organizations, or locations from the provided data.

Any data or information obtained through unauthorized means must not be used in the competition.

---

## 14. Changes to the Rules

The competition committee reserves the right to modify or update the competition rules when necessary.

Any changes will be communicated through the official DSC LOGIKA UI 2026 communication channels.

Participants are responsible for keeping up to date with official announcements throughout the competition.

---

## 15. Final Note

By participating in this competition, participants agree to comply with all applicable competition rules, technical requirements, and decisions made by the DSC LOGIKA UI 2026 committee.

For detailed information about the dataset, columns, relationships, prediction cutoff, and submission requirements, please refer to the accompanying competition documentation.


## Campaign boundaries

- Train only on supplied train labels. Test targets remain unseen. Do not decode
  hidden labels from caches, artifacts, embeddings, or prediction files.
- No submissions, uploads, leaderboard calls, or leaderboard probing during this
  campaign. Prediction files are held for a subsequent explicit user command.
- Default team bound: at most three running workers plus one coordinator; use a
  lower number if the runtime or measured compute allows fewer.
- Start workers in the competition root. Forks inherit the parent's currently
  persisted transcript; fresh sessions have only their supplied prompt.
- Each worker writes only its assigned `models/<run_id>/` folder. Read shared data,
  project instructions, frozen splits, code, and prior evidence without modifying
  them. The coordinator owns shared ledgers, worker state, and held predictions.
- Record one hypothesis before training and retain executable code, logs, exact
  split assignments, OOF predictions, package versions, and finite fold scores.
  A reply or score JSON alone does not establish a completed experiment.
- The coordinator independently recomputes local metrics before accepting gains.
  Only the coordinator promotes a prediction file to `submissions_held/` after
  row count, ID order, headers, finiteness, and task constraint checks.
- Check the preserved deadline before each run. Reserve time for verification and
  reporting, and return partial status if interrupted or out of time.
