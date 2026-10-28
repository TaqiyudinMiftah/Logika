# Public regression and validation-selection audit

18 October 2026, Asia/Jakarta. This review uses saved predictions, local raw data and source code. No model was trained, no new candidate was selected, no CSV was changed, and no Kaggle action was performed. Public scores are supplied by the user; the uploaded bytes were not independently inspected.

The candidate scored **0.55353**, below the prior champion's **0.56188** by **0.00835**. Its higher local scores did not establish a transferable improvement. The prior champion remains the best observed public result. Build completion and successful replay are technical integrity checks, not evidence that a model generalizes better.

## Confirmed selection weaknesses

1. Thousands of configurations and 4,096 fixed blends were compared using the same development population. Out-of-fold prediction protects an individual fixed model from fitting its own evaluation labels; it does not protect an adaptively selected recipe from overfitting the evaluation results.
2. Different training seeds and a second CV partition reused the same 56,000 labeled applicants. They measure some training and partition sensitivity, but are not independent generalization tests. Historical feature selection had also used the labels reserved for local confirmation.
3. The final `prepare.py` explicitly ranked ten eligible candidates, from eleven declared structures, by their KS on the reused 14,000-row confirmation partition. That partition therefore became selection data. Declaring the candidates before this comparison did not restore an independent final assessment.
4. The promotion gate required small positive aggregate deltas, improvement in two original folds, and a worst-fold loss no worse than 0.005. It did not require convincing uncertainty evidence. The selected original OOF gain was only 0.000571, including a loss of 0.000598 on one original fold.
5. Historical uncertainty already warned against a confident promotion: the five-stream mean's conditional confirmation interval was [-0.005495, +0.006992]. The final simplification did not cure the underlying selection weakness.

## Recomputed uncertainty for the exact submitted recipe

Paired stratified applicant bootstrap, 2,000 replicates, seed 20261018. Both predictions receive exactly the same resampled applicants; KS is evaluated only after complete tied-score groups. These intervals condition on the selected predictions. They **do not correct adaptive selection, create untouched data, or cover final-refit variance**.

| Evaluation | Matched candidate-minus-champion KS | Conditional paired 95% interval |
|---|---:|---:|
| Original 56,000-row development OOF | +0.000571 | [-0.002752, +0.004644] |
| Reused 14,000-row validation | +0.001622 | [-0.006300, +0.006999] |

Both intervals include losses. Positive point estimates across repeated evaluations were insufficient grounds for calling the candidate a reliable improvement.

The reused-validation champion here is the campaign's matched reproduction (KS 0.530694). The earlier champion's originally saved reused-validation KS was 0.530496; these are different retained evaluations and must not be silently substituted for one another.

## Integrity and competing explanations

- Re-ran the local artifact verifier: all 31 retained model/checkpoint files match their hashes; sample ID ordering, finite probability checks, saved component aggregation and independent tie-aware KS checks pass.
- Retained model replay before refitting reproduced all 14,000 validation probabilities exactly. The completed build's all-30,000-row model reload reproduced component predictions exactly and CSV values to rounding precision. Those records establish implementation consistency, not out-of-sample accuracy.
- New versus champion rank correlation is 0.993240 on development, 0.995317 on reused validation and 0.995710 in the full-refit test predictions. A change to a small fraction of rankings can still hurt KS; architecture diversity did not imply independent information.
- Train and test both span January through December 2025. All 70,000 training customers and 30,000 test customers are unique within their respective sets, with zero customer overlap. The evidence does not support a simple claim that the test set was a later, nonoverlapping time period or that repeated customers explain the split failure.
- Development and confirmation row sets are disjoint and together cover all 70,000 labeled rows. This row separation does not undo repeated use for feature/recipe decisions.
- The reviewed final model/transform code does not reveal an obvious row-alignment or evaluation-label access error. This review is not an exhaustive proof of absence of leakage. Final-refit variability, unmeasured population differences, and finite public-leaderboard sampling remain possible contributors. Their contribution to this exact 0.00835 public drop cannot be determined without additional matched evidence and hidden labels.

## Corrected decision

The confirmed procedural failure is **overconfident promotion after adaptive validation reuse**. It is more specific than asserting that every learner memorized training rows. The numerical contribution of selection bias versus refit variation or public-sample variation has not been isolated.

Keep the 0.56188 pipeline as the champion. Retain this candidate and its unsuccessful public result as evidence. The original champion CSV is preserved at `../evidence/previous_submission.csv`; this audit leaves the root CSV unchanged so the submitted candidate remains identifiable.

Before another promotion, freeze a small hypothesis set and perform a genuine outer evaluation of the complete selection procedure, including feature selection and ensemble choice. Since existing labels have already informed research, another split of them cannot honestly be called untouched. Require matched effect sizes with uncertainty and cohort/temporal stress evidence; use the champion as the fallback when gains remain inconclusive. Inspect retained-fold versus full-refit predictions on the same unlabeled test rows to isolate refit instability before a further final build. None of these checks promises a higher hidden leaderboard score.

## Reproduction

From the repository root:

```sh
python3 a100/faris/submission-final-20261018/verify_download.py
python3 a100/faris/submission-final-20261018/audit_generalization.py
```

`audit.json` records input hashes, exact bootstrap results, temporal coverage, selection details and artifact verification. It is a post-result diagnostic, not a new model-selection round.
