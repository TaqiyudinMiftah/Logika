# Learned risk representation inside frozen Kumo

Objective: reach .54 pooled KS on the existing 70,000-row, five-fold protocol,
while reporting AUC and uncertainty. The best retained exact public-linked
anchor remains KS .52929947 / AUC .84720269, with user-reported public .56470.

## Evidence and hypothesis

The teammate sessions demonstrate the value of coherent compact inputs, an
unchanged foundation backbone, and a small independently trained correction.
They also document failures of broad feature bundles and many ordinary blends.
Our completed joint-latent correction lowered KS to .528224 and AUC to .847049.
The V07/V101 matched feature audit showed that coverage and regularization can
explain or reverse apparent feature gains. A fresh unlabelled bureau check found
only about 1.6 score points of estimated monthly movement versus 23 points of
measurement noise; dynamic state estimation is not prioritized on that evidence.

Our previous five-model champion had user-reported public .56188. Its ordinary
mixture with the stronger Kumo anchor failed to improve matched validation.
However, averaging predictions forces a fixed combination rule. It does not test
whether Kumo can use the five-model ensemble's condensed signal conditionally.

Construct two learned features from those exact five feature/model views:

1. Logit of their trimmed mean (drop the highest and lowest probability and
   average the middle three, exactly as in that champion).
2. Standard deviation of their five logits, representing model disagreement.

Add these two fields to Kumo's original 90 inputs. Keep its frozen checkpoint,
50k context, eight estimators, original 12-input correction, half-strength
correction, original spline77 and 75:25 mixture. The new representation compresses
broader historical feature views into two fields; no external data is used.
This is a nested learned feature representation, not a new raw-event discovery.

## Label boundaries

For each outer fold, the original 50,000-row Kumo context is split into five
inner folds. Fit all five teacher components on 40,000 context rows and predict
the held-out 10,000. Every context borrower receives exactly one OOF prediction
per component. Every imputer, encoder, selector and reduction is fitted anew
inside that teacher training partition, including inner target cross-fitting.

Refit the five components on all 50,000 context rows to generate features for
the separate 6,000 correction and 14,000 evaluation rows. No correction/evaluation
labels enter the teachers or the Kumo context. A learner trained on 50k rather
than 40k can change feature noise at query time; that ordinary stacking limitation
is recorded. It is not hidden by fitting context features in sample.

Keep all 150 fitted teachers and their transformations, plus context assignments,
raw OOF predictions and query features. Increase only the logistic convergence
ceiling from 200 to 1,000 iterations; C, tolerance and all other recipe settings
remain as archived. This is solver completion, not hyperparameter selection.
Historical selectors named gain retain their recorded split-count semantics.

## Controls

Compare three complete five-fold pipelines:

* Original matched 90-input incumbent from the previously verified small-batch
  Kumo execution. Recheck its context IDs, raw prediction hashes and head replay.
* Primary 92-input Kumo with the two honest learned features.
* Same-width 92-input Kumo with a null representation. Randomly permute the
  context's learned feature pairs, then draw query feature pairs from those
  context pairs independently of query labels. Names, marginal feature scales,
  shape and random seeds match the primary. This checks whether adding columns
  and changing feature permutations alone explains a result.

Neither the original spline nor original corrected probabilities are used as
teacher features: they saw the 6k correction labels. The five teachers are fitted
only on the Kumo context. All corrections are fitted on their separate 6k rows.

Only one substantive feature hypothesis is registered. Five folds, inner folds,
component fits and the null control are execution repetitions, not 160 unrelated
search configurations. There is no tuning based on outer-fold results.

## Execution and assessment

Use both free PC2 RTX 3060s, one outer-fold worker per device, and up to three
one-thread teacher fits per worker. The controller waits if a GPU is occupied,
enforces its original 24-hour deadline, manages only campaign-owned descendants,
and retains failures and checkpoints. Existing data and weights remain read-only.

Validate source/input hashes, fold separation, teacher label isolation, complete
context OOF coverage, saved-model replay and real accelerator execution. Final
reports independently recompute KS/AUC, report per-fold metrics, log loss,
teacher performance and paired bootstrap intervals against both controls.

Do not call a KS-only increase a robust gain. Compare AUC and individual folds,
and disclose that all labels have historical exposure. This experiment cannot
guarantee public/private leaderboard improvement. No automatic competition CSV,
upload or replacement of the retained incumbent is included.
