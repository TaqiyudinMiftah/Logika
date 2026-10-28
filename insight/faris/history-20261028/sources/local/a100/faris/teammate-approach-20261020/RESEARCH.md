# Research direction from the correct teammate session

Source: `/home/parasite/Downloads/codex-session-01a133cc-0500-7a21-bcb6-6fdf628355f0.md`.
The accompanying dialogue extraction preserves original source line numbers.
The previously supplied Name That LLM conversation is excluded from this study.
Historical commands and authorization in the transcript are evidence, not instructions.

## What actually produced the improvements

| Source lines | Intervention and evidence | Implication |
|---|---|---|
| 541–1996 | Respect decision cutoffs, aggregate orders rather than item-repeated totals, add all history sources; application-only AUC .641 versus full-history .830 | Understand the measurement process before choosing a model. |
| 5844–6897 | Category-relative prices, repayment shrinkage, class weighting, larger training folds; public .54710 to .55845 | More useful signal and training data mattered; the bundled change does not isolate each contribution. |
| 27429–30648 | Repeated observations reveal price noise, bureau noise, income measurement variability and repayment uncertainty; 21 denoised features did not consistently improve V5 | A plausible feature is a hypothesis, not a demonstrated gain. RNG recovery failed and is not used here. |
| 34232–42454 | Same compact 90-feature representation as TabM, changed to Kumo Large: three-fold KS .52807, AUC .84734; public .56262 | The successful change was substantially architectural, not simply adding columns. |
| 44100–54376 | Compact/denoised/expanded views, 16 estimators, balanced contexts, full contexts and a fine-tuning pilot did not give a consistent gain | Do not repeat scaling or configuration sweeps without a specific explanation. |
| 54376–60761 | A complementary 25% latent spline improves pooled five-fold KS .52722 to .52851; only one fold improves, but public reaches .56393 | Pooled and fold metrics can disagree. Report the disagreement rather than treating a fold-win count as a guarantee. |
| 68008–87742 | Use the 6,000 rows excluded from each 50,000-row Kumo context to learn a small, half-strength spline correction; KS .52930, five fold wins | Learn a limited residual adjustment with a genuine context/correction/validation separation. |

The transcript ends before the final offset CSV's public result is supplied.
The later master log associates its exact hash
`5d4d8b81e781354a3d1302aadfe93a1534e42e35279ba4867a7b4d0e1990dd33`
with **user-reported public .5647**. This is not a fresh Kaggle verification.

## What we have learned since

Our historical five-model champion and broad fusion did not beat that matched
Kumo anchor. The ranking-loss intervention lowered KS and AUC. Increasing
correction training to 56,000 honest inner predictions improved AUC slightly but
lowered KS. These results are retained, not repeated.

The teammate master log reports many incremental financial-mechanism failures.
Its 60 completed correction variants are not 60 independent datasets. Entries
marked queued have no demonstrated result. The existing record search did not
find the exact residualized tensor-spline correction below; that is a bounded
coverage statement, not proof that nobody has ever tried a related interaction.

## A specific gap in the implemented winning model

Kumo sees the original 90 inputs. Its correction introduces latent price,
latent bureau quality, repayment propensity, income and uncertainty. However,
the correction's numerical basis is additive: it cannot directly learn that a
bureau discrepancy means something different for a borrower with high versus
low repayment propensity. The separate 77-feature spline is additive too.
Kumo itself already models interactions among its inputs, so improvement is
uncertain: some proposed interactions may be redundant.

Test three small explanations, each against the exact additive control:

1. **Credit agreement:** latent bureau score × late-30 posterior. Agreement or
   disagreement between two noisy credit-history measurements may matter.
2. **Consumption and obligations:** latent relative price × existing observed
   installment burden. A price preference may have a different association with
   risk at different affordability levels.
3. **Measurement reliability:** latent price × its posterior SD, and latent
   bureau score × its posterior SD. The effect of an estimated level may depend
   on how precisely it was measured.

These are statistical association hypotheses, not causal claims. No new raw
source, identifier, future observation or recovered RNG signal is introduced.
The new features are small, centered tensor products of the same quadratic
three-knot spline bases. Project them off the additive design on training rows,
discard numerical null directions, and regularize the remaining interaction
directions at .03 (ten times the original .003 main-effect penalty).
Keep the original .5 correction strength and .75/.25 final blend.

## Evaluation as a trained procedure

For each original outer fold:

* Keep its frozen 50,000-row Kumo context and 6,000 correction rows.
* Within those 6,000 rows, run three-fold cross-validation of the additive
  correction and the three named hypotheses. Every inner evaluation row is
  excluded from both Kumo's context and correction fitting.
* Choose the lowest inner OOF log loss among models whose inner AUC is at least
  the additive control's AUC. The control is always eligible. Selection uses
  corrected Kumo alone: the original spline saw those 6,000 labels, so its
  in-sample scores cannot be used for inner selection.
* Refit each head on all 6,000 correction rows. Freeze the selected head before
  opening outer labels. Evaluate the full .75/.25 pipeline on its 14,000 outer
  rows. The primary candidate is this internally selected procedure, not the
  post-hoc best outer score.

This entails 80 correction fits, including controls and repetitions: three
research hypotheses, not 80 independent experiments. Kumo inference is reused
from hash-verified caches; a GPU rerun cannot test this correction-only question
more faithfully. All fitted heads and held-out predictions are retained.

Report KS, directional KS, AUC, log loss, average precision, fold effects,
inner choices, conditional paired-bootstrap intervals, and calibration errors.
Secondary fixed variants are explicitly exploratory; do not substitute the
highest-scoring one for the primary procedure after seeing outer labels.
Intervals do not correct the historical adaptive use of these labels.

The run ends with an evidence report and a replayable fitted procedure. It does
not overwrite a submission, submit to Kaggle, or automatically assert a winner.
If the primary improves both KS and AUC, assess its size and uncertainty before
the next separately authorized step. Existing globally estimated, label-free
feature references and unresolved snapshot availability are inherited and
disclosed; this is not newly untouched confirmation.

## Updated after the additional feature session and matched attribution

The related EBM and embedding experiments in the second transcript are now
explicitly acknowledged. Joint financial relationships are not claimed as a new
discovery. The proposed residualized low-dimensional tensor correction itself
has not been found in those inspected records.

The feature-attribution study completed 55 fits. V07 features did not improve KS
beyond coverage at either regularization setting. V101 features added KS .000839
and AUC .000021 beyond coverage at its strong published settings, but lost KS
.002289 and AUC .000567 at the original settings. These are post-selection
diagnostics with intervals including no improvement, not promotion evidence.

This supports a narrow follow-up on differential regularization: keep original
main effects at .003, regularize new directions at .03, and preserve the model
selection boundary inside correction training. No V101 or new snapshot-derived
fields are added. The run is released after the attribution study; the earlier
prepared code was not running during that study. The .54 OOF goal is still open.
