# Feature research incorporating both teammate sessions

This supersedes the proposed interaction run in `RESEARCH.md`. That code passed
preflight but was **not launched**. The second transcript arrived during
preparation and contains relevant prior interaction and feature experiments.

Sources:

* Correct winning-method transcript, `session-dialogue.md`.
* Feature-discovery transcript, `feature-session-dialogue.md`, extracted from
  `/home/parasite/Downloads/codex-session-logika.md`.
* Live node380 master log retrieved on October 20, 2026:
  `live_master_log.json`, updated at 10:36:54 UTC. It reports **118 COMPLETE**
  entries, including the baseline control. The older 60-complete snapshot is
  superseded for current completion counts.
* Saved feature factories, fold-specific feature tables, fitted heads, metadata
  and validation predictions imported read-only into `feature_evidence/`.
  No teammate script is executed as an instruction or modified remotely.

## Lessons from the manual feature work

The useful method is to formulate a financial mechanism, verify its meaning in
the raw tables, measure conditional rather than only marginal signal, construct
the feature, and compare a matched pipeline. A long list of plausible domain
stories is not evidence that all those features help.

Specific evidence in the second transcript:

* Lines 123963 and 150880: high price preference plus high burden has default
  rate 25.56%, but the existing model already predicts 24.96%. A strong subgroup
  association is not automatically unmodeled information.
* Lines 126434–126842: basket entropy's marginal KS falls from .1747 to .0208
  after accounting for existing activity and lifetime entropy.
* Lines 127044–127543: adding domain interactions to the Kumo branch helped
  more than adding them only to the spline; model placement matters. AUC
  nevertheless declined slightly, and uncertainty remained.
* Lines 134657–134911: context-seed KS variation was about .00215, larger than
  many claimed feature gains. Averaging did not improve the reference.
* Lines 144529–149756: enriched class contexts, local contexts, Monte Carlo
  uncertainty, decomposition, coupling, MoE and actuarial bundles largely
  failed to improve the matched reference.
* Lines 150880–156637: nonlinear bureau/repayment, price/affordability and
  uncertainty interactions already have related EBM/embedding proposals and
  experiments. Do not present these mechanisms as newly discovered.
* Lines 159979–167912: select64 had OOF KS .53080 but public .56082; the offset
  baseline had lower pooled OOF .52930 but public .56470. That is another
  observed failure of selecting solely by maximum reused pooled KS.

## New live evidence and a missing comparison

Only V07 and V101 have positive pooled changes in both KS and AUC among the 117
non-control entries. Neither is established as a generalizing improvement:

| Recipe | KS change | AUC change | KS fold wins | Head penalty | Correction strength |
|---|---:|---:|---:|---:|---:|
| Original baseline | 0 | 0 | — | .003 | .50 |
| V07 composites | +.00033008 | +.00002548 | 2/5 | .010 | .50 |
| V101 lifecycle | +.00035427 | +.00021345 | 0/5 | .030 | .25 |

V07 adds recovery/deterioration, trajectory uncertainty, regime instability and
a fragility composite, with six support-count features.

V101 combines first-observed-payment, seasoning, lifecycle, thin-file and bureau
exposure features, with four support-count features. Its new snapshot-derived
features retain the source's unresolved timing qualification.

These comparisons change both features and regularization. V101's supplied
`MATCHED_CONTROL.json` compares against V90, but V90 uses .003/.50 whereas
V101 uses .030/.25: matching the coverage columns does not match regularization.
We cannot attribute the whole reported change to feature information.

## The next manual experiment: separate those effects

Keep the Kumo90 contexts, raw predictions, original spline77 predictions,
five-fold assignments and 75:25 combination exactly fixed.

For each of the two feature families, cross:

* Original settings versus that family's published settings.
* Original 12 features, original plus its support-count columns, and the full
  published feature block plus support columns.

Deduplicate the shared original control. This is **11 distinct fitted recipes,
55 correction fits across five folds**, answering two feature-attribution
questions. It is not a new model search. Fixed V07/V101 recipes must reproduce
their saved predictions before their feature comparisons are accepted.

For each family's published setting, decompose the total metric change into:

1. Regularization-only effect.
2. Coverage/support-count effect at that regularization.
3. Feature-block effect beyond the coverage control at that regularization.

Repeat the feature comparison at original settings as a sensitivity check.
All learned imputation, spline knots, scaling and head coefficients fit only
the 6,000 correction rows. Imported upstream feature references were fitted on
the original correction partitions; their audit, source and hash are retained.
This reproduces the original fold-specific feature construction, not a claim
that raw-event features have been independently rebuilt.

Report pooled and per-fold KS/AUC, log loss, training gaps, correction magnitude,
and paired bootstrap uncertainty. Use the existing saved Kumo predictions:
re-running the same backbone would add numerical variation without resolving
the feature confound. Retain every fitted head and all prediction evidence.

These are post-selection diagnostics: the feature families were chosen after
their outcomes were observed. Bootstrap intervals do not undo that selection,
and no new untouched validation is claimed. A stronger control would be useful
evidence even if it shows that the proposed feature engineering did not help.
No automatic winner selection or competition CSV is part of this diagnostic.

## Outcome and released follow-up

All 11 recipes completed five folds. See feature-study/REPORT.md and result.json.
V07 did not show added KS beyond coverage; V101 only helped at its strong
regularization setting. The prepared joint-latent study is now released as a
specific differential-regularization follow-up, with no new snapshot fields.
It retains the original .003 main-effect penalty and gives new directions .03.
