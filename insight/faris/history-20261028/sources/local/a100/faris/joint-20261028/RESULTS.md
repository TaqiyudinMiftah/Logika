# Joint feature + modelling campaign — 28 Oct 2026 (closing note)

Remote: `node2001:/mnt/home/faris/logika-joint-20261028`. Evaluation: pooled 70,000-row OOF KS from the original 5-fold split.

## Last step: nested check of subgroup routing
Each borrower population (A/B/C1/C2) got its own greedy blend, chosen from 66 cached candidates (Kumo views × raw/Head18/Head29/mixture, Spline200, 56k GBDTs). The blend for each outer fold was chosen using only the other four folds' labels (`nested_routed_eval.py`, log `nested_routed_eval.log`, predictions `nested_routed_oof.npz`).

| Model | Pooled KS | Fold KS |
|---|---:|---|
| Nested subgroup-routed blend | 0.530402 | .52895 .53435 .54201 .53468 .53003 |
| Nested global greedy blend | 0.529836 | .52401 .53876 .53930 .52978 .53019 |
| H016 (historical) | 0.531506 | .52635 .53658 .53970 .53165 .53202 |
| Matched baseline (maturity Large90) | 0.530636 | .52854 .53816 .54046 .53045 .52945 |

The non-nested figures from earlier in this campaign (0.5331 routed, 0.5340 prior-shifted, 0.5372 oracle thresholds, 0.531823 RefBest) were all chosen using outer-validation labels, so they are selection-biased. Under nested selection the gain disappears. **None of these figures is a supported improvement, and the 0.54 target was not reached.**

Diagnostics that still hold: an oracle with a separate threshold per fold and per population could add about 70–80 bps. That gap cannot be recovered from 6,000-row correction splits: estimating thresholds on the correction split lowered KS to 0.5303.

## Submission
`submissions.csv` (workspace root) is a byte-identical copy of `submission_best_public_V25_056822_verified.csv` (SHA-256 `e7f0748c…f58efe`). That file has the highest authenticated public score (0.56822). Its 30,000 unique IDs match the sample submission order exactly, and every probability is finite and in (0,1). It was not uploaded.
