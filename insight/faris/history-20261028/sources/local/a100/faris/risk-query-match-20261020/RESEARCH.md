# Training-size-matched learned query features

This one follow-up is specified before the parent research result is known. It
shares the parent's original deadline of October 21, 2026, 19:31:11 WIB. It
waits for that complete evaluation and independent audit. If the parent already
reaches .54 KS with nondecreasing AUC, this follow-up skips training.

The learned-feature audit uses no selected-default evaluation labels. Across
the first four prepared folds, context/query distribution KS was .011–.015 for
risk logit and .036–.056 for model disagreement. Context features are produced
by teachers trained on 40k rows; original query features are from 50k-row
teachers. The size difference may partly explain this shift. It is an ordinary
stacking limitation, not proof of label leakage or proof of impaired default KS.

Reuse all the already fitted teachers, transforms and Kumo contexts. For each
outer fold, randomly assign its 20k correction/evaluation queries evenly among
the five retained inner teacher folds. Use that fold's five 40k-row teachers to
produce the query's ensemble risk and disagreement. Query assignment uses a
fixed RNG and no query labels or identifiers. All 40k-row teacher training sets
are subsets of the 50k Kumo context and exclude every query borrower.

This gives queries the same teacher training size as the 50k context's honest
OOF features. Context features are unchanged, so the retained fitted Kumo
context and preprocessing remain exactly applicable. Replaying that context
requires no backbone fit and no teacher training. Refit only the original
12-input correction on the separate 6k correction rows; preserve its penalty,
half strength, original spline77 and 75:25 mixture.

The primary comparison is the matched-query pipeline versus the original query
pipeline, the same-width shuffled control, and the original historical anchor.
Run all five outer folds without performance pruning. Retain all new heads,
query assignments, raw component predictions and final OOF evidence. All old
models and context caches remain in the parent directory, referenced by hash.

Independently verify the parent before admission, replay a saved context on the
same original 512 queries to check serialization, verify all 40k teacher fitting
sets exclude all query rows, and verify full OOF coverage and KS/AUC. Report
paired fold/class-bootstrap intervals. Reused labels and historical adaptation
remain explicit. No competition CSV or upload is created.
