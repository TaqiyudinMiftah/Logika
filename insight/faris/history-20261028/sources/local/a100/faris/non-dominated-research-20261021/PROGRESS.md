# Historical metric audit and next action

All 36,136 discoverable saved screening/legacy ensemble prediction files passed identity and independently recomputed KS checks. AUC, log loss and per-fold metrics were recomputed. All 4,096 existing later fixed ensembles replayed from their saved parents and matched recorded KS. All 118 teammate V7 saved OOF files matched their recorded KS/AUC/log loss.

These 40,350 metric records include ensembles and repeated references. They are not 40,350 independent fitted pipelines. The 56,000-row three-fold and 70,000-row five-fold protocols remain separate. Missing/unavailable histories and unresolved configuration semantics remain explicit in inventory/.

The highest V7 pooled OOF KS is V80 at 0.5307369203, with AUC 0.8470688524 and only two of five fold KS wins over V00. It is not automatically promoted. V25 retains the highest user-reported public score, 0.56822. No new submission was produced.

Executed-algorithm descriptors were calculated for 20,033 distinct legacy designs. Their GPU distance engine matched 96 exact helper comparisons with maximum absolute error 7.57e-8; this validates distance mechanics, not scientific evaluation.

Source inspection found full-training covariate fitting for inherited price, bureau and repayment priors. A context-only preprocessing rebuild is now running on PC2 for fold0. It receives constant target placeholders and preserves historical artifacts. Model training remains unadmitted until its boundaries and the matched new control are ready. Raw DPD snapshot provenance remains separately unresolved.

The 0.54 pooled OOF goal has not been reached. See validation/ALL_METRIC_AUDIT_SUMMARY.json and research-state.json.
