# Additional five experiments V179–V183

Run after V178 has completed and been pushed. Preserve V25 as public incumbent. Kumo is100% frozen; these methods use its existing saved context scores and do not update its weights or inputs. All learnable changes fit only the same6k correction rows, with the same14k outer validation and original spline77. No hyperparameter search on outer labels.

## Evidence and reason for this batch

Direct input additions V169–V175 have not exceeded V25 pooled KS. V174 KS0.5292927367 and V175 0.5284933724 are negative results; changes in AUC/KS differ. This motivates testing small corrections to fixed baseline predictions rather than assuming more Kumo inputs help.

The strongest earlier development candidate V167 has KS0.5307954555, delta+0.0005107954 against V25, but a paired interval crossing zero, slightly lower AUC and head gap0.01610 versus0.01111. Spectral structure is therefore a candidate, not a validated improvement. The new batch targets its variance and coverage sensitivity.

| ID | Method and exact change | Why distinct from previous work | Main rejection criterion |
|---|---|---|---|
| V179 | Group ridge: baseline coefficients penalty.003, only new9 spectral-input spline/missing coefficients penalty.03; fit full correction, infer shrink.5 | V167 applied uniform.003 to11 spectral fields. Here dominant-cycle argmax is excluded and new coefficients receive10x stronger shrinkage while base penalty stays fixed | No repeated/fold improvement, or persistent larger head gap |
| V180 | Frozen V25 head plus spectral residual: eta=eta_V25+.25*w*phi*gamma; w=I(spectral_support)*orders90/(orders90+10). Fit BCE+.03/2||gamma||²+.2/2 mean(delta²) | Previous gated V172 swaps whole model predictions based on allocation history. This gates only the new residual by transaction support, preserving baseline exactly when w0 | Unsupported rows must remain exactly V25; reject worsening supported borrowers or unstable few-order segments |
| V181 | Fixed p=.5*V167+.5*V146 | Combines spectral correction and a tree-complemented model. V146 already contains80% V25 and20% HGB; no learned ensemble weights and no new model fitting | No complementary gain or lower AUC/strong segment deterioration; do not tune weights after seeing scores |
| V182 | Nystrom64 RBF basis on8 existing domain features plus missing indicators; robust scale on6k only, clip scaled numeric values[-5,5], gamma_RBF=1/d. Residual BCE+.05/2||gamma||²+.2/2 mean(delta²), inference.25 | Prior FM/polynomial interactions were negative. This tests smooth bounded local geometry, with64 learned residual weights and fixed baseline teacher | No gain after regularization, unstable support, or excessive train/validation gap |
| V183 | LightGBM residual with eta_V25 as training init_score;80 trees, depth2,4 leaves, min200 rows/leaf, lambda50, alpha1, lr.02. Add.15*clip(tree_margin,-.5,.5) at inference | Earlier tree experts predicted full risk on56k and then blended. Here only a small residual on fixed teacher logits is fitted on6k, with no outer early stopping | Training loss sanity must pass; reject weak OOF separation or worsening weak-history segments |

For V179/V180/V182/V183, final p=.75*sigmoid(corrected Kumo logit)+.25*original spline77. Test probabilities are averaged over all5 folds. V181 uses the exact completed source models' predictions; its effective V25 anchor is retained through V146.

## References and limits

- Williams and Seeger, Using the Nystrom Method to Speed Up Kernel Machines, NeurIPS2000: https://proceedings.neurips.cc/paper/2000/file/19de10adbaa1b2ee13f77f679fa1483a-Paper.pdf . Basis for low-rank kernel approximation, not evidence of a LOGIKA KS gain.
- Friedman, Hastie and Tibshirani, Additive logistic regression: a statistical view of boosting, Annals of Statistics2000: https://projecteuclid.org/euclid.aos/1016218223 . Motivation for additive logit corrections. This implementation uses LightGBM and a fixed baseline offset, not an exact reproduction of LogitBoost.
- LightGBM documentation: https://lightgbm.readthedocs.io/en/stable/Parameters.html . The existing offset must be added explicitly when using the residual model's raw prediction. A training-loss smoke check guards this interpretation.

Reliability weight10 and all regularizers/ensemble weights are prespecified experimental choices, not fitted statistical optima. Group-ridge and reliability designs are derived hypotheses. A low-rank representation can still overfit6k labels. The V25 teacher was itself fitted on these6k rows, so training gains are optimistic; only outer OOF determines candidate utility. Bootstrap intervals remain conditional on fitted scores and repeated development search.

## Operational guarantees

Analytic objective gradients are checked by finite differences before scheduling. No new transaction/DPD snapshots are introduced. Existing baseline snapshot uncertainty is inherited and not certified safe. Every run writes its source snapshots, serialized correction/adapter, OOF, per-fold validation/test scores, KS/AUC/paired CI, error segments including order-history exposure, and a separately hashed submission CSV.

CPU resource profile is0.85GiB available plus conservatively counted cache, no required VRAM. Fetch/review shared main before each run; inspect changed peer results before proceeding. Verify result artifacts before push. Push one completed result at a time; retain negative outcomes. Start only these five additional runs after V178; further experiments need a new research decision.
