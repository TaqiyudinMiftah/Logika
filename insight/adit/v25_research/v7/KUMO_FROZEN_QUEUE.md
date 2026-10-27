# Frozen Kumo experiments after V25

All backbone parameters have `requires_grad=False`, inference mode and no optimizer. `fit` means labeled in-context adaptation, not gradient fine-tuning. Preserve all five outer folds,50k context,6k correction and14k evaluation per fold. Correction is trained only on6k labels. Original spline77 remains fixed.

| Run | Change | Hypothesis |
|---|---|---|
| V168 | Fresh90-input runtime control | Quantify new Torch/CUDA numerical differences before attributing gains to inputs |
| V169 | Add V25's six raw coverage/conditional-liquidity counts directly to Kumo | Foundation-model feature interactions may exploit evidence currently seen only by the correction |
| V170 | V169 plus six spending velocity/liquidity ratios | Encode recent acceleration relative to prior spending and observed ability to absorb the new installment |
| V171 | V169 plus nine spectral inputs | V167 is best development result, but has larger head gap and slightly lower AUC; remove dominant-cycle argmax and keep correction size fixed |
| V172 | V171 when allocation evidence exists; otherwise retain V25 | Prior OOF errors worsen in zero-allocation group; use fixed observable-coverage gate, with selection-bias caveat |
| V173 | Two stratified40k contexts from same eligible50k pool, V169 inputs | Assess sampling robustness without increasing correction complexity or updating Kumo weights |

The spending formulas and exact denominators are in `CONTEXT_INPUT_AUDIT.json`. Undefined ratios remain missing. Spending uses completed purchases; item composition uses item_total, not repeated order_total. No new DPD snapshot derivative is introduced. Legacy baseline snapshot uncertainty is inherited and not certified safe.

SDM0.2.1 automatically offloads context caches to pinned CPU memory. The per-instance cache patch skips that copy, preserving tensor values, estimator ordering and preprocessing. A512-context/64-query test against the original implementation must pass with maximum prediction error<=1e-6. The installed library and peer environments are unchanged.

After each completed run: inspect shared GitHub HEAD and changes, preserve methods/formulas/checkpoints/OOF/error analysis/CI/submission, push own research namespace, verify push, then review the next run. An unreviewed peer commit stops automatic advancement for inspection; no force push. Every model remains frozen. Negative results are retained. V25 remains public incumbent; development-fold selection and bootstrap uncertainty do not prove leaderboard improvement.

## Evidence and selection

The initial five experiments after V168 are V169 through V173. The unrun earlier draft is preserved in KUMO_SUPERSEDED_QUEUE_DRAFT.json. The user subsequently authorized five additional, reviewed hypotheses V174 through V178, documented in KUMO_EXTRA_QUEUE.md; they wait for V173 to finish and be pushed. Do not start any further runs beyond these ten automatically.

V167 pooledKS0.5307954555 versus V25 0.5302846600, with delta95% CI[-0.00165264,+0.00295812]. AUC decreases0.00005090 and correction train/validation KS gap rises from0.01111 to0.01610. Allocation-count zero group has deltaKS -0.002942, while count1:3 improves+0.004588. Variable-income group improves+0.001693. These are exploratory development associations, not causal or independent selection evidence.

V168 fresh runtime control yields KS0.5302020059 and AUC0.8470296023. DeltaKS versus original V25 is -0.00008265 with paired CI[-0.00018184,+0.00014574]; the new CUDA runtime is numerically close, not byte-identical. All new trials compare both original V25 and this fresh control.

V170 is a domain hypothesis: spending increases may matter differently when income and liquidity are constrained. It is not already proved by the variable-income subgroup result. V172's gate was motivated by previous OOF errors; report post-selection optimism and do not claim it is independent confirmation.

## Primary research considered

- Lai et al., Credit Portfolios, Credibility Theory, and Dynamic Empirical Bayes (2012): https://onlinelibrary.wiley.com/doi/full/10.5402/2012/832175 . Connects credit-risk estimation to credibility and longitudinal uncertainty. Motivation for tracking exposure/reliability, not evidence that raw coverage counts or this particular gate improve Kumo.
- Hollmann et al., Accurate predictions on small data with a tabular foundation model, Nature (2025): https://www.nature.com/articles/s41586-024-08328-6 . Uses inference ensembles with differing transformations. Supports testing a frozen inference pipeline, but context bagging here is our experimental adaptation, not a reproduction of the paper or evidence of a Kumo gain.
- Thomas et al., Retrieval & Fine-Tuning for In-Context Tabular Models, NeurIPS (2024): https://proceedings.nips.cc/paper_files/paper/2024/hash/c40daf14d7a6469e65116507c21faeb7-Abstract-Conference.html . Studies local retrieval plus fine-tuning for TabPFN. Full LoCalPFN is not queued because it changes backbone weights. Frozen-only local retrieval may be investigated later; the paper's improvements cannot be assigned to our frozen variant.
- The Tabular Foundation Model TabPFN Outperforms Specialized Time Series Forecasting Models Based on Simple Features (2025): https://arxiv.org/abs/2501.02945 . Supports investigating time-series feature representations for a tabular foundation model. This is forecasting research, not direct credit-default/KS evidence.

No published result is treated as numerical evidence for LOGIKA. Future local/global context experts, Bayesian uncertainty interactions or additional statistical models require the same matched-fold tests and peer-experiment review. For now, run only the five named candidates, record failures and preserve V25.
