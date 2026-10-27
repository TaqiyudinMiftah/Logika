# Five additional frozen-Kumo experiments: reviewed before scheduling

These five follow V173, after its result is saved and pushed. V169 through V173 remain unchanged. There are no Kumo gradient updates. No candidate is claimed to improve KS before matched5fold evaluation. Keep V25 public incumbent and preserve original90 Kumo inputs in this additional batch.

## Evidence used

- V25 pooledKS0.5302846600, AUC0.8470292789. V167 pooledKS0.5307954555, but delta95% CI crosses zero and head gap rises to0.01610 vs0.01111. More correction parameters are not a convincing default.
- V169 pooledKS0.5288125269, delta -0.0014721331; AUC0.8472206798, delta +0.0001914009, paired AUC CI[+0.000030064,+0.000350106]. Top20% captures5860 defaults vs5851 for V25, but the diagnostic32.37% slice captures16 fewer. Different objectives/rank regions disagree; do not promote it as a better single model.
- The additional data sanity review used50k fold0 context for all distribution references and14k outer-held-out applications for descriptive univariate checks. These are exploratory associations, not unique-feature/model gains or an independent discovery holdout.

## Final queue

| ID | Exact change | Empirical/domain support | Primary uncertainty / reject condition |
|---|---|---|---|
| V174 | Original90 plus allocation Beta-Binomial mean, variance, credibility and posterior-times-installment burden; context-only empirical prior | Posterior allocation mean univariate KS0.31685, variance0.14240; burden interaction0.37252 on9806 observed validation rows. Exposure uncertainty is explicit | Mean correlates0.9575 with existing repayment propensity; prior conditional-Bayes V14 was negative. Reject if uncertainty adds no matched gain or harms thin files |
| V175 | Original90 plus four existing latent uncertainty/gap fields: price SD, bureau SD, declared-income gap and outflow/income ratio | These already belong to successful baseline correction, but Kumo has not received them. Price/bureau mean copies were removed as redundant | Univariate uncertainty/gap signals are weak. This is an interaction hypothesis, not proof of new information. Inherited latent/global-reference provenance remains uncertain |
| V176 | Original90 plus item-value entropy30d/90d, change30-vs-prior60 and normalized entropy30d | Fold0 univariate KS0.17917/0.14924/0.10468/0.12394; value composition is distinct from simple total spending and ordinary category count | Spending/frequency confound; earlier correction-only V166 was negative. No unsupported order-status denominator. Reject without gain beyond original spending/frequency inputs |
| V177 | Fixed final probability average0.90*V25+0.10*V169, samefolds, existing frozen checkpoints | V169 improves AUC with paired interval above zero and captures9 more defaults at top20%, despite worse peak KS | Selection uses prior development results. No weight search, no monotonic-calibration KS claim. Cheap cache-only trial; reject if KS/capture tradeoff is unfavorable |
| V178 | Raw Kumo p=.85*global90-input V168+.15*local frozen expert; local context split by observed allocation history, max24k per group, then same18-input correction/spline | Existing error analysis repeatedly flags zero-allocation group. Different information coverage may warrant different comparison populations | Coverage is not automatically an optimal financial neighborhood. No user/entity IDs as features; no outer labels for retrieval. Keep global anchor and reject unstable subgroup shifts |

## Rejected or reduced before spending training compute

1. Provider-normalized consensus: Spearman0.9999983 with existing bureau_score_mean. Provider context means668.28–668.61 and SD53.31–53.43 give little support for a large scale-bias correction. Replaced by V177; this was not a failed training run.
2. All9 latent inputs: price posterior mean Spearman0.9989916 with existing price and bureau posterior mean0.9997274 with bureau history score. V175 retains only four uncertainty/gap inputs.
3. Items/completed-order ratio: inherited order-count status semantics were not re-established, so this proposed derivative was removed. Category-value entropy uses audited completed item_total exclusively.
4. Blindly carrying V169 coverage6 into every new Kumo input set: rejected after its pooledKS decline. Additional runs retain original90 before their specific changes.

Draft definitions and the original sanity review are retained. A high univariate KS does not establish incremental information, and a low correlation does not establish predictive value. All results, including negative ones, go to the master ledger.

## Statistical and research basis

Credibility and dynamic empirical Bayes connect longitudinal evidence strength to credit-risk uncertainty: https://onlinelibrary.wiley.com/doi/full/10.5402/2012/832175 . V174 uses historical allocation timing, not a default label; the paper does not establish a LOGIKA gain.

Tabular foundation-model inference ensembles and alternative feature representations are investigated in https://www.nature.com/articles/s41586-024-08328-6 . V175/V176/V177 are application hypotheses, not reproductions of its benchmark.

Local retrieval for tabular in-context models is studied in https://proceedings.nips.cc/paper_files/paper/2024/hash/c40daf14d7a6469e65116507c21faeb7-Abstract-Conference.html . The full LoCalPFN method includes fine-tuning; V178 deliberately tests only a frozen, coarse coverage-based adaptation. No reported LoCalPFN gains are transferred to Kumo. The weaker neighborhood choice and loss of local class support are explicit risks.

## Execution and evaluation

Wait for the original queue to finish and push V173. Before each additional run, fetch main, review peer changes and prevent redundant runs; unknown peer updates stop automatic advancement for inspection. Keep50k eligible context pool, disjoint6k correction and14k outer validation, identical outer folds. Priors/reference normalization use only context rows. Last-allocation timing must not be called DPD, cure or complete settlement. Existing baseline snapshot uncertainty remains.

Every result records pooled/fold KS and AUC, paired bootstrap CI, error segments, train/held-out diagnostics, source/checkpoint references and an independently hashed submission CSV. Push each completed result before the next. Intervals are conditional on fitted predictions and do not correct repeated development search. The0.531 development goal alone cannot establish leaderboard improvement.
