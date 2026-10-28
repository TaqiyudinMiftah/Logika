# One pipeline intervention: probability and ranking correction

The 22,016-record archive and the completed horizontal study are reused as development evidence. In the matched five-fold study, the teammate Kumo/offset/spline anchor scored KS 0.5292994659 and ROC-AUC 0.8472026884. Every tested cross-architecture mixture lost against it. It is the incumbent, not the weaker local champion.

One falsifiable limitation is the alignment between a correction fitted only for log loss and the competition's ordering metrics. Keep all 90 Kumo features, 12 correction features, 77 spline inputs, original 50k/6k/14k context/correction/validation partitions, 0.5 correction shrinkage, and 75/25 Kumo/spline weights. Replace only the correction fitting objective with 75% binary cross-entropy plus 25% sampled positive-negative logistic ranking loss, retaining L2 0.003. This does not add a broad feature grid or adjust blend weights after validation.

Use 262,144 training pairs drawn with replacement from the correction partition. Two fixed pair-sampling seeds assess numerical/sampling sensitivity, not independent data. Train every outer fold before assessment. The identical encoder is refitted on correction-training covariates; base Kumo never trained on those labels. Existing spline training remains on outer-training rows. Refitting the original correction must reproduce saved predictions within 2e-6 before admitting the intervention.

Evidence motivating the loss:
- Burges et al., Learning to Rank using Gradient Descent (2005), original Microsoft Research abstract, retrieved 2026-10-20: https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/ . Probabilistic pairwise losses motivate a differentiable ranking objective; this implementation is a penalized linear correction, not the paper's neural network.
- Gao and Zhou, On the Consistency of AUC Pairwise Optimization, IJCAI 2015 / arXiv v4 (2014), original abstract retrieved 2026-10-20: https://arxiv.org/abs/1208.0645v4 . Logistic pairwise surrogate has asymptotic AUC consistency under the paper's assumptions. This is not a finite-sample improvement guarantee for this dataset or the mixed objective.
- Prior Consensus access failed authentication; retained primary-source fallback is used. No competition data was sent to research services.

Rejection is explicit: pooled KS and AUC must improve for each fixed seed, KS in at least four original folds and AUC in at least three, no fold KS loss >0.003, no pooled log-loss increase >0.0005, and the simultaneous paired-bootstrap lower bounds for both metrics must exceed zero. A two-metric Bonferroni adjustment applies; 2,000 stratified resamples use shared draws and average seed differences within each draw. Otherwise retain the incumbent's fitted models.

Labels and feature choices have extensive historical exposure. The inherited repayment paid/DPD snapshots remain uncertified for point-in-time availability. This is a bounded conditional development intervention, not permission for broad search or a claim of independent validation. The original pipeline and all artifacts are preserved.

The implementation ceiling for this invocation is two elapsed compute hours, with five CPU workers and an 8 GiB resource ceiling. This is a conservative new job limit, not a renewal of an expired campaign. The program terminates when the finite fits, assessment and packaging finish.
