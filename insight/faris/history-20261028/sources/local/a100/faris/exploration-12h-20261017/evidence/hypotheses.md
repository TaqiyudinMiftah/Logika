# Evidence and hypotheses, 2026-10-17
Consensus Primary search attempted; returned UNAUTHORIZED, oauth_refresh_token_rejected, reauthentication required. No paper findings claimed from Consensus; primary-source fallback used.

H1 temporal transitions and history reliability: engineering hypothesis from inspected raw schemas, matched to v3.
H2 true-gain vs split-count feature selection: verified existing Preprocessor LGBMClassifier uses default importance_type=split; matched experiment, not a renamed claim.
H3 official TabM: https://raw.githubusercontent.com/yandex-research/tabm/main/README.md read usage/training: member-wise loss, probability averaging, numerical embeddings. All preprocessing and embedding bins fitted on training partitions only.
H4 retrieval: https://raw.githubusercontent.com/yandex-research/tabular-dl-tabr/main/README.md read implementation context. Local retrieval_mlp is a TabR-inspired engineering baseline, NOT a faithful reproduction; official TabR family separately marked unavailable until full port is tested. No self-neighbor or evaluation-target access.
H5 TabICLv2: https://raw.githubusercontent.com/soda-inria/tabicl/main/README.md read API, checkpoint and cache settings. https://arxiv.org/html/2602.11139v1 read abstract/method overview. GPU inference with checkpoint tabicl-classifier-v2-20260212.ckpt; offline staged weights, fixed context. Benchmark results are external, not predictions for Logika.
H6 Kumo: existing cached official nvidia/Kumo-Tabular@v1.0.1 and SDK revision329a707bbcf9ca147aa45a63fef8b2029f4ee37b. Context inference separated from training-only episodic AdamW.
H7 nested ensembles and stability: local historical pass2 showed three-seed KS 0.5243529 versus champion0.5257714. Do not assume bagging wins. Learned weights require inner OOF on outer-training rows.
H8 residual MLP, FT-style tokenizer/transformer, GRU and DeepSets are local engineering implementations, not claims of exact paper reproduction.
No external borrower data, paid APIs or leaderboard probing. Retrieval timestamp:2026-10-17. Source content is version-pinned during deployment.

H9: Add explicit actuarial Bernoulli logit/cloglog GLMs, additive spline/EBM controls, WOE scorecards, and beta-binomial credibility scorecards. Predict default frequency, test small champion additions and calibration separately. Compare shrinkage and bin counts by full outer OOF, never reserved labels.
