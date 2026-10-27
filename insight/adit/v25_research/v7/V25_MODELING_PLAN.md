# Fixed V25 modeling research toward pooled OOF KS0.531

User objective on2026-10-27: improve modeling on node480, retain V25 inputs and100%-frozen Kumo; analyze held-out errors for every experiment and preserve all outcomes. Current public incumbent V25 remains0.56822; the explicit development target is pooled KS>=0.531, not a promise of hidden-test performance.

Batch1 has12 prespecified configurations varying correction strength, L2 and fixed probability/logit blending. Training labels and preprocessing are restricted to original6k correction rows;50k Kumo context is disjoint, and outer14k validation is unchanged. The original spline is not refit; its in-sample predictions are never used to train another estimator. Blend weights are fixed configurations, not learned using outer labels. Identical feature values and90-input Kumo/18-input correction/77-input original spline are preserved. No learned fold-specific threshold adjustment or hidden labels.

For each configuration save70k OOF,30k test submission,5 correction checkpoints, per-fold KS/AUC, paired300-resample deltaKS/AUC uncertainty against V25, train-validation correction gap, and error analysis. Error reports cover top10/20/32.368571/40/50% capture, improvements and regressions among true bad/good borrowers, coverage/employment/time/risk segments and target direction. The32.37% diagnostic was observed on V25 and is exploratory; no model is fitted to these outer-OOF segment labels.

Existing outer folds have already been reused extensively. Prespecified batch results and confidence intervals are development evidence; bootstrap omits retraining and model selection. Adaptation after error analysis must be logged, and independent seed/time evidence is needed before claiming generalization. Legacy snapshot and global-reference uncertainties remain. V25 stays public incumbent unless supported by new verified evidence.

Execution: v7/scripts/modeling_features.py. Baseline error analysis is saved before candidates. Every result, including negative, enters EXPERIMENT_MASTER_LOG automatically. No Kaggle upload.

References: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html and https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization.


## Batch2: interaction representations after batch1 error analysis

Batch1 best V127 has pooledKS0.5306712194 versus V25 0.5302846600. It improves highest10/20% capture by2 defaults each, but loses7 at V25 diagnostic32.37% cut. It improves variable-income/self-employed segment KS slightly but worsens low-score and0.15:0.4-score separation. These are exploratory held-out associations, not evidence to hard-code a segment gate. Stronger global correction and radical spline-weight changes did not help.

Batch2 V132:V143 retains all18 V25 correction inputs, the same context/folds and fixed75:25 mixture. Six models add12 prespecified domain-pair products inside the correction representation, with separate L2. Six models jointly learn low-rank4/8 factorization interaction embeddings. The original additive spline component remains, so these test added interactions rather than removing V25 behavior. Input normalization/missing/category references fit only6k correction rows; no outer labels enter optimizers or preprocessing. The optimized loss adds L2 separately for the interaction parameters. Analytic factorization gradients are checked against finite differences before launch.

Bilinear FM contribution is0.5*sum_k[(sum_j u_j*v_jk)^2-sum_j u_j^2*v_jk^2]. This is a new internal representation of unchanged inputs, not new historical feature engineering. All candidates receive full held-out error reports and submission CSVs. Batch2 follows observed development results, so its selection is adaptive and uncertainty is not selection-corrected.


## Batch3: more training observations for a separate tree expert

Batch2 interaction/embedding heads did not improve V25; their pooledKS ranged0.525538:0.529264. They used only6k correction-training rows. The next hypothesis is to learn nonlinear structure with more labeled observations, without supplying in-context/in-sample Kumo scores to a trained meta-model.

V144:V151 train two histogram-GBDT configurations on56k outer-training applications using exactly the union of116 input fields already used by the V25 pipeline. The existing Kumo90, spline77 and correction18 receive no new raw information. The two conditional counts are copied from the existing allocation denominators, and the four support fields from audited raw features; values must equal saved V25 outer-validation inputs. Categories are fitted on training data only. Tree count is selected on an inner15% holdout, then the expert is refitted on all56k; outer14k validation is untouched. Each expert is evaluated standalone and as10/20/35% fixed probability blends with V25. All8 results get OOF/errors/checkpoints/submission and remain logged. No outer labels train the blend.

The user expanded scope beyond modeling if needed, while retaining V25 as the benchmark. Shadow loan additional labels require a fresh provenance/censoring audit; aged loans are not automatically12-month target labels. Joint latent complexity is deferred given V111 KS0.526691 and the current embedding negatives. Query-adaptive frozen Kumo is a later candidate; evidence from LoCalPFN concerns TabPFN retrieval/fine-tuning, not a demonstrated guarantee for frozen Kumo. Ranking objectives remain a candidate if the current expert fails.

References: https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html and https://proceedings.neurips.cc/paper_files/paper/2024/file/c40daf14d7a6469e65116507c21faeb7-Paper-Conference.pdf.


## Batch4: risk-ranking objectives with a V25 anchor

The full-training HGB116 experts alone achieve pooledKS0.521543/0.521150, below V25. Small fixed blends improve some folds, but the best pooledKS is0.530528, below current V127. Their flexibility/data volume did not yield a stronger standalone scorer. Large blends hurt; preserve V25 as the anchor.

V152:V159 implement the user's ranking-objective reference. A90-coefficient student changes V25 corrected-Kumo logits with the same spline representation. Fit BCE plus sampled pairwise logistic loss (32 training negatives per training positive) or a smooth class-CDF separation objective, plus prediction-distance tether to V25 and ridge. CDF thresholds are quantiles50/65/80% of6k training teacher logits, softness0.75; no outer thresholds fit the loss. Loss gradients receive finite-difference directional checks on fold0. All labels for optimizer/pair selection/thresholds are inside6k correction rows; outer labels remain evaluation-only. Fixed original spline25% and five-fold inference remain. Both positive and negative outcomes get error analysis and submission. These are adaptive development experiments, not an independently validated claim of generalization.


## V160: one local regularization follow-up

After40 model variants, V127 remains best pooledKS0.530671. Since L2.001 improved over.003 without a large optimism-gap increase, test one lower L2.0003 while holding18 features, strength0.5 and75:25 blend fixed. This is adaptively chosen development research; no hidden-label evidence or guaranteed KS is claimed. Shared GitHub main was reviewed at49a1b7872b884aa8562c3f974f15d18ec5b8a297. Execute one experiment only, push/review again before the next; enforce RAM headroom at launch.


## V161: best small-head adjustment plus cached tree complement

V160 L2.0003 yieldsKS0.5304969684 and loses to V127 L2.001. Retain the V127 correction and test20% HGB-a, since V146 showed positive pooledKS/AUC at that small weight but35% harmed separation. This is a new combination, not re-running V146 (which used L2.003). Preflight shared main4085af1996334f5e09944123692aec9b9283c58c shows no peer update. Cached expert predictions avoid repeated training and reduce RAM. No target-based gate or fold calibration is fitted.


## V162: normalized financial-stress exposure

V161 low-L2 plus20% tree reachesKS0.530540, below V127. With scope expanded by user, add just one deterministic representation: n_negative/(n_negative+n_nonnegative), NA for zero exposure. Existing additive splines on two counts cannot express this ratio exactly. No additional raw source is read; cutoff and denominators are exactly V25. Keep L2.003, strength.5 and75:25, now19 input fields. This is not a re-run of V14 (conditional late-allocation posterior). Shared main843b807e9d4e081ab7d871c96961753372706298 inspected; no peer update.
