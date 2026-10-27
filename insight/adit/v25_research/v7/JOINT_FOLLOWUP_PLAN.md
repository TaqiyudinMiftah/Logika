# V191-V195: validation/public disagreement follow-ups

V185 pooled OOF KS0.5311516623, AUC0.8471087754, deltaKS+0.0008670022 and deltaAUC+0.0000794965 vsV25. It wins3/5folds and paired95% deltaKS CI[-0.00138981,+0.00325239]. Public score0.56699 is independently verified with Kaggle submission57704310 and exactCSV SHA25696094151d4314f3152e7d637a419ce2f41132c67884198fc0305eae73e8b5645. V25 remains public incumbent0.56822.

This disagreement is a warning against promoting the highest pooled KS alone. It does not establish a unique cause: development selection, population change and sampling variability remain possibilities. No individual hidden-test labels are accessed.

Important implementation distinction: V167 uses all11 spectral features, including two dominant-cycle argmax fields. The previous V171/V179/V180/V188 compact set uses9. Joint experiments explicitly preserve V167's11 for a matched starting point.

## V191: Joint local-context Kumo and29-input spectral correction

V185 combines final scores after independent heads; directly test whether spectral and context signals survive joint training. V185 public .56699 below V25 .56822 warns that an OOF-only gain is insufficient. Expect higher capacity risk; compare29-input head gap to V25.

Formula: `Use exact cached V178 raw Kumo=.85*global+.15*local; refit29-input V167 correction with L2.003 and inference shrink.5; final=.75*sigmoid(logit(pK178)+.5*X29*beta)+.25*original_spline77`

## V192: Joint context/spectral correction with selective group shrinkage

V167 head train-validation KS gap .016101 vsV25 .011108. Test joint information with stronger shrinkage specifically on all11 spectral inputs, including unstable dominant-cycle argmax; keeps input set matched to V191.

Formula: `Same raw V178 Kumo and29 inputs asV191; meanBCE+.003/2*sum(base_beta^2)+.01/2*sum(spectral_beta^2), intercept unpenalized; inference shrink.5; original75:25 spline fusion`

## V193: V185 plus bounded inner-OOF partial-pooling calibration residual

V189 low-dimensional inner-OOF employment/history calibration yielded KS .530638, deltaAUC -0.0000041 vsV25; no generalization claim. Test bounded partial-pooling effect onV185; reject leaderboard recommendation if AUC or stability deteriorates. No fitted score weights.

Formula: `z=logit(pV185); d=.5*clip(logit(pV189)-logit(pV25),-.1,.1); p=sigmoid(z+d); per-fold test fusion then mean5fold`

## V194: Inner-crossfitted spectral/local evidence combination

V185 fixed logit fusion increases pooledKS but only3/5folds and mixed public outcome. Learn complementarity using strictly inner-OOF predictions, never using in-sample original spline or outer labels for mixing weights. Difference from V185 includes fusion before original spline and training-only weight fit.

Formula: `For each outer fold,3 inner splits of6k train base18/global, spectral29/global, local18/V178 heads on4k and predict held2k. Fit w_s,w_l in[0,1.5] by BCE(z_base+w_s*(z_s-z_base)+w_l*(z_l-z_base))+.02/2*((w_s-1)^2+(w_l-1)^2). Apply weights to full6k corrected-Kumo heads; final=.75*sigmoid(eta)+.25*original_spline77`

## V195: Coverage safeguard for V185 spectral correction

V185 no-allocation segment deltaKS -.003403 vsV25, while1:3 observations +.006123 and over3 +.000197; V178 no-allocation deltaKS -.000260. Fixed availability gate tests a mechanism suggested by prior outer errors; it is post-selection and not independent confirmation. Distinct from V172 fallback toV25 on a different spectral-context backbone.

Formula: `p=pV178 when aux_allocation_count<=0 or missing; otherwise pV185; per-fold test fusion then mean5fold`

## Evaluation requirements

Every candidate receives pooled KS/AUC, per-fold metrics, paired300-resample deltaKS/deltaAUC confidence intervals against bothV25 andV185, head training/held-out gap, held-out error analysis against both controls, exact checkpoint and submissionSHA. Report negative results. The bootstrap excludes repeated-search/retraining uncertainty. Rules such as4/5wins and nonnegative deltaAUC are operational heuristics, not statistical significance thresholds.

No automatic public promotion for passing0.531. Public incumbent staysV25. Stronger training separation alone is not evidence of useful representation. Fixed-score recipes report the inherited head diagnostic, not a newly fitted head gap. All additional fitting uses the6k correction partition, with50k Kumo context and14k outer validation unchanged. Kumo weights100% frozen.

Runtime: node480 CPU environment; no new backbone/context inference. One real worker, twoCPUthreads, wait for0.9GiB effective freeRAM using bounded inactive-file allowance, maxprocessRSS2GiB. No reservation buffers. Before every candidate, fetch sharedHEAD and review new peer work; sync audit/results before continuing.

## Completed results and decision

| Run | PooledKS | AUC | DeltaKS vsV185 | DeltaAUC vsV185 | KS fold wins vsV185 |
|---|---:|---:|---:|---:|---:|
| V191 | 0.53118325 | 0.84707551 | +0.00003159 | -0.00003327 | 1/5 |
| V192 | 0.53065097 | 0.84710701 | -0.00050069 | -0.00000177 | 3/5 |
| V193 | 0.53145951 | 0.84710754 | +0.00030785 | -0.00000123 | 4/5 |
| V194 | 0.53093506 | 0.84710888 | -0.00021660 | +0.00000011 | 1/5 |
| V195 | 0.53011746 | 0.84713148 | -0.00103420 | +0.00002271 | 3/5 |

V193 is the numerical development best. Its4/5fold KS wins and near-unchanged AUC are encouraging, but deltaKS95% CI[-0.00038960,+0.00076267] crosses zero and search/retraining uncertainty is excluded. V25 remains the public recommendation.

Fixed-score V185/V193 have no newly trained ensemble head; their inherited V25 head gap must not be interpreted as a new model gap. Relevant source head gaps: V1670.0161010, V1780.0106863, V250.0111077. V193 still inherits spectral-model and search risk. See SUBMISSION_RECOMMENDATION_20261027_AFTER_V195.json for the complete decision.
