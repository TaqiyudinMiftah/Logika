# Fixed small contribution from a full-training spline expert

Peer review: dedbbd9d41d0585ba858ac0f0edd895cc3f8f518. ALL66 development0.5340125 was selected on reused OOF. The archived cross-fitted routing check was KS0.530402/AUC0.848004 versus its maturity baselineKS0.530636. Its exact CSV now has verified publicKS0.55983, below V25. The generalization failure is observed; attribution to any individual component is not identified. Frozen Kumo in-context fitting has no new backbone fine-tuning epochs here.

V254: standalone logistic spline on the twelve original latent/capacity correction inputs. V255: same model on all18 V25 inputs, adding existing evidence counts. V256: same18 inputs plus predefined account/repayment-coverage group indicators and interactions with price, bureau, late30 and installment burden. The four groups are 2I(account_months>0)+I(allocation_count>0); group0 is reference. These are observed coverage categories, not labels of borrower intent.

Each new expert uses context+correction=56k outer-training rows, never the14k held-out labels. Fit signedlog, median imputation, quadratic3-knot quantile spline for continuous inputs, standardized linear/discrete terms, missing flags and one-hot employment. The loss is meanBCE+.01/2||beta||^2, with an unpenalized intercept; sklearn C=1/(n*.01). No outer early stopping, feature selection, weights or routing search.

Final probability is exactly .95*p_V25+.05*p_newGAM. The existing75:25 mixture insideV25 is preserved, giving final component weights .7125 correctedKumo, .2375originalSpline, .05newSpline. These weights are prespecified for the whole batch. Increasing training rows for the separate expert does not change frozen Kumo context or its weights.

The public result motivates restricting model changes, not selecting a new weight from public scores. The new expert may still hurt; compare matched OOF KS,AUC,paired intervals,fold consistency and difficult segments. Preserve every submission and negative outcome. Raw18-input cache contains no target; six evidence counts must agree across all15 cached fold roles before use. Legacy DPD snapshots and global-reference uncertainty are inherited.

This is not a reproduction of the peer's200-feature spline or selectedALL66 routing. It is a smaller independent component test using already-presentV25 information, with fixed5% influence. Training/Kumo roles and feature identity are recorded; heads and checkpoints stay private, aggregate results/source go to the shared adit namespace.
