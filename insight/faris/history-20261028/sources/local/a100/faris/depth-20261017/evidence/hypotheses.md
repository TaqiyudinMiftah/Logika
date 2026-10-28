# Hypotheses
H1 (engineering): the short 250/300-tree probes may underfit useful views. Compare 450-2400 trees, depths 3-8, leaves 15-63 across promising views. Keep full three-fold evaluation and retain each fold pipeline.
H2: compare feature views against the all-source v3 control at identical model parameters. Explicit first anchors fix seed, training budget, weighting and regularization. Effective removal of reducer, splines, interactions or addition of missing indicators tests local dependence; no-op combinations excluded.
H3: class weighting and regularization may explain differences attributed to features. Explore weight 1/3/6 and bounded C/L2. Probability correction retained. Do not select KS thresholds for competition inference.
H4 (carried forward): existing campaign's temporal, repeat-seed and nested ensemble checks continue unchanged; avoid spending the short new window reproducing thousands of cheap fixed blends.
All reported improvements are adaptive development findings. No independent confirmation is available for this extension.
