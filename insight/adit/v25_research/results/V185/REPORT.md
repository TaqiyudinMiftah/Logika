# V185: Bounded additive spectral and local-context logit residuals over V25

z0=logit(V25); d=clip((logit(V167)-z0)+(logit(V178)-z0),-.25,.25); p=sigmoid(z0+d); average5fold test

KS=0.5311516623; AUC=0.8471087754; deltaKS=+0.0008670022; paired95% CI=[-0.0013898093191365229, 0.00325238800669619].

Source-residual correlation=0.008731. No source independence is assumed. All5folds and test IDs match original artifacts. Computed on local CPU from authenticated mirrored artifacts, with local package versions recorded; baseline metrics reproduced before evaluation. No Kumo updates or hidden-test labels. Public score unknown; retain V25 incumbent.
