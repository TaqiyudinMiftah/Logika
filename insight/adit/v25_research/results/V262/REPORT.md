# V262: credit-aware post-hoc on V25

Fit four credit interactions to inner3fold-OOF corrected-Kumo offset on6000 correction rows: meanBCE(offset+R@gamma)+.03/2||gamma||^2+.2/2 mean((R@gamma)^2). R is support-gated, main-effect-residualized late30*allocation-reliability, installment*conditional-deficit, installment*outflow-pressure, and total-burden*allocation-sparsity. Ridge projection .001, reference fitting on6k only. No intercept. Inference eta=logit(p_V25_corrected)+.25*clip(R@gamma,-.3,.3); final=.75*sigmoid(eta)+.25*original_spline; test mean5folds.; mean5fold test probabilities

Pooled OOF KS 0.5303177217; AUC 0.8470283052.
Delta KS +0.0000330617; delta AUC -0.0000009737; fold wins 2/5.
Paired bootstrap 95% delta-KS CI: [-0.000188366354834682, 0.00025193944317325657].

The extra head is trained against inner-OOF corrected Kumo, not an in-sample spline teacher. The original V25 head and 75:25 fusion remain fixed. Allocation-count reliability is only a coverage proxy, not the denominator of the late30 posterior. Deficit months are eligible repayment/account-matched months, not all account months.

All five outer folds have been used in prior research. Bootstrap intervals are conditional on fitted predictions and do not correct experiment selection or training variability. Inherited DPD snapshot and global-reference provenance remains unverified. This is our operational credit post-hoc experiment, not a verified reproduction of the friend method.

Decision: KEEP_V25_NO_ESTABLISHED_GAIN. V25 remains the public incumbent (user-reported 0.56822). '
Candidate public score is unknown; no Kaggle submission was performed.
