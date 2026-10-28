# Peer review and observed ALL66 generalization failure

Reviewed main commits409c80b,f256899,dedbbd9, preserving the Faris namespace and its archive. Read rootAGENTS, FarisREADME, ALL66CONFIG/PROTOCOL/RESULTS/REFIT_POLICY, spline/assembly source, and relevant aggregate-log entries873,889. Archive file counts are not independent experiment counts.

ALL66:45 frozen Kumo component fits,5 spline refits,5 assemblies. The reported developmentKS0.5340125023110701 is selected on reusedOOF. The later cross-fitted routing diagnostic recordedKS0.530402/AUC0.848004 versus its own maturity baselineKS0.530636; it is not an untouched audit of all prior selection. A literal ALL66 reproduction would duplicate peer work and does not establish improvement over our V25.

The KaggleAPI and downloaded CSV now independently verify submission57730355 atpublicKS0.55983, SHA d27250c950145dbc5de5a046d16615090a66e2e3d7c704914d91fcd151987485. Delta againstV25public0.56822 is-0.00839. See ALL66_KAGGLE_READONLY_RECEIPT.json. No automatic submission was performed.

Potential failure mechanisms: repeated choice among many component/routing combinations; group-specific logit shifts toward overall prevalence; correction/test spline asymmetry; larger final context versus originalOOF context. These are plausible contributors, not established causes. The shift source computes .7*(logit(mean group correction prediction)-logit(global correction prevalence)); groups with genuinely different risk can be pulled toward the same overall rate. This should not be described as an epoch effect: these Kumo backbones are frozen.

CurrentV251-V253 residual information controls remain distinct from the peer recipe. NewV254-V256 test a small fixed5% standalone spline contribution trained on56k, keeping95%V25. No copy of OOF-selected routes or shift0.70 is queued. ExistingV186/V189 already cover affine/hierarchical calibration, so those are not repeated.

Selected-loan tenor was also checked against the data dictionary: it is fixed12 months. No fabricated tenor-risk feature or experiment was added.
