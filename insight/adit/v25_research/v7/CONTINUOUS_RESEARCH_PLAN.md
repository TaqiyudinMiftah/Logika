# Continuous statistical/stability queue: V196 onward

Maintain at least3 executable pending experiments in addition to a running job. A controller reloads this plan at every boundary, analyzes each completed run, audits/pushes it, and starts the next resource-safe job. It launches a job only if at least3 vetted pending jobs remain afterwards. Refill from new evidence before exhausting the reserve. This guard preserves a concrete reserve; it is not permission to run random parameter sweeps.

All candidates keep Kumo100% frozen and original outer folds. V25 public incumbent0.56822; V185 public0.56699 exact hash verified. V193 is development best only: KS0.5314595146, AUC0.8471075443. Each candidate has paired KS/AUC uncertainty againstV25,V185,V193, source-complexity caveats, errors, code/checkpoint and own submission. Search uncertainty remains.

## V196: Stable9 spectral correction on local-context Kumo

V191 all11 spectral inputs raised head gap to.017657 and decreased AUC vsV185; phase argmax can be unstable under sparse orders. Test removal while preserving9 band-power/support features and identical context scores.

Formula: `Drop only2 dominant-cycle argmax inputs fromV191; fit27-input head with L2.003, shrink.5 on cached rawV178 Kumo; original75:25 spline fusion`

## V197: Stable9 spectral correction with original global Kumo

Separate spectral selection from the local-context interaction. V167 remains stronger than many expanded inputs, but its11-feature head gap.016101 and V185 public decline suggest instability. This is correction-level removal; V171 previously changed frozen Kumo context inputs.

Formula: `Drop only2 dominant-cycle argmax inputs fromV167; fit27-input head on original global Kumo with L2.003, shrink.5; original75:25 spline fusion`

## V198: Reduced-knot stable9 local-context correction

V192 stronger L2 on11 spectral features reduced KS and left head gap.016393. Test reducing basis complexity rather than shrinking all coefficients; removes one spline coefficient per numeric input.

Formula: `Same27 inputs/rawV178 asV196; quadratic spline2knots instead3, training-only knots/scales; L2.003 and shrink.5; original75:25 spline fusion`

## V199: Gaussian-rank preprocessing for V25 correction

Extreme historical financial measurements affect boundary knots and basis scaling. Train-reference ranks test robustness to outlier magnitudes without fitting on test distributions. Same18 inputs and frozen context.

Formula: `On6k correction only: median impute, QuantileTransformer100quantiles normal output, quadratic3-knot spline, scale; missing indicators+employment; original offset BCE+L2.003 and shrink.5; original75:25 spline fusion`

## V200: P-spline-inspired coefficient roughness correction

Global L2 does not distinguish smooth financial relationships from oscillating residual corrections. Add coefficient-difference regularization while keeping the18 inputs and90-coefficient design. Quantile knots differ from canonical equally spaced P-splines; do not claim an exact continuous curvature penalty.

Formula: `V25 head objective meanBCE+.003/2||beta_nonintercept||^2+.03/2||D2*(beta_spline/scale)||^2; difference penalty within each numeric spline block; shrink.5; original75:25 spline fusion`

## V201: Stabilized Jeffreys-information correction

Sparse combinations of employment/missingness and repayment evidence can bias correction coefficients. Test a stabilized Firth-inspired information penalty; addedridge andL2 mean this is not exact classical Firth logistic regression. No transfer of published performance claims.

Formula: `Minimize meanBCE+.003/2||beta_nonintercept||^2-.5/n*logdet(X^T*W(beta)*X/n+.0001I); original V25 design and shrink.5; original75:25 spline fusion`

## V202: Laplace predictive uncertainty for original V25 head

V189 partial pooling gives a small gain without more input dimensions. Extend reliability to the head posterior: high-leverage applicants receive more predictive integration. Approximate Laplace uncertainty conditions on encoder/prior, excludes Kumo and model-selection uncertainty.

Formula: `Fit original V25 beta; Sigma=(X^T*W*X+n*.003*diag(0,1,...)+1e-6I)^-1; eta=zK+.5Xbeta; v=.25*diag(X*Sigma*X^T); pK~=sigmoid(eta/sqrt(1+pi*v/8)); original75:25 spline fusion`

## V203: Laplace uncertainty for stable9 local-context correction

Test whether uncertainty is especially useful for the higher-dimensional spectral head rather than the compact original18. Same formula and conditional caveats asV202; compareV196 for the specific effect.

Formula: `UseV19627-input local-context head; calculate conditional Laplace coefficient covariance and logistic-normal predictive approximation asV202; original75:25 spline fusion`

## Primary methodological references

- Eilers andMarx1996: https://sites.stat.washington.edu/courses/stat527/s13/readings/EilersMarx_StatSci_1996.pdf ; coefficient-difference smoothing inspiresV200, with quantile-knot caveat.
- Firth1993 and author explanation: https://davidfirth.github.io/blog/2023/01/05/f93-citations-and-history/ ; stabilized information penaltyV201 differs from exact Firth.
- MacKay, Information Theory, Inference and Learning Algorithms, section43.5: https://www.inference.org.uk/itprnn/book.html ; approximate Gaussian integration inspiresV202/V203. No guarantee of KS improvement.

Resource guard uses real work only; twoCPUthreads, at least0.9GiB effective RAM and maxRSS2GiB. Current jobs reuse frozen Kumo caches, requiring no newGPUcontext. The shared16GiB cgroup is monitored, and no buffers reserve unusedRAM. Review sharedHEAD before every fit and after result preparation; never force-push or overwrite peer work.

## Refill after V196 error review

V196 KS0.53046043, AUC0.84712534; deltaKS vsV193-0.00099909 and deltaAUC+0.00001780. No gain established. Preserve the three-job reserve with additional executable hypotheses:

### V204: Gaussian-rank stable9 local-context correction

V196 removes unstable cycle argmax and improves AUC vsV193 +0.0000178 but loses KS and retains a high head gap. Test bounded train-reference ranks of the stable financial/power inputs, rather than expanding the representation.

Formula: `V19627 inputs andrawV178; training-only100-quantile Gaussian ranks before3-knot correction splines; L2.003, shrink.5; original75:25 spline fusion`

### V205: Reduced-knot original18-input V25 correction

Isolate model-capacity reduction on the public-incumbent pipeline; V198 separately tests reduced knots with27 local/spectral inputs. Avoid confusing different feature/context effects with coefficient count.

Formula: `Same18 original V25 correction inputs andglobal frozenKumo; quadratic spline2knots rather than3; L2.003, shrink.5; original75:25 spline fusion`

### V206: Roughness-regularized stable9 local-context correction

V196 retains high train-validation head gap despite removing2 cycle features. Test smoothing within each financial/spectral response, complementary to removing features or increasing global L2; same27 inputs permit a matched comparison.

Formula: `V19627-input rawV178 head; meanBCE+.003/2||beta_nonintercept||^2+.03/2||D2*(beta_spline/scale)||^2; shrink.5; original75:25 spline fusion`


## Refill after V200 convergence and error review

The same P-spline objective converged with analytic-Hessian Newton after L-BFGS hit its500-iteration cap; partial files/source and failure are archived. V200 head gap0.00748166, AUC0.84731949, KS0.52956578.

### V213: V193 anchor plus10% low-gap P-spline expert

V200 AUC.84731949 exceedsV193 by.00021194 and its head gap.007482 is lower thanV25.011108, while KS.52956578 falls. Test a bounded contribution from a lower-variance correction, without selecting weights on outer labels; V200 deltaAUC CI vsV193 still crosses zero.

Formula: `Perfold p=.90*pV193+.10*pV200; fixed weights before own evaluation; mean5foldtest`

### V214: Smooth capacity effects while retaining sharp credit-risk responses

V200 smoothing all17 numeric responses improves AUC and reduces head gap but loses KS. It may over-smooth threshold-like delinquency/bureau effects. Apply the same penalty only to five income/cashflow/burden measurements; keep credit/history-response bases at original L2.

Formula: `V25 original18/globalKumo; BCE+.003/2||beta_nonintercept||^2+.03/2||D2*(beta_spline/scale)||^2 only on capacity inputs; other bases unchanged; shrink.5 and original75:25spline77fusion`

### V215: Smooth evidence reliability while retaining risk-level thresholds

Coverage counts and posterior SDs can create unstable residual bends, while burden/delinquency levels may need thresholds. Smooth only8 measurement-reliability/coverage effects; preserve financial-risk response flexibility. This tests a different mechanism from smoothing monetary capacity.

Formula: `V25 original18/globalKumo; BCE+.003/2||beta_nonintercept||^2+.03/2||D2*(beta_spline/scale)||^2 only on evidence inputs; other bases unchanged; shrink.5 and original75:25spline77fusion`

