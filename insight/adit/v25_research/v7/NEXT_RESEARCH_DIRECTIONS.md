# Next mechanisms to implement during the rolling research goal

DRAFT hypotheses only: not executable queue entries, not counted toward the3-job reserve. Build and preflight code before marking readiness. User explicitly requested continuous research until they stop; do not mark the goal complete just because a numerical score rises.

1. Frozen Large population context by employment. Adapt extra_frozen_context.local_predictions, using3 actual employment categories rather than binary allocation coverage. Each expert context stays within the original50k context pool; cap24k,8estimators, fixed15%local+85%cachedglobalV168 before6k correction and25%spline77. Query groups use only attributes at decision. Fallbackglobal for missing/unsupported category. This tests different cashflow/reliability mechanisms under salaried/self-employed/variable income. No outer target may choose the groups or weight.

2. Frozen Large population context by underwriting APR regime. Compute the median annual_interest_rate only from each fold context pool, group low/high, cap24k and same15%anchor asV178. APR is a decision-time policy-generated variable; audit source dependence/drift and do not equate it with independent risk measurement. Keep target proportions natural inside groups; no query-label matching.

3. Average coverage-local and employment-local rawKumo predictions50:50, then refit the original18-input correction on identical6k. This gives85%global +7.5%each local before nonlinear correction. Compare source-specific contexts and combined representation. No feature or score weight fitting on outer labels. Requires completed source predictions.

Implementation references: kumo_gpu_experiment_v3.py, extra_frozen_context.py lines76+, cached globalV168 rawparquets, Large pinned weights and per-instance GPU-cache parity check. Stream checksum to reduce850MB temporary host allocation. Use existing malloc_trim between local experts. Do not modify installedSDM or peer notebooks. Respect3GiB ownRSS ceiling and available shared16GiB cgroup. GPU80GB is free when no job running; Large contexts previously fit but host availability must be checked.

Read the latest per-experiment CONTINUOUS_INSIGHT.json, risk reviews, paired KS/AUC vsV25/V185/V193 and shared GitHubHEAD first. Current lower-capacity Small controls did not establish improvement; repeated development selection remains. Preserve all failed-attempt receipts and negative results.
