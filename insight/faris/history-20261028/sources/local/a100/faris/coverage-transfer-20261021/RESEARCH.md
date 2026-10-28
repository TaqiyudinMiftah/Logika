# V25 public evidence and the allocation-opportunity hypothesis

The user reports public scores .56822 for V25, .56464 for V40, .56088 for
basket/domain, and .56082 for select64. These are user reports, not a fresh Kaggle
API read. The supplied V25 CSV SHA256 is
e7f0748c6c8ded776f0a6a03bdebcd0787cf6c0057d0b078c31196fb34f58efe,
identical to both imported master records. It contains 30,000 unique application
IDs and valid probabilities. The previous recorded public reference was .56470.
Original submissions and the root submission.csv are preserved.

V25 keeps Kumo Large90, its 50k context, original spline77, .003 head penalty,
.5 correction strength and 75:25 blend. It adds six coverage columns to the
original 12-feature correction: negative/nonnegative cash-flow due-month counts
with an observed allocation timestamp; recent account months; completed orders
in 90 days; ecommerce observed days; allocated installment count.

V25's saved OOF KS is .5302846600 versus .5292994659, improving all five folds.
Its AUC is .8470292789, slightly below .8472026884. V40 has higher pooled KS
(.5305700974), lower AUC (.8469707750), and lower reported public scoring than
V25. The public result is a counterexample to treating any AUC decline as an
absolute veto. KS remains primary; AUC, calibration, fold consistency and
uncertainty are complementary evidence. This does not prove that small AUC
declines are harmless generally or that leaderboard feedback is untouched data.

The readout-adaptation proposal is deferred before any new model was trained.
This study follows the new stronger public evidence. Raw source code reveals
that V25's monthly counts exclude installments without a recorded allocation.
That can mix history availability with financial opportunity and repayment
observation. One hypothesis is that separating fully unallocated and mixed
months under negative/nonnegative complete-month cash flow adds information.
Allocation timestamp presence is not full payment or settlement. No DPD field
is used in these four new features. A count of zero retains its actual meaning:
no observed eligible overlap, not asserted safety.

Before fitting, independently rebuild all six V25 features from the original
raw tables. Match two counts against the extension raw export and four support
columns on every correction/validation partition. Then replay all five retained
V25 heads to their 70k saved OOF predictions. Population estimates and default
labels are not used to build these row-local count features. Learned imputation,
splines, scaling and coefficients fit only each original 6k correction partition.
No evaluation labels or identifier predictors enter fitting. The original
baseline's snapshot provenance qualification remains disclosed.

Freeze 13 recipes: original, support-only, months-only, V25, six leave-one-column
V25 ablations, the primary four-count opportunity extension, and two diagnostic
two-count subsets. All have identical penalty .003 and strength .5: 65 fitted
heads total across the five frozen folds. The primary is the full four-count
extension; secondary candidates are not substitutes for primary success.
No parameter or ensemble weight search and no new Kumo context fits are needed.

Retain every head, preprocessing object, aligned OOF array, raw hashes, lineage,
replay diagnostics and independently checked KS/AUC. Report the primary paired
fold/class-stratified bootstrap interval; secondary contrasts are descriptive.
Historical evaluation labels and public results have already been reused, so
these checks do not establish fresh private generalization. The goal remains
.54 full 70k five-fold OOF KS; this study does not lower or declare that target.
No new competition CSV or upload is authorized by this research stage.

The inherited absolute deadline remains 1792585871.823748, October 21 at
19:31:11.823748 WIB. PC2 runs five ordinary CPU workers, two numerical threads
per worker, in a campaign-only CPU/RAM group. No agents or recurring AI monitor
supervise execution, and no unrelated workload is stopped or changed.
