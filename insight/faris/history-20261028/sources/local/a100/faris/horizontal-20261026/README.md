# Horizontal Kumo experiment — 26 October 2026

**Completed at 05:08 WIB. No improvement: final KS 0.528137 versus matched baseline 0.530636.** All 1,510 fits passed the completion audit. See [final result and interpretation](RESULT_SUMMARY.md).

The target is original fivefold pooled development KS >= 0.54, followed by assessing whether an eligible candidate merits a Kaggle submission. A local improvement does not guarantee a public or private leaderboard gain.

The catalog contains **21,514 candidate feature definitions: 2,408 existing and 19,106 new formulas**. The 290 input packs each contain at most 100 columns. Standard packs combine 20 common anchors with up to 80 features from one family. The existing 90-feature model and a 20-anchor model are controls. Fold-0 constant and duplicate filtering retains 17,210 feature IDs across the packs and excludes 4,304. Counts measure definitions, not independent information.

Features are generated one pack at a time. The catalog covers debt service, debt-to-income, liquidity relative to income, payments and expenses, net debt, cash-flow shocks, amortization, repayment credibility, temporal changes and volatility. Sources and the difference between established mechanisms and proposed extensions are recorded in SOURCE_LEDGER.json and CATALOG.json.

## Evaluation

Each of five original folds uses 50,000 context rows, 6,000 correction rows and 14,000 outer rows. Kumo Small with two estimators screens every pack. All preprocessing, empirical priors, vocabulary and duplicate detection use context rows. The correction set is split into 3,000 ranking and 3,000 selection rows. Packs are ranked on the first half; ensemble counts and blend weights are selected on the second. A fixed maturity correction and spline follow the mixture. Top packs selected within each fold receive Medium and Large runs with eight estimators, including matching 90-column controls.

This is **reused development validation**: the original outer labels have prior experiment exposure. Timestamp and delinquency-age consistency checks passed for the raw repayment history, but exact upstream snapshot capture semantics remain an assumption. Final paired bootstrap intervals describe conditional development uncertainty and cannot remove adaptive selection bias.

The synthetic SYNTHESIS_GATE confirms that changing outer predictions leaves selection unchanged, that selected mixtures recompute correctly, and that mismatched row order is rejected. It does not establish model performance.

## First results and interpretation

The initial three-pack pilot completed all five folds. Raw Small-two-estimator pooled KS was 0.524236 for the 90-feature control, 0.459866 for the pilot cash-stress pack and 0.460336 for the pilot net-debt pack. These individual semantic packs are weaker predictors on their own. The completed campaign did not establish complementary gain; the final Medium/Large ensemble underperformed the matched baseline. See RESULT_SUMMARY.md for the full result.

## Execution locations (workers completed)

- Node 2003: /mnt/home/faris/logika-horizontal-20261026. Two RTX 3060 workers run logika-horizontal-lane0-20261026 and logika-horizontal-lane1-20261026 as user services. They own catalog pack indices 0 through 149; reserved directories delegate indices 150 through 289 to node2002.
- Node 2001: the same home-relative directory holds the finite coordinator, logika-horizontal-finish-cluster-20261026. It collects screening predictions from both GPU hosts, freezes inner selection, schedules larger checkpoints across all four workers, and writes final reports. Its two GPUs were occupied by unrelated jobs when the move was made.
- Node 2002: /home/faris/logika-horizontal-20261026. Two RTX 3060 workers run logika-horizontal-node2002-gpu0-20261026 and logika-horizontal-node2002-gpu1-20261026. They own catalog pack indices 150 through 289. The exact Python, numerical packages, input tables and pretrained checkpoint files were replicated and hash checked. A 20,000-row baseline replay matched node2001 predictions exactly (maximum absolute difference 0.0). Authentication uses a temporary SSH control connection; its bootstrap password files were removed.
- Faris A100: retired from this campaign at the user request. Its capacity waiter was terminated and a STOP marker saved. No Kumo fit had started there. Campaign execution used nodes 2001, 2002 and 2003 and has now completed.

The frozen search deadline is **26 October 2026, 20:16 WIB**, and the campaign deadline is **27 October 2026, 00:16 WIB**. Resuming does not reset either deadline. Main-coordinator LIVE_STATE.json, per-fit RESULT/FAILURE records and reports are the authority for completion. The local folder is an evidence mirror and may lag the running nodes.

Per-fit outputs retain recipes, seeds, source and checkpoint hashes, predictions and metrics. No fitted weights or KV caches are saved. Execution uses shared original pretrained checkpoints, copied once to node2002 for local execution. The runner does not submit to Kaggle automatically; any submission needs a completed result review.

The earlier broader foundation-model and actuarial-model requests remain incomplete. Nineteen foundation-model default configurations completed fivefold evaluation in the separate model arena; the best corrected result was Kumo Medium at about 0.530400, below the matched maturity baseline of 0.530636. The recent loan-value, stress, credibility and binary-link actuarial batch also produced no gain over its matched two-seed baseline. Those negative studies are not proof that every model or feature family is exhausted.

The resource-only move is recorded in RESOURCE_MOVE.json and TOPOLOGY.json. It increases active GPU workers from two to four; the scientific protocol and original deadlines are unchanged. NODE2002_PORTABILITY_GATE.json records the exact replay check.
