# Final Logika training and blending results

All work is bounded by 30 October 2026, 20:00 WIB (13:00 UTC). No Kaggle upload or new full-data base-model refit was performed in this continuation. Earlier study clocks and final selections remain unchanged.

| Assessment | Complete eligible recipes | Best pooled development KS | Result |
|---|---:|---:|---|
| Cross-network base-model campaign | 520 complete designs / 2,080 composition cohorts | 0.5316414268134972 | Equal to the previous reference |
| Historical probability, linear, neural and greedy blends | 104 | 0.5307384643313067 | Below the reference |
| Additional pairwise-ranking GPU meta-models | 16 | 0.5302751348592203 | Below the reference |
| Gated-expert GPU meta-models | 16 | 0.5306168643404208 | Below the reference |
| Blends with matching saved validation and fresh-production test predictions | 48 | **0.5319679799870825** | Higher point estimate; uncertainty includes no improvement |

The final CSV exports follow the official sample row order. EXPORT_AUDIT verifies every ID/probability pair is preserved exactly after parsing, and the original candidate files remain unchanged. The selected paired recipe is `current_fusion_cann_kumo70_p0.003_w0.1`: 90% of the current LocalGLM-CANN prediction plus 10% of the saved CANN/Kumo70 fusion with its previously declared 0.003 meta penalty. Its held test CSV has SHA256 `f14cb4555575016828f656d722aa9cb9f9d85217bb3518ba18eddd59c7b10f62`.

The paired recipe improves the reused-development point by 0.000326553173585209. Its 2,000-replicate conditional 95% interval for that difference is **[-0.00048090211658640135, 0.0008210384593371835]**. Two folds improve, two decline, and one ties. This does not establish a supported gain, reach 0.54, or establish a public/private leaderboard improvement. All 70,000 cases have been adaptively reused; no untouched audit exists. These intervals also exclude candidate-selection and refitting uncertainty.

## Execution and provenance

The base campaign completed **5,208 of 6,160 planned fits**, with zero predictive failures and one provider interruption. Exactly 520 designs have both seeds and all five folds. Eight successful fits belong to incomplete designs; one attempt was interrupted and another 951 planned slots were not completed. Incomplete designs do not receive cohort scores. An independent terminal audit verified all raw hashes, row roles, seed identities, training boundaries, all 2,080 cohort means and SciPy KS scores, the baseline, the reference and the frozen final choice.

The history bank contains 1,925 compatible complete pipelines from 29 study/family groups. Coverage is restricted to accessible, hash-verified artifacts using the matching validation protocol. It is not a universal claim about every historical experiment, archived ZIP or inaccessible host. Two linear blend recipes were excluded because their optimizers did not converge; their evidence is preserved rather than classified as a predictive negative.

There were 780 new meta-training/selection fit packets: 120 linear, 160 probability-based GPU neural, 180 greedy fold selections, 160 ranking-based GPU neural, and 160 gated-expert GPU neural. Fixed arithmetic recipes and the 48 paired recompositions are not new independent base-model fits. The paired study reconstructed eleven previously declared recipes from the saved 30 fresh production members; it did not retrain those members.

Both authorized RTX 3060 GPUs on node2003 were used. All ranking and gated-expert fits passed the original strict 5e-6 batch replay gate. The separate gated study was preregistered at commit ad7fbcc9dcaa4202d1c6d5cc050f134ba60a30e8 before fitting and completed with 16 audited recipes; it did not change earlier frozen selections. Node680 login succeeded, but its A100 80GB was already at 100% utilization on another user's workload; that workload was untouched. The authorized account contributed 45 unique saved test vectors. Its older three-fold validation protocol is incompatible with this run's five folds, so those local scores were not combined. Our inventory kernel was explicitly shut down (HTTP204).

Fresh authenticated checks confirmed zero active assignments and zero usage rate on all eleven Colab accounts. Account2's observed balance changed from 298.62 to 253.57 CU during this campaign, a 45.05 CU decrease; this is an account-level observation, not a per-fit bill.

## Test-only comparisons and delivery

There are 164 unique schema-valid test vectors across the accessible local, canonical and node680 inventories. One constant submission template was excluded as a non-model, leaving **163 eligible model vectors** and 13,203 pair comparisons. Their minimum rank correlation is about 0.9212. Uniform probability, logit and rank blends were saved as explicitly unscored held CSVs. Their hidden-test KS is unknown; disagreement alone does not establish useful complementarity.

The selected scored CSV and three unscored CSVs are retained locally under `submissions_held/final-blends-20261030/`. Raw predictions, labels, credentials and fitted weights are excluded from Git. Sanitized source, configurations, seeds, aggregate metrics, failures, exclusions and artifact hashes are published to the shared repository. A separate compact backup on host42 is verified by file hashes; its receipt explicitly excludes the large input bank, which remains on canonical and node2003 storage.

## Subsequent explicitly authorized submission

After the held research delivery, the user explicitly requested submission of this paired blend. Kaggle receipt **57806062**, dated 30 October 2026 12:56:57 UTC (19:56:57 WIB), confirms COMPLETE status and **public KS 0.56716**. Private score is unavailable. The uploaded CSV SHA256 is `f14cb4555575016828f656d722aa9cb9f9d85217bb3518ba18eddd59c7b10f62`. No new training or fallback was used. This is below the separately verified public V25 score 0.56822; the higher local development point did not translate into public improvement. Frozen research audits retain their original no-upload status; SUBMISSION_DELIVERY.json records this later authorized action.
