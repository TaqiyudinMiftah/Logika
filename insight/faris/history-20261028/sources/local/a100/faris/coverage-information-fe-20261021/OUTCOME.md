# Fixed90-input coverage replacement: completed, not promoted

Allfivefolds completed with zero failed/unstarted jobs. Full70000-row metrics:

| Recipe | KS | AUC | Log loss |
|---|---:|---:|---:|
| Repaired V25 baseline | .5302185367 | .8470258500 | .2926003248 |
| Maturity-only feature candidate | .5306361042 | .8470560913 | .2925702718 |
| Four distinct coverage inputs replace four correlated prices | .5295852962 | .8471039033 | .2925283578 |

The tested replacement improved AUC/log loss but lostKS in4of5folds versus the original reference. Do not promote it for the KS objective. Its conditional simultaneous95percent delta intervals include zero for bothKS andAUC. It does not demonstrate that every coverage feature is useless, nor identify whether lost price-weighting information or weak added signal explains the result. That narrower causal attribution would require an independently justified ablation; none is automatically admitted.

FINAL_EVIDENCE.json independently verifies tied empiricalCDF/rankAUC for all70000rows, including the cold-replayed predictions. CONTENT_AUDIT.json passes all5serializedcontexts/source/checkpoint/seeds, rawcoldreplay, correctionobjectives, independently refittedheads and finalpredictions. Both raw and final cold replay are exact here. Every distinct context/head remains retained; unchanged fittedspline77models remain retained in maturity-placement-fe-20261021. No attentioncachecheckpoint was created or discarded.

Resource observation: kernelcampaignmemoryhighwater19448963072bytes (about18.11GiB), sampledownedGPUpeak6366MiB each. Sampling began after initial launch, soGPUobservations are not guaranteed lifetimepeaks. All other processes were preserved. Per-fold actual timings and source hashes are in runs/fold*/boundary.json.

Target.54 is unmet. Current feature-only lead remains the small maturity-aligned posterior replacement, whose uncertainty includes no gain. Confirmation remains closed and no submission/public-score probe was created. The original deadline remains unchanged.
