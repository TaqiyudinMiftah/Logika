# Midnight multiseed continuation — final result

The fixed four-seed bag reached pooled development **KS 0.5311092649**, below the previous two-seed reference **0.5319679800**. AUC was **0.8469012714**, versus reference **0.8468285962**. This run provides no supported KS improvement and does not reach 0.54.

The KS difference was **-0.0008587151**; the registered 2,000-replicate conditional 95% interval was **[-0.0021656647, 0.0006028221]**. Four folds declined and one improved. All 70,000 cases are adaptively reused development. The interval excludes model-refitting and historical-selection uncertainty; no untouched assessment or private leaderboard evidence exists.

## Work completed within the original clock

Admissions closed at **16:15 UTC**, model deadline **16:40 UTC**, final deadline **17:00 UTC / 00:00 WIB**. These clocks and all scientific sources were preserved. The terminal scientific audit finished at approximately **16:16 UTC**. Remaining registered work was not admitted after the fixed admission window.

| Successful fit packets | Count |
|---|---:|
| LocalGLM and CANN, two new seeds × five folds × validation/production | 40 |
| Kumo70, two new seeds × five folds × validation/production | 20 |
| Full-size reference qualification | 6 |
| Scalar and coefficient Transformer probes, one fold × validation/production | 4 |
| Total | **70** |

Of 686 registered slots, 616 were not admitted: 540 additional main-branch slots and 76 Transformer slots. Three extra worker attempts failed before scientific training; their diagnostics and the separate setup issues are retained in STARTUP_ISSUES.json and private evidence. Successful, failed, partial and excluded work is recorded separately.

## Frozen pipeline and boundaries

The two new seeds were **10103 and 20076**, combined equally with original seeds **2026 and 7731**. Inclusion was frozen from complete artifact availability before new outer scores. Every included seed has all three branches, five folds and both phases. Individual seed metrics were calculated afterward as diagnostics and did not filter inclusion.

Per member, the registered prediction is 90% LocalGLM meta-adaptive blend plus 10% positive-logit anchor/CANN/Kumo70 fusion with penalty 0.003. Validation has 50,000 context / 6,000 correction / 14,000 outer rows per fold. CPU encoders and stopping roles were reconstructed at 42,500 / 7,500. Production trains on all 70,000 labels, predicts 30,000 test rows and uses the existing phase-specific production feature packets. Its 59,500 / 10,500 stopping split is not independent assessment.

Both Transformer heads completed their full-size probe pairs. The remaining four-seed/five-fold grid failed the conservative runtime gate. No Transformer achieved complete OOF coverage, entered the final blend, or received an outer-performance claim. The implementation is FT-inspired; it is not an exact BERTNET reproduction.

## Verification and retention

PRE_EVALUATION_AUDIT verifies all 70 successful output hashes, source and packet hashes, row identities, seeds, disjoint roles, inner/stopping boundaries and finite predictions. CPU reference replay stayed below 5e-6; strict Kumo same-cache replay stayed below 1e-7. Kumo validation's cross-device reference diagnostic differed by up to 0.0014764071; production reference replay was exact. The preregistered cross-device diagnostic exception was preserved.

The separate arithmetic implementation rebuilt the previous reference within 3.34e-16 and the new validation/test blend exactly. It independently recomputed baseline/reference/candidate KS, fold differences, seed diagnostics, the 2,000 bootstrap replicates and exact CSV round-trip alignment. This is an arithmetic audit, not independent research confirmation.

The held CSV is `evaluation/paired_equal_seedbag/submission_HELD.csv`, SHA256 **a1589bd807ffc21f90d6df90535d56dd39e69ec60f8cf0b9b13f66dab101a1e0**, with 30,000 unique IDs in official sample order. It has not been uploaded. Previous Kaggle receipt 57806062 / public KS 0.56716 remains the latest actual submission. No fallback or extra upload was performed.

All owned worker processes and units ended; both lab2003 GPUs returned to 1 MiB / 0% utilization. Fresh authenticated checks found all eleven Colab servers empty. The access relay remains available. Actual environment manifests, parameters, sources, predictions, logs and failures are retained privately; newly fitted weights and KV caches were not serialized, as requested. Full feature packets remain on the campaign hosts. Compact-archive and local-collection receipts state their exact coverage; the optional bulk feature transfer to node380 was deferred after its measured throughput exceeded the remaining finalization window.
