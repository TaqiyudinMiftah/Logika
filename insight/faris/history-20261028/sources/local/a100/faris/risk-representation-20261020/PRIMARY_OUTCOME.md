# Full primary OOF check

All 70,000 primary predictions are complete, each row appears exactly once,
and all five saved correction models replay exactly under the training thread
configuration. KS and AUC were independently recomputed.

| Pipeline | KS | AUC |
|---|---:|---:|
| Retained historical incumbent | 0.52929947 | 0.84720269 |
| Learned risk representation | 0.52758231 | 0.84690213 |

The primary is not an improvement and does not reach .54. Full shuffled-control
and cache-integrity reporting are still handled by the running controller.
This check does not assert their completion. All produced models remain.

The already queued training-size-matched query follow-up is one separate fixed
hypothesis. It waits for complete parent results and independent verification,
then reuses saved teachers and Kumo contexts. Both programs share the original
absolute deadline. No competition submission has been generated.
