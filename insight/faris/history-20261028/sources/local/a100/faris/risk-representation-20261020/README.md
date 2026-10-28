# Learned risk representation training

One focused hypothesis: give frozen Kumo the previous five-model champion's
cross-fitted risk estimate and model disagreement as two learned features.
Full scientific boundaries and controls are in RESEARCH.md.

The durable PC2 service is logika-ri[REDACTED_CREDENTIAL].service.
The most recent live check confirmed successful real-data gates, CUDA prediction
and saved-context replay, plus two active outer-fold workers. Workers first
construct honest teacher features on CPU, then use their assigned RTX 3060.
GPU idleness during teacher construction is expected; no foundation-model result
is reported before all five outer folds complete.

The original deadline is 2026-10-21T19:31:11.823748+07:00; restart preserves it.
The fixed main run comprises 150 teacher fits and ten Kumo contexts: two GPU
conditions across five outer folds, not 160 unrelated search hypotheses.
All fitted teachers and Kumo context caches are retained. Existing checkpoint
weights are referenced once by their verified hash.

Follow the run from any terminal:

```bash
ssh faris@labkc-comm1 'tail -f /mnt/home/logika-faris-exploration/ri[REDACTED_CREDENTIAL]/controller.log'
```

Detailed progress, including teacher completions, is available in events.jsonl:

```bash
ssh faris@labkc-comm1 'tail -f /mnt/home/logika-faris-exploration/ri[REDACTED_CREDENTIAL]/events.jsonl'
```

A terminal line with stage DONE and status complete indicates reports are saved.
Failure is explicit and preserves models. REPORT.md and result.json contain the
final scores and intervals. Nothing uploads to Kaggle or writes a submission.
The .54 OOF KS target remains unachieved at this handoff.

## Null-control isolation repair

The context permutation is restricted to each teacher held-out fold. This
ensures that neither a feature donor nor its recipient supplied a label to the
feature's teacher. The real five-fold assignment audit checked 250,000 context
recipient cases. It is retained in incidents/null-isolation/AUDIT.json. No Kumo
condition had started before the repair; all completed teachers were reused.
The campaign resumed with its original absolute deadline. An initial resealing
command failed and the hash gate correctly refused restart; the corrected seal
was verified before the successful resume. No unrelated process was stopped.
