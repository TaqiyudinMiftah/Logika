# Kumo temperature adaptation study

Launched on node380 on 2026-10-21. This is one new training recipe and a hardware-matched frozen control across all five original folds. The predeclared secondary comparison is their equal probability blend. No competition submission is generated.

Actual worker PID at launch: 693901; detached supervisor: 693900. The first verified stage is the fold-0 frozen full-context control, with 50,000 context rows and 20,000 correction/evaluation queries. GPU process memory was 2,982 MiB at that observation. These are activity facts, not predictive results.

Only 384 attention-temperature parameters are trainable; other pretrained parameters remain frozen and are hashed. Every fold retains both fitted caches, correction heads and adaptation checkpoints. Results are evaluated only after complete five-fold prediction coverage, including independent KS/AUC checks and paired uncertainty. Historical label reuse remains disclosed.

The original deadline is preserved in plan.json and enforced by the detached supervisor. Other users' processes are untouched. The worker has a 24-GiB allocator cap with live free-memory admission. Three actual gradient, exact-batch and fit/adapt/replay preflights passed before admission.

Read the live tail from the repository root:

```sh
python3 a100/profile.py node380 term 'tail -n 20 /home/taqiyudinmiftah/faris/kumo-temperature-adaptation-20261021/training.log'
```

Terminal files on node380: `SUPERVISOR.json`, `state.json`, `result.json` and `OOF.npz`. A running supervisor or completed preflight is not a completed performance evaluation.

## Complete outcome

Allfivefolds,1,000 adaptation episodes and10full fitted contexts are complete. The initial evaluator omitted targets of rows appearing only in correction roles; its source/logs are retained under incidents. The repaired evaluator and separate local SciPy/rank implementation agree to1e-12 on all70,000 original OOF rows. FINAL_STATUS.json and FINAL_AUDIT.json supersede the historical evaluator failure.

| Pipeline | KS | AUC |
|---|---:|---:|
| Matched frozen96 control |0.529222|0.847176|
|384-temperature adaptation|0.527331|0.845257|
|Fixed50/50 blend|0.528028|0.846596|

Adaptation lost KS in allfivefolds; its AUC change interval is[-0.002514,-0.001289] under the conditional paired bootstrap. The KS change interval includes zero. No candidate is promoted. Goal0.54 remains unmet. All models,20adapter checkpoints,ten fitted caches and unsuccessful evidence are retained. Historical labels are reused and these intervals do not undo adaptive research exposure.
