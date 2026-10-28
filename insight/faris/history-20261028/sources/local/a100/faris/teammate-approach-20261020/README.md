# Teammate-guided feature research

The correct winning-method session and the additional manual feature session
are recorded in evidence.json. Their extracted dialogue retains source line
numbers. See FEATURE_RESEARCH.md for the current study; RESEARCH.md records the
previous interaction proposal, which was held before launch.

The live source master contains 118 complete entries including its control.
The new comparison separates feature information, support counts and head
regularization for V07 and V101, using 11 recipes across five original folds.

## Durable execution

Host: faris@labkc-comm1

Remote root:
/mnt/home/logika-faris-exploration/teammate-approach-20261020/feature-study

The service log can be followed with:

```bash
ssh faris@labkc-comm1 'tail -f /mnt/home/logika-faris-exploration/teammate-approach-20261020/feature-study/controller.log'
```

The last line is DONE=true with status=complete or status=failed. The systemd
service is logika-feature-attribution-20261020.service. Only its owned children
are managed. A fixed 24-hour absolute deadline is preserved in state.json.

Outputs: REPORT.md, result.json, scores.csv, feature_effects.csv, oof.npz,
bootstrap.npz and all 55 correction heads under folds/. Preflight reproduced
all ten published fold predictions to numerical precision and verified the
54 imported source artifacts. The study generates no competition submission.

These are reused development labels. A score of .54 has not been demonstrated
by the baseline or by the source session; the active objective remains open.
