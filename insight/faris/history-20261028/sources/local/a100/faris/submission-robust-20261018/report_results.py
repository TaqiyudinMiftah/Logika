"""Summarize completed evidence without changing model choices."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text())


def main():
    p=read(ROOT/'benchmark_plan.json');r=read(ROOT/'benchmark_result.json');a=read(ROOT/'benchmark_assessment.json');v=read(ROOT/'verification.json');c=read(ROOT/'artifacts/complete.json');review=read(ROOT/'review_summary.json')
    fixed=c.get('fixed_comparison')
    z=np.load(ROOT/'evidence/fixed_selected_oof.npz' if fixed else ROOT/'evidence/nested_oof.npz')
    from scipy.stats import ks_2samp
    def ks(y,q):return float(ks_2samp(q[y==0],q[y==1]).statistic)
    baseline=[ks(z['target'][z['fold']==f],z['champion'][z['fold']==f]) for f in range(3)]
    selected=[ks(z['target'][z['fold']==f],z['prediction'][z['fold']==f]) for f in range(3)]
    winner=next(x for x in p['candidate_components'] if x['id']==c['addition']['component']);weight=c['addition']['weight'];ci=fixed['selected']['familywise_95ci'] if fixed else a['primary_mean_fold_uncertainty']['paired_stratified_95ci']
    model=winner['recipe']['model'] if winner['kind']=='historical' else winner.get('model',winner.get('configuration',{}).get('model'))
    rep=winner['recipe']['representation'] if winner['kind']=='historical' else winner['representation']
    rows='\n'.join(f'| {f+1} | {b:.9f} | {s:.9f} | {s-b:+.9f} |' for f,(b,s) in enumerate(zip(baseline,selected)))
    text=f'''# Final champion-anchored candidate

The delivered CSV contains **{1-weight:.0%} of the preserved 0.56188 public champion and {weight:.0%} of one {model} addition**. Its exact historical source is `{winner['original_id']}`. The prior 0.55353 replacement is not the anchor. Decision branch: `{c['decision_type']}`. Evidence status: **{c['quality_status']}**. This candidate is not assigned a new public score or promoted to proven champion.

The addition uses source blocks `{', '.join(rep['config']['sources'])}`, view `{rep['config']['view']}`, encoding `{rep['config']['encoding']}`, and reduction `{rep['config']['reducer']}`. All fit-dependent preprocessing and feature selection are retained with the model.

## What was reviewed

- {review['audited']:,} completed aligned prediction records passed independent score/row checks. These are records, not a claim of that many statistically independent experiments.
- Earlier fixed and nested ensemble scoreboards and the historical Logika inventory remain referenced with source hashes.
- The older depth-search ledger includes {review['depth_records']} attempts, of which {review['depth_verified']} complete aligned screening predictions were included in the audit.
- Descriptive comparisons of every aligned recipe used the same 5%/10%/20% additions. Only three compatible family representatives were frozen for the actual nested benchmark; the broader descriptive screen is not presented as validation.

## Primary nested procedure: failed consistency gate

All 70,000 labels participated in a three-fold outer cross-fit. For each outer training partition, separate inner fits selected among only the frozen additions and 5%/10% weights. The outer fold never supplied labels to that fold's selection. The adaptive procedure improved one outer fold and selected the champion fallback in the other two. Its mean-fold KS change was {r['primary_mean_fold_delta']:+.9f}, but it failed the predeclared two-fold consistency gate. **That failure is retained and not relabelled a success.**

## Fixed-recipe secondary benchmark

The same three configurations and two weights had already been frozen before benchmark fitting. Their six fixed combinations were compared after the primary result, using their complete fold-local outer predictions. No new model configuration or weight was introduced. The selected fixed combination improved all three folds; it is a different candidate from the variable adaptive selector. Selection among the six uses the highest Bonferroni-adjusted lower bootstrap bound among stable positive-fold contrasts. This is explicitly a **post-primary secondary analysis**, conditional on historical shortlist discovery.

| Outer fold | Champion KS | Selected fixed candidate KS | Change |
|---|---:|---:|---:|
{rows}
| Mean | {np.mean(baseline):.9f} | {np.mean(selected):.9f} | {np.mean(selected)-np.mean(baseline):+.9f} |

The fixed candidate's matched mean-fold improvement is **{np.mean(selected)-np.mean(baseline):+.9f} KS**, with conditional six-comparison familywise 95% interval **[{ci[0]:+.9f}, {ci[1]:+.9f}]**. Pooled OOF KS is separately **{ks(z['target'],z['prediction']):.9f} versus {ks(z['target'],z['champion']):.9f}**; it mixes outputs of separately fitted ensembles. These figures describe the selected fixed recipe, not the failed primary selector.

**Historical labels already influenced the feature definitions and descriptive shortlist. This is not untouched private validation.** The 4,000 paired bootstrap replicates stratify by outer fold and target; Bonferroni adjustment covers the six fixed comparisons conditionally. It does not erase historical adaptive-selection bias or full-refit variation. A positive point estimate does not guarantee a higher Kaggle score; if the interval includes losses, the candidate remains exploratory. No separate confirmation-set tuning or public leaderboard probing was used.

## Final fit and verification

The fixed secondary comparison selected `{winner['original_id']}` at weight `{weight}`. That one addition was fitted on all 70,000 labels without changing its feature definition, parameters or weight after fitting. The champion keeps its exact previous five fitted models and preprocessing. Reloading those models reproduced all 30,000 original predictions to the original CSV's rounding precision.

The final new model and all champion models were reloaded to replay every test row. The CSV matches the official sample order, has 30,000 unique IDs and finite probabilities in range. Verification independently reconstructed every fit/query split and label hash, and confirmed exclusion of outer-validation rows from all corresponding inner fits.

- New trained fit jobs retained: **{v['verified_fits']}** (nested validation fits plus the one final addition).
- Retained model/transformer files, including preserved champion artifacts: **{v['retained_model_files']}**.
- Submission SHA-256: `{c['submission_sha256']}`.
- Root promotion is recorded separately in `delivery.json`; no competition upload is performed.

## Reproduction

On PC2, from `/mnt/home/logika-faris-exploration/submission-robust-20261018`, using the preserved exploration environment:

```sh
../exploration-12h-20261017/.venv/bin/python benchmark.py
../exploration-12h-20261017/.venv/bin/python assess_benchmark.py
../exploration-12h-20261017/.venv/bin/python analyze_fixed.py
../exploration-12h-20261017/.venv/bin/python build_submission.py --fixed-secondary
../exploration-12h-20261017/.venv/bin/python build_submission.py --replay-only
../exploration-12h-20261017/.venv/bin/python verify_run.py
```

The original two-hour build deadline is preserved. Completed fits are hash-checked before reuse; the existing deadline is never reset. Inference-only replay remains available after the deadline.
'''
    (ROOT/'REPORT.md').write_text(text)
    print(str(ROOT/'REPORT.md'))


if __name__=='__main__':main()
