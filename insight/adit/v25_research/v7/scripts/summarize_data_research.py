"""Aggregate matched item-data experiments without promoting noisy winners."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json

ROOT = Path(__file__).resolve().parents[2]
V = ROOT / 'v7'


def summarize():
    rows = []
    details = []
    plan = json.loads((V / 'data_fe_plan.json').read_text())
    for spec in plan:
        path = V / 'runs' / spec['id'] / 'RESULT.json'
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        assert r['status'] == 'COMPLETE' and r['frozen_Kumo_fraction'] == 1.0
        ci = r['confidence_interval']['delta_KS']
        support = r.get('comparison_to_support_control')
        if support:
            sc = support['confidence_interval']['delta_KS']
            if sc[0] > 0 and support['delta_AUC'] >= 0 and support['fold_wins'] >= 4:
                decision = 'PROMISING_CONDITIONAL_ON_REUSED_OOF'
            elif sc[1] < 0:
                decision = 'VALUE_FEATURES_HARM_RELATIVE_TO_SUPPORT'
            else:
                decision = 'UNIQUE_VALUE_NOT_ESTABLISHED'
        else:
            sc = [None, None]
            decision = 'AVAILABILITY_CONTROL_ONLY'
        rows.append(dict(experiment=spec['id'], label=spec['label'],
            KS=r['OOF']['KS'], AUC=r['OOF']['AUC'], delta_KS_V25=r['delta_KS'],
            delta_AUC_V25=r['delta_AUC'], fold_wins_V25=r['fold_wins'],
            delta_KS_V25_CI_low=ci[0], delta_KS_V25_CI_high=ci[1],
            delta_KS_support=support['delta_KS'] if support else None,
            delta_AUC_support=support['delta_AUC'] if support else None,
            fold_wins_support=support['fold_wins'] if support else None,
            delta_KS_support_CI_low=sc[0], delta_KS_support_CI_high=sc[1],
            parameters=r['folds'][0]['parameters'], decision=decision,
            submission=r['submission'], submission_sha256=r['submission_sha256']))
        segments_path = path.parent / 'ERROR_SEGMENTS.csv'
        segments = list(csv.DictReader(segments_path.open(encoding='utf-8-sig', newline='')))
        details.append(dict(experiment=spec['id'], formula=spec['formula'],
            uncertainty=r['confidence_interval']['caveat'],
            error_segments=segments, per_fold=r['folds'],
            overfitting_assessment=r['overfitting_assessment'],
            support_comparison=support))
    output = V / 'data_fe/reports'
    receipt = dict(updated_UTC=datetime.now(timezone.utc).isoformat(), completed=len(rows),
        planned=len(plan), public_incumbent='V25', public_KS=0.56822,
        no_automatic_public_promotion=True,
        interpretation='Support-control comparisons condition on existing fitted OOF; they do not correct repeated model search or establish hidden-test improvement.',
        experiments=rows, analysis=details)
    (output / 'BATCH_RESULTS.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    if rows:
        with (output / 'BATCH_RESULTS.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    text = ['# Matched tests of raw item-price information', '',
        f'Completed {len(rows)}/{len(plan)}. Kumo100% frozen; identical five outer folds.', '',
        'V243 is the common eight-input availability/count control. V244-V250 retain those same controls. Positive KS against V25 alone cannot establish information in price values.', '',
        '| Experiment | KS | AUC | Delta KS vs V25 | Delta KS vs support | Decision |',
        '|---|---:|---:|---:|---:|---|']
    for row in rows:
        delta = row['delta_KS_support']
        text.append(f"| {row['experiment']} | {row['KS']:.8f} | {row['AUC']:.8f} | {row['delta_KS_V25']:+.8f} | {delta:+.8f} | {row['decision']} |" if delta is not None else
                    f"| {row['experiment']} | {row['KS']:.8f} | {row['AUC']:.8f} | {row['delta_KS_V25']:+.8f} | control | {row['decision']} |")
    text += ['', 'The CSV records paired intervals, fold wins, model size and exact submission hashes. The JSON retains formulas, per-fold fits and held-out segment errors. Intervals exclude retraining, repeated selection and leaderboard uncertainty; these folds have already been used extensively.', '',
        'The new raw item statistics use only each application\'s prior completed orders. No new population reference or target enters feature construction. The full model still inherits unresolved historical snapshot provenance from the baseline. V25 remains the public incumbent.']
    (output / 'BATCH_RESULTS.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(json.dumps(dict(status='UPDATED', completed=len(rows), planned=len(plan))), flush=True)


if __name__ == '__main__':
    summarize()
