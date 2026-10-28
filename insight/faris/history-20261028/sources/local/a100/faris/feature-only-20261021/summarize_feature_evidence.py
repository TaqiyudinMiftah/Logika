"""Consolidate matched feature-only evidence; no new fitting or recipe selection."""
from pathlib import Path
import json,hashlib,time,csv
R=Path(__file__).resolve().parent;A=R.parent
STUDIES=[
 ('feature-only-20261021',['repayment_cashflow','bureau_trajectory','shopping_change'],'Three separately predefined six-field correction additions'),
 ('repayment-severity-fe-20261021',['matched_maturity','joint'],'Two replacement means; maturity control and joint-severity hypothesis'),
 ('maturity-placement-fe-20261021',['spline_only','both'],'Same maturity means placed into the fixed spline; both is primary'),
 ('loan-credibility-fe-20261021',['cluster'],'Two loan-cluster posterior means in fixed head and spline'),
 ('mature-denominator-fe-20261021',['eligible_denominators'],'Five Kumo/spline aliases changed; prior-filled missing history'),
 ('mature-ma[REDACTED_CREDENTIAL]',['mask_preserved'],'Same mature aliases; original missing-history masks restored'),
 ('coverage-information-fe-20261021',['coverage_information'],'Four distinct coverage signals replace four correlated price inputs'),
 ('source-support-fe-20261021',['support'],'Full account-month and scored-report counts in existing correction and spline'),
 ('account-conditional-fe-20261021',['conditional_account'],'Conditional correlated-account estimates replace five existing income aliases'),
 ('account-trait-fe-20261021',['account_trait'],'Persistent account random-intercept posteriors replace the same five income aliases'),
 ('shopping-periodicity-fe-20261021',['true','shuffled'],'One V80 temporal feature block with a matched within-customer shuffled-time control'),
 ('shopping-bands-fe-20261021',['bands','shuffled_bands'],'Ablate the four selected spectral extrema; retain six fixed band powers and support'),
]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rows=[];inventory=[]
 for study,recipes,hypothesis in STUDIES:
  root=A/study;result=root/'result.json';point=root/'POINT_RESULT.json'
  if result.exists():payload=read(result);source=result
  elif point.exists():payload=read(point);source=point
  else:
   status=root/'TRAINING_STATUS.json';stage=root/'STAGE_STATUS.json';build=root/'BUILD_COMPLETE.json'
   state=read(status)['status']if status.exists()else read(stage)['status']if stage.exists()else'features_ready_comparison_pending'if build.exists()else'no_terminal_result'
   inventory.append(dict(study=study,status=state,recipes=recipes,hypothesis=hypothesis));continue
  assert payload['historical_label_reuse']is True
  metrics=payload['metrics'];base=metrics.get('baseline',metrics.get('control90'));assert base is not None and abs(base['KS']-.5302185366909258)<1e-10 and abs(base['AUC']-.847025849979685)<1e-10
  fold=payload['fold_metrics'];basekey='baseline'if'baseline'in fold else'control90'
  for name in recipes:
   mm=metrics[name];assert mm['rows']==70000 and len(fold[name])==5
   delta=[fold[name][f]['KS']-fold[basekey][f]['KS']for f in range(5)]
   boundaries=list(root.glob(f'folds/{name}/fold*/boundary.json'))+list(root.glob(f'heads/{name}/fold*/boundary.json'))
   if not boundaries and study in ['maturity-placement-fe-20261021','loan-credibility-fe-20261021','mature-denominator-fe-20261021','mature-ma[REDACTED_CREDENTIAL]','coverage-information-fe-20261021']:boundaries=list(root.glob('fold[0-4]/boundary.json'))+list(root.glob('models/fold*/boundary.json'))+list(root.glob('runs/fold*/boundary.json'))
   times=[read(p).get('fit_seconds',read(p).get('seconds'))for p in boundaries];times=[t for t in times if t is not None]
   rows.append(dict(study=study,recipe=name,KS=mm['KS'],AUC=mm['AUC'],logloss=mm.get('logloss'),delta_KS=mm['KS']-base['KS'],delta_AUC=mm['AUC']-base['AUC'],KS_fold_wins=sum(x>0 for x in delta),KS_fold_ties=sum(x==0 for x in delta),worst_fold_delta_KS=min(delta),completed_outer_folds=5,OOF_rows=70000,source_sha256=sha(source),result_status=payload['status'],measured_fit_seconds_sum=sum(times)if times else None,fit_timing_records=len(times),hypothesis=hypothesis))
  inventory.append(dict(study=study,status=payload['status'],recipes=recipes,source=str(source),source_sha256=sha(source)))
 records=dict(status='snapshot',created=time.time(),baseline=dict(KS=.5302185366909258,AUC=.847025849979685),rows=rows,inventory=inventory,goal_KS=.54,goal_point_reached=any(x['KS']>=.54 for x in rows),confirmation_opened=False,historical_label_reuse=True,scope='Feature-only matched repaired V25 comparisons. Controls, repeated placements and attribution contrasts are not independent discoveries. No final promotion is inferred.',qualifications=['All results reuse development labels.','Conditional intervals omit retraining and adaptive selection uncertainty.','DPD snapshot timing remains unresolved.','Same-window priors and NaN changes are distinct feature semantics.','Unavailable timing records remain null, not zero.','Original public V25 score is user-reported context and is not used as a selection metric.'])
 (R/'FEATURE_EVIDENCE.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
 if rows:
  with(R/'FEATURE_EVIDENCE.csv').open('w',newline='')as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 lines=['# Matched feature-only evidence','', 'Frozen reference: KumoLarge90, original50k/6k/14kroles,5foldseed31027,headL2.003/strength.5,spline77,75:25probabilityblend. BaselineKS.5302185367/AUC.8470258500.','', '| Feature comparison | KS | AUC | Delta KS | Delta AUC | KS fold wins |','|---|---:|---:|---:|---:|---:|']
 for row in rows:lines.append(f"| {row['recipe']} | {row['KS']:.8f} | {row['AUC']:.8f} | {row['delta_KS']:+.8f} | {row['delta_AUC']:+.8f} | {row['KS_fold_wins']}/5 |")
 lines+=['','The maturity-only replacement in the fixedhead+spline is provisional; simultaneous conditional intervals include zero. Other point gains are not evidence of fresh private generalization. .54 remains the target. No confirmation partition is opened by this reporting script.','', 'The field builder and parity/poisoning/invariance evidence are in handoff/. Every distinct fitted model, transform and failed result remains in its original study. Research is bounded by the original deadline; this snapshot neither resets it nor admits another recipe.','', 'This table counts full five-fold recipe comparisons, including controls/placements as described; it does not equate these with independent discoveries or inflate counts with folds/retries. Null fit times are genuinely unavailable in this snapshot.']
 pending=[x for x in inventory if x['status']not in ['complete']]
 if pending:lines+=['','Pending or nonterminal: '+', '.join(x['study']+': '+x['status']for x in pending)]
 (R/'FEATURE_EVIDENCE.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(scored_recipe_comparisons=len(rows),pending=pending,goal_point_reached=records['goal_point_reached'])),flush=True)
if __name__=='__main__':main()
