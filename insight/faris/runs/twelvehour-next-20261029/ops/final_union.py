"""Preregistered maximum across separately audited source-version studies."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import json,sys,time
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
import numpy as np
from common import load,dump,sha,utc,check_seal
from evaluate import paired_interval,metrics
clock=check_seal();plan=load('UNION_PLAN.json');assert plan['clock_sha256']==sha(R/'CLOCK.json')
candidates={};incomplete=[]
for namespace in plan['component_studies']:
 root=R if namespace=='.'else R/namespace
 if not(root/'FINAL_EVALUATION.json').exists()or not(root/'ARTIFACT_CONSISTENCY_AUDIT.json').exists():incomplete.append(namespace);continue
 audit=json.load(open(root/'ARTIFACT_CONSISTENCY_AUDIT.json'));assert audit['status']=='PASS'
 result=json.load(open(root/'FINAL_EVALUATION.json'));best=result.get('best_development')
 if not best:continue
 cohort=result['selected_cohort'];p=root/f'evaluation/cohorts/{cohort}/prediction.npz';assert sha(p)==result['prediction_sha256']
 candidates[namespace+'/'+cohort]=dict(namespace=namespace,cohort=cohort,KS=best['metrics']['KS'],prediction=str(p.relative_to(R)),prediction_sha256=sha(p),component_evaluation_sha256=sha(root/'FINAL_EVALUATION.json'),component_audit_sha256=sha(root/'ARTIFACT_CONSISTENCY_AUDIT.json'),source_seal_sha256=sha(root/'SOURCE_SEAL.json'))
assert candidates,'No complete audited candidate'
key=min(candidates,key=lambda n:(-candidates[n]['KS'],n));choice=candidates[key]
identity=dict(UTC=utc(),selected=key,candidate=choice,candidates=candidates,incomplete_components=incomplete,plan_sha256=sha(R/'UNION_PLAN.json'),clock_sha256=sha(R/'CLOCK.json'),rule='largest complete audited component/cohort KS, namespace/cohortlexicographictie',no_learned_mix=True,all70k_adaptive_development=True,component_conditional_results_already_exposed=True,union_conditional_assessment_not_yet_read=True)
frozen=R/'UNION_SELECTION.json'
if frozen.exists():assert load('UNION_SELECTION.json')['candidate']==choice
else:dump(identity,frozen)
y=np.load(R/'evaluation_inputs/target.npy');base=np.load(R/'evaluation_inputs/baseline.npz');previous=np.load(R/'evaluation_inputs/previous.npz')['prediction'];prediction=np.load(R/choice['prediction'])['prediction']
assert prediction.shape==y.shape==(70000,)and abs(metrics(y,prediction)['KS']-choice['KS'])<1e-12
baseline_interval=paired_interval(y,prediction,base['prediction'],base['fold'],clock)
previous_interval=paired_interval(y,prediction,previous,base['fold'],clock)
dump(dict(status='complete_conditional_union_assessment',UTC=utc(),choice=choice,selection_sha256=sha(frozen),baseline_interval=baseline_interval,previous_interval=previous_interval,target_054_point_reached=choice['KS']>=.54,independent_confirmation=False,public_gain_established=False,incomplete_components=incomplete,conditional_on_predictions_and_adaptive_selection=True,automatic_refit=False,submission_made=False,original_deadline_preserved=True),R/'UNION_EVALUATION.json')
