"""Fixed seed pairs on one owned CPU backend; full-size50k/70k pilots gate rest."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
import argparse,concurrent.futures,json,subprocess,sys,time,traceback
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'source'))
from common import load,dump,sha,utc,check_seal
a=argparse.ArgumentParser();a.add_argument('--owner',required=True);owner=a.parse_args().owner
C=check_seal();plan=load('EXECUTION_PLAN.json')['owners'][owner];pairs=plan['pairs'];P=Path(plan['production_root']);design=next(iter(load('CONFIG.json')['designs']));started=time.time()
def one(pair,pilot=False):
 assert time.time()<C['admission_end_epoch'];seed,fold=pair['seed'],pair['fold'];task=pair['task'];log=R/'logs'/f'{task}.log';log.parent.mkdir(exist_ok=True)
 dev=R/f'runs/{task}/attempt1';prod=P/f'predictions/selected_localglm-cann/s{seed}/fold{fold}'
 assert not dev.exists()and not prod.exists(),'No implicit retry or duplicate production fit'
 t=time.time()
 with log.open('ab')as f:subprocess.run([sys.executable,'-u',str(R/'source/worker.py'),'--task',task]+(['--pilot']if pilot else[]),stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(1,min(950,C['model_deadline_epoch']-time.time()-10)))
 v=load(dev/'RESULT.json');assert sha(dev/'START.json')==v['start_sha256']and sha(dev/'prediction.npz')==v['prediction_sha256'];devtime=time.time()-t;t=time.time()
 assert time.time()<C['admission_end_epoch']
 with log.open('ab')as f:subprocess.run([sys.executable,'-u',str(P/'source/refit_selected.py'),'--namespace','localglm-cann','--seed',str(seed),'--fold',str(fold)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=max(1,min(1250,C['model_deadline_epoch']-time.time()-10)))
 v=load(prod/'RESULT.json');assert sha(prod/'START.json')==v['start_sha256']and sha(prod/'prediction.npz')==v['prediction_sha256']
 return dict(seed=seed,fold=fold,dev_seconds=devtime,production_seconds=time.time()-t,production_prediction_sha256=v['prediction_sha256'])
results=[];failures=[]
try:
 pilot=one(pairs[0],True);results.append(pilot);project=(len(pairs)-1)*(pilot['dev_seconds']+pilot['production_seconds'])*1.5/plan['slots'];qualified=time.time()+project+300<C['model_deadline_epoch']
 g=dict(status='PASS'if qualified else'FAIL_RESOURCE_FEASIBILITY',UTC=utc(),owner=owner,clock_sha256=sha(R/'CLOCK.json'),config_sha256=sha(R/'CONFIG.json'),source_seal_sha256=sha(R/'SOURCE_SEAL.json'),contract_sha256=sha(R/'VALIDATION_CONTRACT.json'),pilot=pilot,projected_remaining_seconds=project,safety_factor=1.5,finish_margin_seconds=300,score_used=False)
 dump(g,R/('GATE_FAMILY_'+load('CONFIG.json')['designs'][design]['admission_group']+'.json'));dump(g,R/'PAIR_GATE.json')
 if qualified:
  with concurrent.futures.ThreadPoolExecutor(max_workers=plan['slots'])as pool:
   fs={pool.submit(one,p):p for p in pairs[1:]}
   for f in concurrent.futures.as_completed(fs):
    try:results.append(f.result())
    except Exception:failures.append(dict(pair=fs[f],trace=traceback.format_exc(),no_retry=True))
    dump(dict(UTC=utc(),owner=owner,completed_pairs=len(results),planned_pairs=len(pairs),failed_pairs=len(failures)),R/'BACKEND_STATE.json')
except Exception:failures.append(dict(pair=pairs[0],trace=traceback.format_exc(),no_retry=True))
finally:
 dump(dict(status='terminal',UTC=utc(),owner=owner,results=results,failures=failures,completed_pairs=len(results),planned_pairs=len(pairs),source_and_recipe_unchanged=True),R/'BACKEND_FINAL.json')
